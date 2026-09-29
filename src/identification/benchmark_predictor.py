"""
Open-loop predictor benchmark: model identified by Concurrent Learning (Approach C) vs the
model identified by the black-box Lb-LSTM (Approach A).

Protocol (identical for every model; only the adaptation law differs):
    1. Training data: one weak-excitation (non-PE) trajectory of the rig (encoder noise and
       quantization on). Every observer sees the same measurements and inputs, uses the same
       input normalization (s, mu), the same LSTM width L, and the same random gate
       initialization (same seed).
    2. Observers:
         * "A: Lb-LSTM"          Approach A, instantaneous law on all weights (gates + readout).
         * "C: CL-Lb-LSTM"       Approach C, dual-loop concurrent-learning law (readout: both
                                 loops; gates: CL loop).
         * "Std law (ablation)"  Approach C's observer with the CL loop disabled, i.e. the
                                 instantaneous law on the readout of a frozen random LSTM.
    3. The adapted weights are frozen at the end of the run and the LSTM is decoupled from the
       observer feedback (Approach A's `LbLSTMDigitalTwin`, reused unchanged).
    4. Every twin is driven open loop by inputs absent from the training data
       (`generalization_test_suite`) and scored against the true plant. Primary metrics:
         * one-step NMSE of Phi_hat on the true states (model error only),
         * short-horizon NMSE: the twin is re-initialized from the true state (and the
           teacher-forced LSTM memory) every `horizon` seconds and rolled out open loop over
           that window; positions are scored over all windows.
       Secondary diagnostic: free-running NMSE over the whole 8-15 s test. Around the hanging
       equilibrium the plant is only marginally stable (linearized poles 0, -0.019,
       -0.020 +/- 2.54j: free cart = double integrator, pendulum damping ratio ~0.008), so any
       model error integrates into cart drift and pendulum phase error that no feedback
       removes; long free runs diverge for every model and rank them poorly.
       NMSE = MSE / Var(truth): 1 is the score of predicting the mean.

CLI:  python -m src.identification.benchmark_predictor --seeds 3 [--scenario decaying]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.concurrent_learning.scenarios import (
    DT,
    ZETA_INPUT_SCALE,
    generalization_test_suite,
    measurement_offset,
    simulate_weak_excitation,
)
from src.identification.extract_model import FrozenLbLSTM, LbLSTMDigitalTwin, STATE_NAMES
from src.observers.blackbox_lstm import LbLSTMObserver, LbLSTMObserverConfig
from src.observers.cl_lstm_observer import CLLbLSTMObserver, CLLbLSTMObserverConfig
from src.simulation.open_loop import OpenLoopTrajectory, simulate_open_loop

MODEL_A = "A: Lb-LSTM"
MODEL_C = "C: CL-Lb-LSTM"
MODEL_STD = "Std law (ablation)"
MODEL_C_FROZEN = "C, frozen gates"  # CL on the readout only: the case the convergence proof covers exactly


# ---------------------------------------------------------------------- configurations
def approach_a_config(offset: Tuple[float, ...], seed: int) -> LbLSTMObserverConfig:
    """Approach A's validated configuration (feature/approach-a-blackbox-lstm)."""
    return LbLSTMObserverConfig(
        input_scale=ZETA_INPUT_SCALE, input_offset=offset, sign_mode="sgn", tanh_eps=0.005, seed=seed,
    )


def approach_c_config(offset: Tuple[float, ...], seed: int, **overrides: object) -> CLLbLSTMObserverConfig:
    """Approach C: Approach A's observer gains plus the concurrent-learning loop."""
    base = approach_a_config(offset, seed)
    cfg = CLLbLSTMObserverConfig(**{k: getattr(base, k) for k in (
        "input_scale", "input_offset", "sign_mode", "tanh_eps", "seed")})
    return replace(cfg, **overrides) if overrides else cfg  # type: ignore[arg-type]


# ---------------------------------------------------------------------- training
@dataclass
class TrainingTrace:
    """Observer history on the training trajectory (decimated where noted)."""
    t: np.ndarray               # (K,) decimated times
    theta_h: np.ndarray         # (K, L n) readout weights
    lambda_min: np.ndarray      # (K,) lambda_min(Omega) of the history stack (0 for Approach A)
    stack_size: np.ndarray      # (K,)
    fixed_point_dist: np.ndarray  # (K,) ||W_h - W_H||_F, W_H = stack least squares (nan if rank deficient)
    velocity: np.ndarray        # (N, 2) full-rate velocity estimates
    features: np.ndarray        # (N, L) full-rate LSTM hidden outputs h(t) (CL observers only)
    phi_hat: np.ndarray         # (N, 2) full-rate Phi_hat


@dataclass
class TrainedModel:
    name: str
    frozen: FrozenLbLSTM
    trace: TrainingTrace
    observer: LbLSTMObserver


def run_observer(obs: LbLSTMObserver, traj: OpenLoopTrajectory, record_every: int = 10) -> TrainingTrace:
    """Runs an observer over a recorded trajectory and logs its adaptation."""
    y0 = traj.measurements[0]
    obs.reset(np.array([y0[0], 0.0, y0[1], 0.0]))
    N = len(traj.t)
    L = obs.config.hidden_dim
    h_slice = obs.layout.block_slice("h")
    is_cl = isinstance(obs, CLLbLSTMObserver)

    K = (N + record_every - 1) // record_every
    t_rec = np.zeros(K)
    th_rec = np.zeros((K, h_slice.stop - h_slice.start))
    lam_rec = np.zeros(K)
    size_rec = np.zeros(K, dtype=int)
    fp_rec = np.full(K, np.nan)
    vel = np.zeros((N, 2))
    feats = np.zeros((N, L)) if is_cl else np.zeros((0, L))
    phi = np.zeros((N, 2))

    for k in range(N):
        est = obs.update(traj.measurements[k], traj.u_cmd[k], DT)
        vel[k] = est[[1, 3]]
        phi[k] = obs.phi_hat
        if is_cl:
            assert isinstance(obs, CLLbLSTMObserver)
            feats[k] = obs.h_features
        if k % record_every == 0:
            j = k // record_every
            t_rec[j] = traj.t[k]
            th_rec[j] = obs.theta[h_slice]
            if is_cl:
                assert isinstance(obs, CLLbLSTMObserver)
                lam_rec[j] = obs.lambda_min()
                size_rec[j] = len(obs.stack)
                W_H = obs.stack_least_squares()
                if W_H is not None:
                    fp_rec[j] = float(np.linalg.norm(obs.W_h - W_H))
    return TrainingTrace(t_rec, th_rec, lam_rec, size_rec, fp_rec, vel, feats, phi)


def _freeze(obs: LbLSTMObserver, t_now: float) -> FrozenLbLSTM:
    if isinstance(obs, CLLbLSTMObserver):
        return obs.freeze()
    return FrozenLbLSTM(theta=np.copy(obs.theta), config=obs.config, t_freeze=float(t_now))


def train_models(
    traj: OpenLoopTrajectory,
    seed: int,
    include: Tuple[str, ...] = (MODEL_A, MODEL_C, MODEL_STD),
    c_overrides: Dict[str, object] | None = None,
) -> Dict[str, TrainedModel]:
    """Trains every requested observer on the same trajectory and freezes it at the end."""
    offset = measurement_offset(traj)
    overrides = c_overrides or {}
    factories: Dict[str, Callable[[], LbLSTMObserver]] = {
        MODEL_A: lambda: LbLSTMObserver(approach_a_config(offset, seed)),
        MODEL_C: lambda: CLLbLSTMObserver(approach_c_config(offset, seed, **overrides)),
        MODEL_STD: lambda: CLLbLSTMObserver(approach_c_config(offset, seed, **{**overrides, "cl_enabled": False})),
        MODEL_C_FROZEN: lambda: CLLbLSTMObserver(approach_c_config(offset, seed, **{**overrides, "gamma_cl_gates": 0.0})),
    }
    out: Dict[str, TrainedModel] = {}
    for name in include:
        obs = factories[name]()
        trace = run_observer(obs, traj)
        out[name] = TrainedModel(name, _freeze(obs, float(traj.t[-1] + DT)), trace, obs)
    return out


# ---------------------------------------------------------------------- evaluation
SHORT_HORIZON = 0.5   # [s] window of the short-horizon prediction metric


def _mse_nmse(truth: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    mse = float(np.mean((truth - pred) ** 2))
    var = float(np.var(truth))
    return mse, mse / var if var > 1e-12 else float("nan")


def teacher_forced_memory(
    twin: LbLSTMDigitalTwin, truth: OpenLoopTrajectory, dt: float = DT, t_settle: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """LSTM memory (c_hat, h_hat) integrated along the true states, shape (N, L) each."""
    u_seq = truth.u_cmd.reshape(-1, 1)
    x1, x2 = truth.positions, truth.velocities
    c_hat, h_hat = twin.settle_memory(x1[0], x2[0], u_seq[0], t_settle, dt)
    N, L = len(u_seq), twin.layout.hidden_dim
    C, Hm = np.zeros((N, L)), np.zeros((N, L))
    for k in range(N):
        C[k], Hm[k] = c_hat, h_hat
        cache = twin._lstm(x1[k], x2[k], u_seq[k], c_hat, h_hat)
        c_hat = c_hat + dt * twin.config.b_c * (cache.c - c_hat)
        h_hat = h_hat + dt * twin.config.b_h * (cache.h - h_hat)
    return C, Hm


def short_horizon_rollout(
    twin: LbLSTMDigitalTwin, truth: OpenLoopTrajectory, horizon: float = SHORT_HORIZON, dt: float = DT,
) -> np.ndarray:
    """
    Piecewise open-loop prediction (N, 4): every `horizon` seconds the twin restarts from the
    true state and the teacher-forced memory, then runs open loop (RK4) until the next restart.
    """
    n = twin.n
    C, Hm = teacher_forced_memory(twin, truth, dt)
    u_seq = truth.u_cmd.reshape(-1, 1)
    N = len(u_seq)
    H = max(1, int(round(horizon / dt)))
    pred = np.zeros((N, 2 * n))
    for k0 in range(0, N, H):
        z = np.concatenate((truth.positions[k0], truth.velocities[k0], C[k0], Hm[k0]))
        for k in range(k0, min(k0 + H, N)):
            pred[k] = np.column_stack((z[:n], z[n:2 * n])).ravel()
            u = u_seq[k]
            k1 = twin._vector_field(z, u)
            k2 = twin._vector_field(z + 0.5 * dt * k1, u)
            k3 = twin._vector_field(z + 0.5 * dt * k2, u)
            k4 = twin._vector_field(z + dt * k3, u)
            z = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return pred

@dataclass
class PredictionResult:
    model: str
    test: str
    truth: OpenLoopTrajectory
    rollout_state: np.ndarray       # (N, 4) free-running twin
    short_horizon_state: np.ndarray # (N, 4) piecewise rollouts restarted every SHORT_HORIZON s
    accel_one_step: np.ndarray      # (N, 2) Phi_hat on the true states
    metrics: Dict[str, float]


def evaluate_model(
    name: str, frozen: FrozenLbLSTM, test: str, truth: OpenLoopTrajectory, dt: float = DT,
) -> PredictionResult:
    twin = LbLSTMDigitalTwin(frozen)
    rollout = twin.simulate(truth.u_cmd, truth.state[0], dt)
    short = short_horizon_rollout(twin, truth, SHORT_HORIZON, dt)
    accel_tf = twin.teacher_forced_acceleration(truth.positions, truth.velocities, truth.u_cmd, dt)
    metrics: Dict[str, float] = {}
    for j, s in enumerate(("x_ddot", "theta_ddot")):
        _, metrics[f"one-step NMSE {s}"] = _mse_nmse(truth.accel[:, j], accel_tf[:, j])
    for j, s in ((0, "x"), (2, "theta")):
        _, metrics[f"short-horizon NMSE {s}"] = _mse_nmse(truth.state[:, j], short[:, j])
    for j, s in enumerate(STATE_NAMES):
        metrics[f"rollout MSE {s}"], metrics[f"rollout NMSE {s}"] = _mse_nmse(truth.state[:, j], rollout.state[:, j])
    return PredictionResult(name, test, truth, rollout.state, short, accel_tf, metrics)


def simulate_test_suite(dt: float = DT) -> List[Tuple[str, OpenLoopTrajectory]]:
    """True-plant responses to the unseen inputs (no actuator dead-zone, as in training)."""
    return [
        (name, simulate_open_loop(u_fn, horizon, dt, x0, apply_actuator_effects=False))
        for name, u_fn, horizon, x0 in generalization_test_suite()
    ]


def evaluate_models(
    models: Dict[str, FrozenLbLSTM], tests: List[Tuple[str, OpenLoopTrajectory]] | None = None,
) -> List[PredictionResult]:
    tests = tests if tests is not None else simulate_test_suite()
    return [evaluate_model(name, frozen, test, truth) for name, frozen in models.items() for test, truth in tests]


def results_frame(results: List[PredictionResult], seed: int | None = None) -> pd.DataFrame:
    rows = []
    for r in results:
        row: Dict[str, object] = {"model": r.model, "test": r.test}
        if seed is not None:
            row["seed"] = seed
        row.update(r.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def run_predictor_benchmark(
    seeds: List[int], scenario: str = "decaying", t_final: float = 60.0,
    c_overrides: Dict[str, object] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, TrainedModel], List[PredictionResult]]:
    """
    Multi-seed benchmark. Seed s sets the gate initialization of every model and the
    encoder-noise realization of the training trajectory. Returns the per-seed table plus the
    trained models and prediction results of the first seed (for plotting).
    """
    tests = simulate_test_suite()
    frames: List[pd.DataFrame] = []
    first_models: Dict[str, TrainedModel] = {}
    first_results: List[PredictionResult] = []
    for i, seed in enumerate(seeds):
        traj, _ = simulate_weak_excitation(scenario, t_final, noise_seed=42 + seed)
        models = train_models(traj, seed, c_overrides=c_overrides)
        results = evaluate_models({k: m.frozen for k, m in models.items()}, tests)
        frames.append(results_frame(results, seed))
        if i == 0:
            first_models, first_results = models, results
    return pd.concat(frames, ignore_index=True), first_models, first_results


PRIMARY_METRICS = [
    "one-step NMSE x_ddot", "one-step NMSE theta_ddot", "short-horizon NMSE x", "short-horizon NMSE theta",
]


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Median over seeds of the headline metrics, per model and test."""
    cols = PRIMARY_METRICS
    return df.groupby(["test", "model"], sort=False)[cols].median().reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Open-loop predictor benchmark: Approach C vs Approach A.")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--scenario", choices=("decaying", "single_freq"), default="decaying")
    parser.add_argument("--t-final", type=float, default=60.0)
    parser.add_argument("--out", type=Path, default=Path("results/approach_c/predictor_benchmark.csv"))
    args = parser.parse_args()

    df, _, _ = run_predictor_benchmark(list(range(args.seeds)), args.scenario, args.t_final)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(summary_table(df).to_string(index=False))
    print(f"\nPer-seed results: {args.out}")


if __name__ == "__main__":
    main()
