"""
System identification stage for the black-box Lb-LSTM observer (Approach A).

1. Freeze the adapted weights theta_hat once the observer has settled (t >= t_freeze).
2. Decouple the LSTM from the observer error feedback: drop the auxiliary filter
   (p, nu), the robust term k_s sgn(e), chi and the adaptation law. What remains is the
   autonomous forward model (digital twin)
       x1_dot    = x2
       x2_dot    = Phi_hat(zeta, c_hat; theta_frozen)
       c_hat_dot = -b_c c_hat + b_c (f * c_hat + i * c*)
       h_hat_dot = -b_h h_hat + b_h (o * sigma_c(c))
   with zeta = [s * ([x1, x2, u] - mu), h_hat, 1] built from the twin's *own* states.
3. Drive the twin open loop with inputs it never saw during adaptation (step doublet,
   chirp) and compare against the true plant: state-trajectory MSE of the free-running
   rollout, plus the one-step (teacher-forced) acceleration MSE.

CLI:  python -m src.identification.extract_model --weights results/approach_a/lblstm_frozen.npz
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Dict, List
import numpy as np

from src.adaptation.jacobian_engine import LSTMCache, LSTMWeightLayout, lstm_forward
from src.observers.blackbox_lstm import LbLSTMObserver, LbLSTMObserverConfig
from src.plant.pendulum_plant import PendulumParameters
from src.simulation.open_loop import OpenLoopTrajectory, simulate_open_loop

STATE_NAMES = ("x", "x_dot", "theta", "theta_dot")


# ---------------------------------------------------------------------- frozen weights
@dataclass
class FrozenLbLSTM:
    """Snapshot of an adapted Lb-LSTM: weights plus the architecture needed to evaluate it."""
    theta: np.ndarray
    config: LbLSTMObserverConfig
    t_freeze: float

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            theta=self.theta,
            t_freeze=self.t_freeze,
            config_json=json.dumps(asdict(self.config)),
        )

    @classmethod
    def load(cls, path: str | Path) -> FrozenLbLSTM:
        data = np.load(path)
        raw = json.loads(str(data["config_json"]))
        known = {f.name for f in fields(LbLSTMObserverConfig)}
        cfg_kwargs = {k: v for k, v in raw.items() if k in known}
        for key in ("input_scale", "input_offset"):
            if cfg_kwargs.get(key) is not None:
                cfg_kwargs[key] = tuple(cfg_kwargs[key])
        return cls(
            theta=np.asarray(data["theta"], dtype=np.float64),
            config=LbLSTMObserverConfig(**cfg_kwargs),
            t_freeze=float(data["t_freeze"]),
        )


def freeze_observer(observer: LbLSTMObserver, t_now: float, min_settle_time: float = 20.0) -> FrozenLbLSTM:
    """Copies the observer's current weights. Refuses to freeze before the settling time."""
    if t_now < min_settle_time:
        raise ValueError(
            f"Observer has not settled: t = {t_now:.2f} s < {min_settle_time:.2f} s."
        )
    return FrozenLbLSTM(theta=np.copy(observer.theta), config=observer.config, t_freeze=float(t_now))


# ---------------------------------------------------------------------- digital twin
@dataclass
class TwinTrajectory:
    t: np.ndarray
    state: np.ndarray   # Shape (N, 2n): interleaved [x1_0, x2_0, x1_1, x2_1, ...]
    accel: np.ndarray   # Shape (N, n): Phi_hat along the rollout


class LbLSTMDigitalTwin:
    """Open-loop forward model built from frozen Lb-LSTM weights (no observer feedback)."""

    def __init__(self, frozen: FrozenLbLSTM) -> None:
        cfg = frozen.config
        self.config = cfg
        self.theta = np.copy(frozen.theta)
        self.n = cfg.n_coords
        self.m = cfg.n_inputs
        L = cfg.hidden_dim
        self.layout = LSTMWeightLayout(2 * self.n + self.m + L + 1, L, self.n)
        if self.theta.shape != (self.layout.n_params,):
            raise ValueError("Frozen theta does not match the configured architecture.")
        self.input_scale = (
            np.ones(2 * self.n + self.m) if cfg.input_scale is None
            else np.asarray(cfg.input_scale, dtype=np.float64)
        )
        self.input_offset = (
            np.zeros(2 * self.n + self.m) if cfg.input_offset is None
            else np.asarray(cfg.input_offset, dtype=np.float64)
        )

    def _lstm(self, x1: np.ndarray, x2: np.ndarray, u: np.ndarray, c_hat: np.ndarray, h_hat: np.ndarray) -> LSTMCache:
        signals = (np.concatenate((x1, x2, u)) - self.input_offset) * self.input_scale
        zeta = np.concatenate((signals, h_hat, (1.0,)))
        return lstm_forward(self.theta, zeta, c_hat, self.layout)

    def _vector_field(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """z = [x1, x2, c_hat, h_hat]."""
        n, L = self.n, self.layout.hidden_dim
        x1, x2 = z[:n], z[n:2 * n]
        c_hat, h_hat = z[2 * n:2 * n + L], z[2 * n + L:]
        cache = self._lstm(x1, x2, u, c_hat, h_hat)
        return np.concatenate((
            x2,
            cache.phi,
            self.config.b_c * (cache.c - c_hat),
            self.config.b_h * (cache.h - h_hat),
        ))

    def settle_memory(
        self, x1: np.ndarray, x2: np.ndarray, u: np.ndarray, t_settle: float = 2.0, dt: float = 0.001,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Relaxes (c_hat, h_hat) to their equilibrium for a held state and input."""
        L = self.layout.hidden_dim
        c_hat = np.zeros(L)
        h_hat = np.zeros(L)
        for _ in range(int(round(t_settle / dt))):
            cache = self._lstm(x1, x2, u, c_hat, h_hat)
            c_hat = c_hat + dt * self.config.b_c * (cache.c - c_hat)
            h_hat = h_hat + dt * self.config.b_h * (cache.h - h_hat)
        return c_hat, h_hat

    def simulate(
        self, u_seq: np.ndarray, initial_state: np.ndarray, dt: float, t_settle: float = 2.0,
    ) -> TwinTrajectory:
        """
        Free-running RK4 rollout under the sampled input u_seq (zero-order hold).
        initial_state is interleaved [x1_0, x2_0, x1_1, x2_1, ...].
        """
        n = self.n
        u_seq = np.asarray(u_seq, dtype=np.float64).reshape(len(u_seq), self.m)
        x0 = np.asarray(initial_state, dtype=np.float64).reshape(n, 2)
        c_hat, h_hat = self.settle_memory(x0[:, 0], x0[:, 1], u_seq[0], t_settle, dt)
        z = np.concatenate((x0[:, 0], x0[:, 1], c_hat, h_hat))

        N = len(u_seq)
        states = np.zeros((N, 2 * n))
        accel = np.zeros((N, n))
        for k in range(N):
            states[k] = np.column_stack((z[:n], z[n:2 * n])).ravel()
            u = u_seq[k]
            k1 = self._vector_field(z, u)
            accel[k] = k1[n:2 * n]
            k2 = self._vector_field(z + 0.5 * dt * k1, u)
            k3 = self._vector_field(z + 0.5 * dt * k2, u)
            k4 = self._vector_field(z + dt * k3, u)
            z = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return TwinTrajectory(t=np.arange(N) * dt, state=states, accel=accel)

    def teacher_forced_acceleration(
        self, x1_seq: np.ndarray, x2_seq: np.ndarray, u_seq: np.ndarray, dt: float, t_settle: float = 2.0,
    ) -> np.ndarray:
        """
        One-step prediction: Phi_hat evaluated on the *true* state sequence, with only the
        LSTM memory (c_hat, h_hat) integrated forward. Isolates model error from drift.
        """
        u_seq = np.asarray(u_seq, dtype=np.float64).reshape(len(u_seq), self.m)
        c_hat, h_hat = self.settle_memory(x1_seq[0], x2_seq[0], u_seq[0], t_settle, dt)
        out = np.zeros((len(u_seq), self.n))
        for k in range(len(u_seq)):
            cache = self._lstm(x1_seq[k], x2_seq[k], u_seq[k], c_hat, h_hat)
            out[k] = cache.phi
            c_hat = c_hat + dt * self.config.b_c * (cache.c - c_hat)
            h_hat = h_hat + dt * self.config.b_h * (cache.h - h_hat)
        return out


# ---------------------------------------------------------------------- unseen test inputs
def step_doublet(amplitude: float = 1.0, t_on: float = 1.0, width: float = 1.0) -> Callable[[float], float]:
    """+A for `width` seconds starting at t_on, then -A for `width` seconds, then zero."""
    def u(t: float) -> float:
        if t_on <= t < t_on + width:
            return amplitude
        if t_on + width <= t < t_on + 2.0 * width:
            return -amplitude
        return 0.0
    return u


def chirp(
    amplitude: float = 1.0, f0: float = 0.3, f1: float = 1.2, t_sweep: float = 15.0, t_ramp: float = 2.0,
) -> Callable[[float], float]:
    """
    Linear chirp f0 -> f1 Hz with a raised-cosine amplitude ramp. The ramp avoids the
    secular cart drift a sine switched on abruptly would inject into the free cart.
    """
    def u(t: float) -> float:
        tc = min(t, t_sweep)
        phase = 2.0 * np.pi * (f0 * tc + 0.5 * (f1 - f0) * tc * tc / t_sweep)
        env = 0.5 * (1.0 - np.cos(np.pi * min(t / t_ramp, 1.0))) if t_ramp > 0 else 1.0
        return float(amplitude * env * np.sin(phase)) if t < t_sweep else 0.0
    return u


# ---------------------------------------------------------------------- evaluation
@dataclass
class TwinTestResult:
    name: str
    truth: OpenLoopTrajectory
    rollout: TwinTrajectory
    accel_teacher_forced: np.ndarray
    state_mse: Dict[str, float]
    state_nmse: Dict[str, float]
    accel_mse: Dict[str, float]
    accel_nmse: Dict[str, float]


def _mse_nmse(truth: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((truth - pred) ** 2))
    var = float(np.var(truth))
    return mse, mse / var if var > 1e-12 else float("nan")


def evaluate_digital_twin(
    twin: LbLSTMDigitalTwin,
    name: str,
    u_fn: Callable[[float], float],
    t_final: float,
    initial_state: np.ndarray,
    dt: float = 0.001,
    plant_params: PendulumParameters | None = None,
    apply_actuator_effects: bool = False,
) -> TwinTestResult:
    """Runs the true plant and the frozen twin under the same unseen input and scores the twin."""
    truth = simulate_open_loop(
        u_fn, t_final, dt, initial_state, plant_params=plant_params,
        apply_actuator_effects=apply_actuator_effects,
    )
    rollout = twin.simulate(truth.u_cmd, truth.state[0], dt)
    accel_tf = twin.teacher_forced_acceleration(truth.positions, truth.velocities, truth.u_cmd, dt)

    state_mse, state_nmse = {}, {}
    for j, s in enumerate(STATE_NAMES):
        state_mse[s], state_nmse[s] = _mse_nmse(truth.state[:, j], rollout.state[:, j])
    accel_mse, accel_nmse = {}, {}
    for j, s in enumerate(("x_ddot", "theta_ddot")):
        accel_mse[s], accel_nmse[s] = _mse_nmse(truth.accel[:, j], accel_tf[:, j])

    return TwinTestResult(name, truth, rollout, accel_tf, state_mse, state_nmse, accel_mse, accel_nmse)


def default_test_suite() -> List[tuple[str, Callable[[float], float], float]]:
    """Unseen inputs: (name, u(t), horizon [s]). None of them appears in the adaptation data."""
    return [
        ("step_doublet", step_doublet(amplitude=1.0, t_on=1.0, width=1.0), 8.0),
        ("chirp", chirp(amplitude=1.0, f0=0.3, f1=1.2, t_sweep=15.0), 15.0),
    ]


def format_results_table(results: List[TwinTestResult]) -> str:
    lines = [
        f"{'Test':<14}{'MSE x':>11}{'MSE x_dot':>11}{'MSE theta':>11}{'MSE th_dot':>11}"
        f"{'NMSE x_ddot':>13}{'NMSE th_ddot':>14}"
    ]
    for r in results:
        lines.append(
            f"{r.name:<14}{r.state_mse['x']:>11.2e}{r.state_mse['x_dot']:>11.2e}"
            f"{r.state_mse['theta']:>11.2e}{r.state_mse['theta_dot']:>11.2e}"
            f"{r.accel_nmse['x_ddot']:>13.3f}{r.accel_nmse['theta_ddot']:>14.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen Lb-LSTM as an open-loop digital twin.")
    parser.add_argument("--weights", type=Path, default=Path("results/approach_a/lblstm_frozen.npz"))
    args = parser.parse_args()

    frozen = FrozenLbLSTM.load(args.weights)
    twin = LbLSTMDigitalTwin(frozen)
    hanging = np.array([0.0, 0.0, np.pi, 0.0])
    results = [
        evaluate_digital_twin(twin, name, u_fn, horizon, hanging)
        for name, u_fn, horizon in default_test_suite()
    ]
    print(f"Frozen Lb-LSTM (theta frozen at t = {frozen.t_freeze:.1f} s, ||theta|| = {np.linalg.norm(frozen.theta):.2f})")
    print(format_results_table(results))


if __name__ == "__main__":
    main()
