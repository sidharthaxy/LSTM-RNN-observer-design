#!/usr/bin/env python3
"""
Approach A validation: black-box Lb-LSTM observer on the Feedback 33-936S rig.

Stage 1 (observer validation, 50 s):
    Open-loop excitation u(t) = 2.0 sin(1.5 t) + 1.2 cos(3.0 t), encoder noise
    U(-0.5 deg, +0.5 deg) on theta (plus the testbed's +/-0.2 mm cart noise and
    4096-count quantization). Velocity estimates [x_dot_hat, theta_dot_hat] of the
    Lb-LSTM are compared with the true velocities, a dirty-derivative filter and the
    shallow continuous RNN observer (Dinh et al., 2014).

Stage 2 (system identification):
    theta_hat is frozen at t_freeze >= 20 s and at the end of the run, the LSTM is
    decoupled from the observer feedback, and the resulting digital twin is driven open
    loop by unseen inputs (step doublet, chirp). Test MSE is reported.

Scenario notes:
    * The pendulum starts 0.4 rad from the hanging (stable) equilibrium: the upright
      equilibrium is open-loop unstable and the specified u(t) is open loop.
    * The cart starts at v0 = -2 / (1.5 (M + m)), the velocity that cancels the secular
      drift the 2 sin(1.5 t) term injects into a free cart starting from rest; without
      it the cart runs into the +/-0.5 m bumpers.
    * Actuator dead-zone / stiction is off by default (--actuator-effects enables it);
      its asymmetric dead-band biases the net force and drives the open-loop cart onto
      the bumper.

Usage:
    python experiments/run_blackbox_validation.py [--seeds 5] [--no-plots]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.baselines.classical_estimators import ContinuousShallowRNNObserver, DirtyDerivativeFilter
from src.identification.extract_model import (
    FrozenLbLSTM,
    LbLSTMDigitalTwin,
    TwinTestResult,
    default_test_suite,
    evaluate_digital_twin,
    format_results_table,
    freeze_observer,
)
from src.observers.blackbox_lstm import LbLSTMObserver, LbLSTMObserverConfig
from src.plant.pendulum_plant import PendulumParameters, PendulumPlant
from src.simulation.open_loop import OpenLoopTrajectory, simulate_open_loop
from src.utils.metrics import compute_chattering_index

DT = 0.001
SETTLE_TIME = 20.0
HANGING_STATE = np.array([0.0, 0.0, np.pi, 0.0])

# Operating-range normalization for zeta: 1 / (expected half-range) of
# [x, theta, x_dot, theta_dot, u] = [0.33 m, 0.33 rad, 0.5 m/s, 1 rad/s, 2 N].
INPUT_SCALE = (3.0, 3.0, 2.0, 1.0, 0.5)


def excitation(t: float) -> float:
    return 2.0 * np.sin(1.5 * t) + 1.2 * np.cos(3.0 * t)


def make_config(offset: tuple[float, ...], seed: int, sign_mode: str = "sgn") -> LbLSTMObserverConfig:
    return LbLSTMObserverConfig(
        input_scale=INPUT_SCALE,
        input_offset=offset,
        sign_mode="tanh" if sign_mode == "tanh" else "sgn",
        tanh_eps=0.005,
        seed=seed,
    )


def simulate_scenario(t_final: float, noise_seed: int, actuator_effects: bool) -> OpenLoopTrajectory:
    p = PendulumParameters()
    v0 = -2.0 / (1.5 * (p.M + p.m))
    x0 = np.array([0.0, v0, np.pi - 0.4, 0.0])
    return simulate_open_loop(
        excitation, t_final, DT, x0, apply_actuator_effects=actuator_effects, seed=noise_seed
    )


def measurement_offset(traj: OpenLoopTrajectory) -> tuple[float, ...]:
    """Data-driven centering of zeta from the first second of encoder data (velocities, u: 0)."""
    n0 = min(len(traj.t), int(round(1.0 / DT)))
    mean_meas = traj.measurements[:n0].mean(axis=0)
    return (float(mean_meas[0]), float(mean_meas[1]), 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------- stage 1
@dataclass
class LbLSTMRun:
    velocity: np.ndarray       # (N, 2)
    theta_norm: np.ndarray     # (N,)
    gate_norm: np.ndarray      # (N,)
    out_norm: np.ndarray       # (N,)
    gate_drift: np.ndarray     # (N,): ||theta_gates(t) - theta_gates(0)||
    out_drift: np.ndarray      # (N,): ||W_h(t) - W_h(0)||
    phi_hat: np.ndarray        # (N, 2)
    frozen: Dict[str, FrozenLbLSTM]


def run_lblstm(traj: OpenLoopTrajectory, config: LbLSTMObserverConfig, t_freeze: float) -> LbLSTMRun:
    obs = LbLSTMObserver(config)
    y0 = traj.measurements[0]
    obs.reset(np.array([y0[0], 0.0, y0[1], 0.0]))
    n_gate = obs.layout.n_gate_params

    N = len(traj.t)
    vel = np.zeros((N, 2))
    th_norm = np.zeros(N)
    gate_norm = np.zeros(N)
    out_norm = np.zeros(N)
    gate_drift = np.zeros(N)
    out_drift = np.zeros(N)
    theta0 = np.copy(obs.theta)
    phi = np.zeros((N, 2))
    frozen: Dict[str, FrozenLbLSTM] = {}
    k_freeze = int(round(t_freeze / DT))

    for k in range(N):
        est = obs.update(traj.measurements[k], traj.u_cmd[k], DT)
        vel[k] = est[[1, 3]]
        phi[k] = obs.phi_hat
        th_norm[k] = obs.theta_norm
        gate_norm[k] = np.linalg.norm(obs.theta[:n_gate])
        out_norm[k] = np.linalg.norm(obs.theta[n_gate:])
        gate_drift[k] = np.linalg.norm(obs.theta[:n_gate] - theta0[:n_gate])
        out_drift[k] = np.linalg.norm(obs.theta[n_gate:] - theta0[n_gate:])
        if k == k_freeze:
            frozen[f"t={t_freeze:g}s"] = freeze_observer(obs, traj.t[k] + DT, SETTLE_TIME)
    frozen[f"t={traj.t[-1] + DT:g}s"] = freeze_observer(obs, traj.t[-1] + DT, SETTLE_TIME)
    return LbLSTMRun(vel, th_norm, gate_norm, out_norm, gate_drift, out_drift, phi, frozen)


def run_dirty_derivative(traj: OpenLoopTrajectory, tau_d: float = 0.02) -> np.ndarray:
    filters = [DirtyDerivativeFilter(tau_d), DirtyDerivativeFilter(tau_d)]
    for j, f in enumerate(filters):
        f.reset(traj.measurements[0, j])
    vel = np.zeros((len(traj.t), 2))
    for k in range(len(traj.t)):
        vel[k] = [f.update(traj.measurements[k, j], DT) for j, f in enumerate(filters)]
    return vel


def run_shallow_rnn(traj: OpenLoopTrajectory) -> np.ndarray:
    """Dinh et al. (2014) observer; grey-box: it carries the nominal plant model."""
    rnn = ContinuousShallowRNNObserver(PendulumPlant())
    y0 = traj.measurements[0]
    rnn.reset(np.array([y0[0], 0.0, y0[1], 0.0]))
    vel = np.zeros((len(traj.t), 2))
    for k in range(len(traj.t)):
        vel[k] = rnn.update(traj.measurements[k], traj.u_cmd[k], DT)[[1, 3]]
    return vel


def velocity_metrics(t: np.ndarray, v_true: np.ndarray, v_hat: np.ndarray) -> Dict[str, float]:
    ss = t >= SETTLE_TIME
    err = v_true - v_hat
    rmse_all = np.sqrt(np.mean(err ** 2, axis=0))
    rmse_ss = np.sqrt(np.mean(err[ss] ** 2, axis=0))
    norm_ss = np.linalg.norm(err[ss], axis=1)
    chat = [compute_chattering_index(v_hat[ss, j], v_true[ss, j], DT)["chattering_ratio"] for j in range(2)]
    return {
        "RMSE x_dot [m/s]": rmse_all[0],
        "RMSE th_dot [rad/s]": rmse_all[1],
        "SS RMSE x_dot [m/s]": rmse_ss[0],
        "SS RMSE th_dot [rad/s]": rmse_ss[1],
        "SS mean ||x2-x2_hat||": float(np.mean(norm_ss)),
        "Chatter x_dot": float(chat[0] or np.nan),
        "Chatter th_dot": float(chat[1] or np.nan),
    }


# ---------------------------------------------------------------------- stage 2
def run_identification(frozen: FrozenLbLSTM) -> List[TwinTestResult]:
    twin = LbLSTMDigitalTwin(frozen)
    return [evaluate_digital_twin(twin, name, u_fn, horizon, HANGING_STATE, DT)
            for name, u_fn, horizon in default_test_suite()]


def _seed_trial(args: tuple[int, float, float, int, bool]) -> Dict[str, float]:
    seed, t_final, t_freeze, noise_seed, actuator = args
    traj = simulate_scenario(t_final, noise_seed, actuator)
    run = run_lblstm(traj, make_config(measurement_offset(traj), seed), t_freeze)
    row: Dict[str, float] = {"seed": seed}
    row.update({k: v for k, v in velocity_metrics(traj.t, traj.velocities, run.velocity).items() if k.startswith("SS RMSE")})
    for label, frozen in run.frozen.items():
        for r in run_identification(frozen):
            row[f"{label} {r.name} NMSE th_ddot"] = r.accel_nmse["theta_ddot"]
            row[f"{label} {r.name} MSE theta"] = r.state_mse["theta"]
    return row


# ---------------------------------------------------------------------- plotting
# Reference categorical palette (validated order): slot 1 blue, 2 orange, 3 aqua.
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
COLORS = {
    "Lb-LSTM": "#2a78d6",
    "Dirty derivative": "#eb6834",
    "Shallow RNN": "#1baf7a",
}


def _style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 1.1, "font.family": "DejaVu Sans",
    })


def _moving_rms(x: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.sqrt(np.convolve(x ** 2, kernel, mode="same"))


def make_figures(
    out_dir: Path, traj: OpenLoopTrajectory, estimates: Dict[str, np.ndarray],
    run: LbLSTMRun, config: LbLSTMObserverConfig, twin_results: Dict[str, List[TwinTestResult]],
) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    t = traj.t
    v_true = traj.velocities
    saved: List[Path] = []
    labels = [r"$\dot{x}$ [m/s]", r"$\dot{\theta}$ [rad/s]"]

    # 1. Tracking trajectories: transient window and steady-state window.
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), sharex="col")
    windows = [(0.0, 5.0, "Transient, 0-5 s"), (45.0, 50.0, "Steady state, 45-50 s")]
    for col, (t0, t1, title) in enumerate(windows):
        m = (t >= t0) & (t <= t1)
        for row in range(2):
            ax = axes[row, col]
            for name in ("Dirty derivative", "Shallow RNN", "Lb-LSTM"):
                ax.plot(t[m], estimates[name][m, row], color=COLORS[name], label=name,
                        lw=0.8 if name == "Dirty derivative" else 1.2,
                        alpha=0.7 if name == "Dirty derivative" else 1.0)
            ax.plot(t[m], v_true[m, row], color=INK, lw=1.6, ls="--", label="True")
            if col == 0:
                ax.set_ylabel(labels[row])
            if row == 0:
                ax.set_title(title, loc="left")
            if row == 1:
                ax.set_xlabel("Time [s]")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper right", ncol=4, bbox_to_anchor=(0.99, 0.995))
    fig.suptitle("Velocity reconstruction under U(±0.5°) encoder noise", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    saved.append(out_dir / "tracking_trajectories.png")
    fig.savefig(saved[-1])
    plt.close(fig)

    # 2. Estimation error norm ||x2 - x2_hat|| (100 ms moving RMS, log scale).
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for name in ("Dirty derivative", "Shallow RNN", "Lb-LSTM"):
        err = np.linalg.norm(v_true - estimates[name], axis=1)
        ax.semilogy(t, _moving_rms(err, 100), color=COLORS[name], label=name, lw=1.2)
    ax.axvline(SETTLE_TIME, color=INK_2, lw=0.8, ls=":")
    ax.text(SETTLE_TIME + 0.3, ax.get_ylim()[0] * 1.15, "settling time 20 s", color=INK_2, fontsize=8, va="bottom")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\|x_2 - \hat{x}_2\|$ (100 ms RMS)")
    ax.set_title("Velocity estimation error norm", loc="left")
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    saved.append(out_dir / "estimation_error_norm.png")
    fig.savefig(saved[-1])
    plt.close(fig)

    # 3. Phase planes after settling.
    ss = t >= SETTLE_TIME
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    pos = traj.positions
    for j, (ax, xl, yl) in enumerate(zip(
        axes, (r"$x$ [m]", r"$\theta$ [rad]"), (r"$\dot{x}$ [m/s]", r"$\dot{\theta}$ [rad/s]"),
    )):
        ax.plot(pos[ss, j], estimates["Lb-LSTM"][ss, j], color=COLORS["Lb-LSTM"], lw=0.7, label="Lb-LSTM estimate")
        ax.plot(pos[ss, j], v_true[ss, j], color=INK, lw=0.9, ls="--", label="True")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(("Cart", "Pendulum")[j] + " phase plane, t ≥ 20 s", loc="left")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    saved.append(out_dir / "phase_plane.png")
    fig.savefig(saved[-1])
    plt.close(fig)

    # 4. Weight norm evolution with the projection ball.
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
    ax.plot(t, run.theta_norm, color=COLORS["Lb-LSTM"], lw=1.4, label=r"$\|\hat{\theta}\|$ (all weights)")
    ax.plot(t, run.gate_norm, color=COLORS["Dirty derivative"], lw=1.0, ls="--",
            label=r"$\|[W_c, W_i, W_f, W_o]\|$")
    ax.plot(t, run.out_norm, color=COLORS["Shallow RNN"], lw=1.0, label=r"$\|W_h\|$")
    w_bar = config.w_bar
    ax.axhline(w_bar, color=INK_2, lw=0.9, ls="--")
    ax.text(t[-1], w_bar, r"$\bar{W}$ = " + f"{w_bar:g}", color=INK_2, fontsize=8, ha="right", va="bottom")
    ax.axhline(w_bar / np.sqrt(1.0 + config.proj_eps), color=INK_2, lw=0.6, ls=":")
    ax.set_ylim(0.0, w_bar * 1.08)
    ax.set_ylabel("Weight norm")
    ax.set_title("Weight norm evolution under smooth projection", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.72), ncol=3)
    ax2.plot(t, run.gate_drift, color=COLORS["Dirty derivative"], lw=1.2, label="gate weights")
    ax2.plot(t, run.out_drift, color=COLORS["Shallow RNN"], lw=1.2, label=r"readout $W_h$")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel(r"$\|\hat{\theta}_{blk}(t) - \hat{\theta}_{blk}(0)\|$")
    ax2.set_title("Drift from initialization, per block", loc="left")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    saved.append(out_dir / "weight_norm.png")
    fig.savefig(saved[-1])
    plt.close(fig)

    # 5. Digital-twin rollouts on unseen inputs.
    freeze_labels = list(twin_results)
    first = twin_results[freeze_labels[0]]
    fig, axes = plt.subplots(len(first), 2, figsize=(10, 2.8 * len(first)), squeeze=False)
    twin_colors = [COLORS["Lb-LSTM"], COLORS["Dirty derivative"]]
    for i, ref in enumerate(first):
        for j, (idx, yl) in enumerate(((0, r"$x$ [m]"), (2, r"$\theta - \pi$ [rad]"))):
            ax = axes[i, j]
            off = np.pi if idx == 2 else 0.0
            for c, label in zip(twin_colors, freeze_labels):
                r = twin_results[label][i]
                ax.plot(r.rollout.t, r.rollout.state[:, idx] - off, color=c, lw=1.1,
                        label=f"Twin, θ̂ frozen at {label[2:]}")
            ax.plot(ref.truth.t, ref.truth.state[:, idx] - off, color=INK, lw=1.4, ls="--", label="True plant")
            ax.set_ylabel(yl)
            ax.set_title(f"{ref.name.replace('_', ' ')}: {('cart', 'pendulum')[j]}", loc="left")
            if i == len(first) - 1:
                ax.set_xlabel("Time [s]")
    axes[0, 0].legend(loc="best")
    fig.suptitle("Frozen Lb-LSTM as an open-loop digital twin (unseen inputs)", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    saved.append(out_dir / "digital_twin.png")
    fig.savefig(saved[-1])
    plt.close(fig)
    return saved


# ---------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--t-final", type=float, default=50.0)
    parser.add_argument("--t-freeze", type=float, default=20.0, help="freeze time (>= 20 s)")
    parser.add_argument("--seed", type=int, default=0, help="Lb-LSTM weight-initialization seed")
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=0,
                        help="extra initialization seeds for the robustness table (0 = skip)")
    parser.add_argument("--actuator-effects", action="store_true",
                        help="enable the PCI-1711 dead-zone / stiction model")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--fig-dir", type=Path, default=ROOT_DIR / "figures" / "approach_a")
    parser.add_argument("--results-dir", type=Path, default=ROOT_DIR / "results" / "approach_a")
    args = parser.parse_args()
    if args.t_freeze < SETTLE_TIME or args.t_freeze >= args.t_final:
        parser.error(f"--t-freeze must lie in [{SETTLE_TIME}, t_final).")

    pd.set_option("display.width", 200)
    print("=" * 96)
    print("APPROACH A: BLACK-BOX Lb-LSTM OBSERVER, FEEDBACK 33-936S (open loop, hanging start)")
    print("u(t) = 2.0 sin(1.5 t) + 1.2 cos(3.0 t);  encoder noise U(-0.5 deg, +0.5 deg)")
    print("=" * 96)

    t0 = time.perf_counter()
    traj = simulate_scenario(args.t_final, args.noise_seed, args.actuator_effects)
    config = make_config(measurement_offset(traj), args.seed)
    config_tanh = replace(config, sign_mode="tanh")

    with ProcessPoolExecutor(max_workers=4) as pool:
        fut_sgn = pool.submit(run_lblstm, traj, config, args.t_freeze)
        fut_tanh = pool.submit(run_lblstm, traj, config_tanh, args.t_freeze)
        fut_rnn = pool.submit(run_shallow_rnn, traj)
        fut_dd = pool.submit(run_dirty_derivative, traj)
        run, run_tanh = fut_sgn.result(), fut_tanh.result()
        estimates = {
            "Lb-LSTM": run.velocity,
            "Lb-LSTM (tanh)": run_tanh.velocity,
            "Shallow RNN": fut_rnn.result(),
            "Dirty derivative": fut_dd.result(),
        }
    print(f"[+] Stage 1 simulated in {time.perf_counter() - t0:.1f} s "
          f"({len(traj.t)} samples @ {1 / DT:.0f} Hz, {run.frozen[next(iter(run.frozen))].theta.size} weights)")

    table = pd.DataFrame({name: velocity_metrics(traj.t, traj.velocities, v) for name, v in estimates.items()}).T
    print("\nStage 1 - velocity estimation (SS = t >= 20 s; chatter = TV(estimate) / TV(truth)):")
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))
    ss = traj.t >= SETTLE_TIME
    nmse_online = np.mean((run.phi_hat[ss] - traj.accel[ss]) ** 2, axis=0) / np.var(traj.accel[ss], axis=0)
    print(f"\nOnline model fit (t >= 20 s): NMSE(Phi_hat, true accel) = "
          f"[x_ddot {nmse_online[0]:.3f}, theta_ddot {nmse_online[1]:.3f}], "
          f"final ||theta_hat|| = {run.theta_norm[-1]:.2f} (W_bar = {config.w_bar:g})")

    print("\nStage 2 - frozen digital twin on unseen inputs (from rest at the hanging equilibrium):")
    twin_results: Dict[str, List[TwinTestResult]] = {}
    for label, frozen in run.frozen.items():
        twin_results[label] = run_identification(frozen)
        print(f"\n  theta_hat frozen at {label[2:]}:")
        print("  " + format_results_table(twin_results[label]).replace("\n", "\n  "))
    print("  (state MSE: free-running rollout; NMSE accel: one-step, fed the true states)")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.results_dir / "lblstm_frozen.npz"
    run.frozen[f"t={args.t_freeze:g}s"].save(weights_path)
    print(f"\n[+] Frozen weights saved to {weights_path.relative_to(ROOT_DIR)}")

    if args.seeds > 0:
        seeds = list(range(1, args.seeds + 1))
        print(f"\nRobustness over weight-initialization seeds {seeds}:")
        trials = [(s, args.t_final, args.t_freeze, args.noise_seed, args.actuator_effects) for s in seeds]
        with ProcessPoolExecutor() as pool:
            rows = list(pool.map(_seed_trial, trials))
        rob = pd.DataFrame(rows).set_index("seed")
        summary = pd.concat([rob.T, rob.agg(["median", "min", "max"]).T], axis=1)
        summary.columns = [f"seed {c}" for c in rob.index] + ["median", "min", "max"]
        print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    if not args.no_plots:
        paths = make_figures(args.fig_dir, traj, estimates, run, config, twin_results)
        print("\n[+] Figures:")
        for p in paths:
            print(f"    {p.relative_to(ROOT_DIR)}")
    print("=" * 96)


if __name__ == "__main__":
    main()
