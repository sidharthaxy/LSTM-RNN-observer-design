#!/usr/bin/env python3
"""
Generate a publication-quality 3-panel side-by-side comparison of:
  - Dirty derivative
  - Shallow RNN (Dinh et al., 2014)
  - Lb-LSTM (Griffis et al., 2024, Approach A)

Across all three benchmark scenarios:
  1. S1: Hanging pendulum, two-sine excitation (Approach A validation)
  2. S2: Swing-through with +50% mass/inertia step at 25 s (Approach B validation)
  3. S3: Weak decaying excitation, non-PE (Approach C validation)

Saves the result to figures/approach_a/estimation_error_norm.png.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.baselines.classical_estimators import ContinuousShallowRNNObserver, DirtyDerivativeFilter
from src.observers.blackbox_lstm import LbLSTMObserver, LbLSTMObserverConfig
from src.plant.pendulum_plant import PendulumParameters, PendulumPlant
from src.plant.sensor_noise import SensorNoiseConfig, SensorNoiseModel

DT = 0.001
SETTLE_TIME = 20.0

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
COLORS = {
    "Lb-LSTM": "#2a78d6",
    "Dirty derivative": "#eb6834",
    "Shallow RNN": "#1baf7a",
}


@dataclass
class ScenarioDataset:
    name: str
    label: str
    subtitle: str
    t: np.ndarray
    state: np.ndarray
    meas: np.ndarray
    u: np.ndarray
    shift_time: float | None = None

    @property
    def velocities(self) -> np.ndarray:
        return self.state[:, [1, 3]]


def _simulate_scenario(
    name: str,
    label: str,
    subtitle: str,
    u_fn: Callable[[float], float],
    t_final: float,
    x0: np.ndarray,
    noise_deg: float,
    noise_seed: int,
    params: PendulumParameters | None = None,
    shift: Tuple[float, PendulumParameters] | None = None,
) -> ScenarioDataset:
    p = params or PendulumParameters()
    plant = PendulumPlant(p)
    plant.reset(np.asarray(x0, dtype=np.float64))
    cfg = SensorNoiseConfig(theta_noise_deg=noise_deg)
    sensor = SensorNoiseModel(cfg)
    sensor.seed(noise_seed)
    t = np.arange(0.0, t_final, DT)
    N = t.size
    state = np.zeros((N, 4), dtype=np.float64)
    meas = np.zeros((N, 2), dtype=np.float64)
    u = np.zeros(N, dtype=np.float64)
    shifted = False

    for k, tk in enumerate(t):
        if shift is not None and not shifted and tk >= shift[0] - 1e-9:
            plant.params = shift[1]
            shifted = True
        s = plant.state.to_array()
        state[k] = s
        meas[k] = sensor.measure(s[0], s[2])
        u[k] = u_fn(float(tk))
        plant.step(u[k], DT)

    return ScenarioDataset(
        name=name,
        label=label,
        subtitle=subtitle,
        t=t,
        state=state,
        meas=meas,
        u=u,
        shift_time=None if shift is None else shift[0],
    )


def _drift_cancelling_state(u_fn: Callable[[float], float], t_final: float, p: PendulumParameters | None = None) -> Tuple[float, float]:
    p = p or PendulumParameters()
    t = np.arange(0.0, t_final, DT)
    vel = np.cumsum([u_fn(float(tk)) for tk in t]) * DT / p.total_cart_mass
    v0 = -float(np.mean(vel))
    pos = np.cumsum(vel + v0) * DT
    return -0.5 * float(pos.max() + pos.min()), v0


def build_scenarios(seed: int = 0) -> list[ScenarioDataset]:
    noise_seed = 42 + seed
    p = PendulumParameters()

    # Scenario 1: Hanging, two-sine (Approach A home validation)
    u1 = lambda t: 2.0 * np.sin(1.5 * t) + 1.2 * np.cos(3.0 * t)
    x0_1 = np.array([0.0, -2.0 / (1.5 * p.total_cart_mass), np.pi - 0.4, 0.0])
    s1 = _simulate_scenario(
        name="S1",
        label="Scenario 1: Hanging, two-sine",
        subtitle=r"$u = 2\sin 1.5t + 1.2\cos 3t$, $\theta_0 = \pi - 0.4$, $\pm 0.5^\circ$ noise",
        u_fn=u1,
        t_final=50.0,
        x0=x0_1,
        noise_deg=0.5,
        noise_seed=noise_seed,
    )

    # Scenario 2: Swing-through + mass step (Approach B home validation)
    u2 = lambda t: 2.0 * np.sin(3.0 * t) + 1.2 * np.cos(6.0 * t)
    x0_2 = np.array([0.0, -2.0 / (3.0 * p.total_cart_mass), 0.5, 0.0])
    p_after = replace(p, m=1.5 * p.m, I=1.5 * p.I)
    s2 = _simulate_scenario(
        name="S2",
        label="Scenario 2: Swing-through + mass step",
        subtitle=r"$u = 2\sin 3t + 1.2\cos 6t$, $\theta_0 = 0.5$, mass $+50\%$ at 25 s, $\pm 1^\circ$ noise",
        u_fn=u2,
        t_final=50.0,
        x0=x0_2,
        noise_deg=1.0,
        noise_seed=noise_seed,
        shift=(25.0, p_after),
    )

    # Scenario 3: Weak decaying excitation (Approach C home validation)
    u3 = lambda t: 3.0 * np.exp(-t / 8.0) * np.sin(1.6 * t)
    x_c3, v0_3 = _drift_cancelling_state(u3, 60.0)
    x0_3 = np.array([x_c3, v0_3, np.pi - 0.3, 0.0])
    s3 = _simulate_scenario(
        name="S3",
        label="Scenario 3: Weak decaying excitation",
        subtitle=r"$u = 3e^{-t/8}\sin 1.6t$ (non-PE), $\theta_0 = \pi - 0.3$, $\pm 0.5^\circ$ noise",
        u_fn=u3,
        t_final=60.0,
        x0=x0_3,
        noise_deg=0.5,
        noise_seed=noise_seed,
    )

    return [s1, s2, s3]


def run_dirty_derivative(data: ScenarioDataset, tau_d: float = 0.02) -> np.ndarray:
    filters = [DirtyDerivativeFilter(tau_d), DirtyDerivativeFilter(tau_d)]
    for j in range(2):
        filters[j].reset(data.meas[0, j])
    vel = np.zeros((len(data.t), 2), dtype=np.float64)
    for k in range(len(data.t)):
        vel[k] = [filters[0].update(data.meas[k, 0], DT), filters[1].update(data.meas[k, 1], DT)]
    return vel


def run_shallow_rnn(data: ScenarioDataset) -> np.ndarray:
    rnn = ContinuousShallowRNNObserver(PendulumPlant())
    rnn.reset(np.array([data.meas[0, 0], 0.0, data.meas[0, 1], 0.0]))
    vel = np.zeros((len(data.t), 2), dtype=np.float64)
    for k in range(len(data.t)):
        vel[k] = rnn.update(data.meas[k], data.u[k], DT)[[1, 3]]
    return vel


def _compute_normalization(data: ScenarioDataset, window_s: float = 5.0) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    n0 = min(len(data.t), int(round(window_s / DT)))
    y = data.meas[:n0]
    mu = y.mean(axis=0)
    half = np.maximum(np.abs(y - mu).max(axis=0), 1e-3)
    vel = np.zeros((n0, 2), dtype=np.float64)
    for j in range(2):
        f = DirtyDerivativeFilter(0.05)
        f.reset(y[0, j])
        vel[:, j] = [f.update(v, DT) for v in y[:, j]]
    vmax = np.maximum(np.abs(vel).max(axis=0), 1e-3)
    u_max = max(float(np.abs(data.u[:n0]).max()), 1e-3)

    input_scale = (float(1.0 / half[0]), float(1.0 / half[1]), float(1.0 / vmax[0]), float(1.0 / vmax[1]), float(1.0 / u_max))
    input_offset = (float(mu[0]), float(mu[1]), 0.0, 0.0, 0.0)
    return input_scale, input_offset


def run_lblstm(data: ScenarioDataset, seed: int = 0) -> np.ndarray:
    input_scale, input_offset = _compute_normalization(data)
    cfg = LbLSTMObserverConfig(
        input_scale=input_scale,
        input_offset=input_offset,
        sign_mode="sgn",
        tanh_eps=0.005,
        seed=seed,
    )
    obs = LbLSTMObserver(cfg)
    obs.reset(np.array([data.meas[0, 0], 0.0, data.meas[0, 1], 0.0]))
    vel = np.zeros((len(data.t), 2), dtype=np.float64)
    for k in range(len(data.t)):
        est = obs.update(data.meas[k], data.u[k], DT)
        vel[k] = est[[1, 3]]
    return vel


def _moving_rms(x: np.ndarray, window: int = 100) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.sqrt(np.convolve(x ** 2, kernel, mode="same"))


def _evaluate_scenario_job(data: ScenarioDataset) -> Tuple[str, Dict[str, np.ndarray]]:
    v_dd = run_dirty_derivative(data)
    v_rnn = run_shallow_rnn(data)
    v_lstm = run_lblstm(data)
    return data.name, {
        "Dirty derivative": v_dd,
        "Shallow RNN": v_rnn,
        "Lb-LSTM": v_lstm,
    }


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.15,
        "font.family": "DejaVu Sans",
    })


def generate_comparison_plot(out_path: Path, seed: int = 0) -> Path:
    _style()
    print("[*] Simulating benchmark scenarios (S1, S2, S3)...")
    scenarios = build_scenarios(seed=seed)

    print("[*] Running estimators across scenarios (Dirty derivative, Shallow RNN, Lb-LSTM)...")
    with ProcessPoolExecutor(max_workers=3) as pool:
        results = dict(pool.map(_evaluate_scenario_job, scenarios))

    print("[*] Rendering consolidated 3-scenario side-by-side figure...")
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.0), sharey=True)

    for idx, (ax, data) in enumerate(zip(axes, scenarios)):
        est = results[data.name]
        v_true = data.velocities

        for name in ("Dirty derivative", "Shallow RNN", "Lb-LSTM"):
            err = np.linalg.norm(v_true - est[name], axis=1)
            rms_err = _moving_rms(err, 100)
            ax.semilogy(data.t, rms_err, color=COLORS[name], label=name, lw=1.2)

        # Settling time marker (20s)
        ax.axvline(SETTLE_TIME, color=INK_2, lw=0.8, ls=":")
        ax.text(
            SETTLE_TIME + 0.5,
            0.0007,
            "settling time 20 s",
            color=INK_2,
            fontsize=7.5,
            va="bottom",
        )

        # Scenario 2 mass-step marker (25s)
        if data.shift_time is not None:
            ax.axvline(data.shift_time, color="#8b0000", lw=0.8, ls="--", alpha=0.7)
            ax.text(
                data.shift_time + 0.5,
                0.22,
                "mass step (+50%)",
                color="#8b0000",
                fontsize=7.5,
                va="bottom",
            )

        ax.set_xlabel("Time [s]", fontweight="bold", labelpad=4)
        if idx == 0:
            ax.set_ylabel(r"$\|x_2 - \hat{x}_2\|$ (100 ms RMS)", fontweight="bold", labelpad=6)

        # Panel titles with clear scenario descriptor
        ax.set_title(f"{data.label}\n", loc="left", fontsize=9.5, fontweight="bold", pad=2)
        ax.text(
            0.0,
            1.01,
            data.subtitle,
            transform=ax.transAxes,
            fontsize=7.2,
            color=INK_2,
            va="bottom",
        )

        ax.set_ylim(4e-4, 2.0)
        ax.set_xlim(-0.5, data.t[-1] + 0.5)

    # Unified clean legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        ncol=3,
        bbox_to_anchor=(0.99, 1.03),
        frameon=True,
        facecolor="white",
        edgecolor=GRID,
    )

    fig.suptitle(
        "Velocity estimation error norm across all three benchmark scenarios",
        x=0.015,
        y=1.02,
        ha="left",
        fontweight="bold",
        fontsize=11.5,
        color=INK,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved comparison figure to: {out_path}")
    return out_path


def main() -> None:
    target_path = ROOT_DIR / "figures" / "approach_a" / "estimation_error_norm.png"
    generate_comparison_plot(target_path)


if __name__ == "__main__":
    main()
