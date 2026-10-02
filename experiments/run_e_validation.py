#!/usr/bin/env python3
"""
Approach E validation: the B + D hybrid against Approach B (PI-LSTM) and Approach D (PI-ICL).

Scenarios (the home scenarios of Approaches A, B and C, so that the hybrid is tested where each
parent is strong and where it is weak):
    hanging   pendulum hanging, u = 2 sin 1.5t + 1.2 cos 3t, theta(0) = pi - 0.4, U(+-0.5 deg), 50 s
    swing     swing-through from 0.5 rad off upright, u = 2 sin 3t + 1.2 cos 6t, U(+-1 deg),
              pendulum mass and inertia +50 % at 25 s, 50 s
    weak      non-PE input u = 3 exp(-t/8) sin 1.6t, hanging, U(+-0.5 deg), 60 s

Models:
    B                  PI-LSTM (Approach B defaults)
    D                  PI-ICL (Approach D defaults)
    E                  the hybrid (this branch's defaults)
    E, no freeze       own-coordinate inertia freeze off
    E, no blend        Approach D's Newton CL and rank gate (gamma_inst = 2), freeze kept
    E + residual       Approach B's friction LSTM added as a dissipative residual

Outputs: results/approach_e/*.csv, figures/approach_e/*.png
Usage:   python experiments/run_e_validation.py [--seeds 5] [--workers 8] [--no-plots]
         RUN_E_QUICK=1 shortens every scenario to 10 s (smoke test of the pipeline; not for results).
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.baselines.classical_estimators import DirtyDerivativeFilter  # noqa: E402
from src.observers.el_linear_model import LinearELModel  # noqa: E402
from src.observers.hybrid_observer import HybridObserver, HybridObserverConfig  # noqa: E402
from src.observers.pi_icl_observer import PIICLObserver, PIICLObserverConfig  # noqa: E402
from src.observers.pilstm_observer import PILSTMObserver, PILSTMObserverConfig  # noqa: E402
from src.plant.pendulum_plant import PendulumParameters, PendulumPlant  # noqa: E402
from src.plant.sensor_noise import SensorNoiseConfig, SensorNoiseModel  # noqa: E402

DT = 0.001
QUICK = os.environ.get("RUN_E_QUICK") == "1"
PHYS = ("M+m", "ml", "I+ml^2", "mgl")
SCENARIOS = ("hanging", "swing", "weak")
STEADY = {"hanging": (35.0, 50.0), "swing": (35.0, 50.0), "weak": (45.0, 60.0)}
if QUICK:
    STEADY = {k: (7.0, 10.0) for k in STEADY}
MODELS = ("B", "D", "E", "E, no freeze", "E, no blend", "E + residual")
E_VARIANTS: Dict[str, Dict[str, Any]] = {
    "E": {},
    "E, no freeze": {"freeze_own_inertia": False},
    "E, no blend": {"blend": False, "gamma_inst": 2.0},
    "E + residual": {"residual": True},
}


@dataclass
class Data:
    t: np.ndarray
    state: np.ndarray          # (N, 4) true [x, x_dot, theta, theta_dot]
    accel: np.ndarray          # (N, 2)
    meas: np.ndarray           # (N, 2) encoder [x, theta]
    u: np.ndarray              # (N,) commanded force
    params_pre: PendulumParameters
    params_final: PendulumParameters
    shift_time: float | None
    bumper: bool

    @property
    def velocities(self) -> np.ndarray:
        return self.state[:, [1, 3]]


def simulate(u_fn: Callable[[float], float], t_final: float, x0: np.ndarray, noise_deg: float, seed: int,
             shift: Tuple[float, PendulumParameters] | None = None) -> Data:
    p = PendulumParameters()
    plant = PendulumPlant(p)
    plant.reset(np.asarray(x0, dtype=np.float64))
    sensor = SensorNoiseModel(SensorNoiseConfig(theta_noise_deg=noise_deg))
    sensor.seed(42 + seed)
    if QUICK:
        t_final, shift = 10.0, None if shift is None else (5.0, shift[1])
    t = np.arange(0.0, t_final, DT)
    N = t.size
    state, accel, meas, u = np.zeros((N, 4)), np.zeros((N, 2)), np.zeros((N, 2)), np.zeros(N)
    shifted = False
    for k, tk in enumerate(t):
        if shift is not None and not shifted and tk >= shift[0] - 1e-9:
            plant.params = shift[1]
            shifted = True
        s = plant.state.to_array()
        state[k] = s
        meas[k] = sensor.measure(s[0], s[2])
        u[k] = u_fn(float(tk))
        accel[k] = plant.forward_dynamics(s, u[k])
        plant.step(u[k], DT)
    bumper = bool(np.any(np.abs(state[:, 0]) >= p.x_limit - 1e-9))
    return Data(t, state, accel, meas, u, p, plant.params, None if shift is None else shift[0], bumper)


def scenario(name: str, seed: int) -> Data:
    p = PendulumParameters()
    mt = p.M + p.m
    if name == "hanging":
        u = lambda t: 2.0 * np.sin(1.5 * t) + 1.2 * np.cos(3.0 * t)
        return simulate(u, 50.0, np.array([0.0, -2.0 / (1.5 * mt), np.pi - 0.4, 0.0]), 0.5, seed)
    if name == "swing":
        u = lambda t: 2.0 * np.sin(3.0 * t) + 1.2 * np.cos(6.0 * t)
        return simulate(u, 50.0, np.array([0.0, -2.0 / (3.0 * mt), 0.5, 0.0]), 1.0, seed,
                        shift=(25.0, replace(p, m=1.5 * p.m, I=1.5 * p.I)))
    if name == "weak":
        u = lambda t: 3.0 * np.exp(-t / 8.0) * np.sin(1.6 * t)
        tt = np.arange(0.0, 60.0, DT)
        vel = np.cumsum([u(float(x)) for x in tt]) * DT / mt        # drift-cancelling cart state
        v0 = -float(np.mean(vel))
        pos = np.cumsum(vel + v0) * DT
        return simulate(u, 60.0, np.array([-0.5 * float(pos.max() + pos.min()), v0, np.pi - 0.3, 0.0]), 0.5, seed)
    raise KeyError(name)


def velocity_scale(data: Data) -> Tuple[float, float]:
    """Approach B's data-driven velocity normalization (dirty derivative over the first 5 s)."""
    n0 = round(5.0 / DT)
    out = []
    for j in range(2):
        f = DirtyDerivativeFilter(0.05)
        f.reset(data.meas[0, j])
        v = np.array([f.update(y, DT) for y in data.meas[:n0, j]])
        out.append(float(1.0 / max(np.abs(v).max(), 1e-3)))
    return out[0], out[1]


def make_observer(model: str, data: Data, seed: int) -> Any:
    vs = velocity_scale(data)
    if model == "B":
        return PILSTMObserver(PILSTMObserverConfig(velocity_scale=vs, seed=seed))
    if model == "D":
        return PIICLObserver(PIICLObserverConfig(seed=seed))
    return HybridObserver(HybridObserverConfig(seed=seed, velocity_scale=vs, **E_VARIANTS[model]))


def run(model: str, data: Data, seed: int, name: str, log: bool = False) -> Dict[str, Any]:
    obs = make_observer(model, data, seed)
    obs.reset(np.array([data.meas[0, 0], 0.0, data.meas[0, 1], 0.0]))
    N = len(data.t)
    vel, acc = np.full((N, 2), np.nan), np.full((N, 2), np.nan)
    is_el = isinstance(obs, PIICLObserver)
    phys_log, w_log, t_log, m_err = [], [], [], []
    plant = PendulumPlant(data.params_pre)
    a_ss = STEADY[name][0]
    diverged = None
    for k in range(N):
        est = obs.update(data.meas[k], data.u[k], DT)
        if not np.all(np.isfinite(est)) or np.abs(est).max() > 1e3:
            diverged = float(data.t[k])
            break
        vel[k] = est[[1, 3]]
        acc[k] = obs.phi_hat if is_el else obs.last_cache.phi
        if is_el and k % 100 == 0 and data.t[k] >= a_ss:
            plant.params = data.params_final
            M_true = plant.mass_matrix(data.state[k, 2])
            m_err.append(np.linalg.norm(obs.inertia_estimate(data.state[k, [0, 2]]) - M_true) / np.linalg.norm(M_true))
        if log and is_el and k % 100 == 0:
            t_log.append(data.t[k])
            pp = obs.physical_parameters()
            phys_log.append([pp[n] for n in PHYS])
            if isinstance(obs, HybridObserver):
                w = obs.information_weights()
                w_log.append(np.concatenate([w.get("actuated", np.full(obs.cols_A.size, np.nan)),
                                             w.get("row 1 (normalized)", np.full(1 + obs.unact[0].own.size, np.nan))]))
    out: Dict[str, Any] = {"model": model, "seed": seed, "diverged": float(diverged is not None), "vel": vel}
    err = vel - data.velocities
    t = data.t
    a, b = STEADY[name]
    m = (t >= a) & (t < b)
    rmse = np.sqrt(np.nanmean(err[m] ** 2, axis=0))
    out["RMSE x_dot steady"], out["RMSE th_dot steady"] = float(rmse[0]), float(rmse[1])
    mt = (t >= 0.1) & (t < 5.0)
    out["RMSE th_dot transient"] = float(np.sqrt(np.nanmean(err[mt, 1] ** 2)))
    out["peak th_dot err 0.1-2 s"] = float(np.nanmax(np.abs(err[(t >= 0.1) & (t < 2.0), 1])))
    if data.shift_time is not None:
        ms = (t >= data.shift_time) & (t < data.shift_time + 5.0)
        out["RMSE th_dot post-shift"] = float(np.sqrt(np.nanmean(err[ms, 1] ** 2)))
    out["online NMSE th_ddot steady"] = float(np.nanmean((acc[m, 1] - data.accel[m, 1]) ** 2) / np.var(data.accel[m, 1]))
    if is_el and diverged is None:
        truth = LinearELModel.physical_truth(data.params_final)
        pp = obs.physical_parameters()
        for n in PHYS:
            out[f"err {n}"] = abs(pp[n] - truth[n]) / abs(truth[n])
        out["err M(q) steady"] = float(np.median(m_err))
        out["purges"] = len(obs.purge_times)
    if log:
        out.update(t_log=np.array(t_log), phys_log=np.array(phys_log), w_log=np.array(w_log))
    return out


def job(args: Tuple[str, str, int, bool]) -> Dict[str, Any]:
    name, model, seed, log = args
    data = scenario(name, seed)
    r = run(model, data, seed, name, log=log)
    if log:
        return r
    r.pop("vel")
    r["scenario"] = name
    r["bumper"] = data.bumper
    return r


# ---------------------------------------------------------------------- plotting
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Reference categorical palette, fixed slots: B = 2 (orange), D = 4 (yellow), E = 7 (violet), as in the shared benchmark.
COLORS = {"B": "#eb6834", "D": "#eda100", "E": "#4a3aa7"}
LABELS = {"B": "B: PI-LSTM", "D": "D: PI-ICL", "E": "E: hybrid"}
TITLES = {"hanging": "hanging, small swings", "swing": "swing-through, +50 % mass at 25 s", "weak": "weak, fading input"}


def _style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": INK_2, "ytick.color": INK_2,
        "legend.frameon": False, "legend.fontsize": 8, "font.family": "DejaVu Sans", "lines.linewidth": 2.0,
    })


def _moving_rms(x: np.ndarray, window: int) -> np.ndarray:
    return np.sqrt(np.convolve(x ** 2, np.ones(window) / window, mode="same"))


def plot_velocity(ex: Dict[str, Dict[str, Any]], out: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for ax, name in zip(axes, SCENARIOS):
        data = ex[name]["data"]
        for m in ("B", "D", "E"):
            e = ex[name][m]["vel"][:, 1] - data.velocities[:, 1]
            ax.plot(data.t, _moving_rms(np.nan_to_num(e), 500), color=COLORS[m], label=LABELS[m])
        if data.shift_time is not None:
            ax.axvline(data.shift_time, color=INK_2, linewidth=0.8, linestyle=":")
        ax.set_yscale("log")
        ax.set_title(TITLES[name])
        ax.set_xlabel("time [s]")
    axes[0].set_ylabel(r"$\dot\theta$ error, 0.5 s RMS [rad/s]")
    axes[0].legend(loc="upper right")
    fig.suptitle("Angular-velocity estimation error over time (seed 0)", fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out)
    plt.close(fig)


def plot_parameters(ex: Dict[str, Dict[str, Any]], out: Path) -> None:
    import matplotlib.pyplot as plt
    names = ("hanging", "swing")
    fig, axes = plt.subplots(len(names), len(PHYS), figsize=(13.5, 5.6), squeeze=False)
    for r, name in enumerate(names):
        data = ex[name]["data"]
        for c, pname in enumerate(PHYS):
            ax = axes[r][c]
            for m in ("D", "E"):
                run_ = ex[name][m]
                ax.plot(run_["t_log"], run_["phys_log"][:, c], color=COLORS[m], label=LABELS[m])
            tr = [LinearELModel.physical_truth(p)[pname] for p in (data.params_pre, data.params_final)]
            ts = data.shift_time if data.shift_time is not None else data.t[-1]
            ax.plot([0, ts, ts, data.t[-1]], [tr[0], tr[0], tr[1], tr[1]], color=INK, linewidth=1.0, linestyle="--", label="truth")
            lo, hi = min(tr), max(tr)
            ax.set_ylim(lo - 0.9 * abs(hi), hi + 0.9 * abs(hi))
            if r == 0:
                ax.set_title(pname.replace("^2", "²"))
            if c == 0:
                ax.set_ylabel(TITLES[name].split(",")[0])
            if r == len(names) - 1:
                ax.set_xlabel("time [s]")
    axes[0][0].legend(loc="upper right")
    fig.suptitle("Physical parameters read from the model (seed 0); curves leaving the axes are off by more than 90 %",
                 fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out)
    plt.close(fig)


def plot_weights(ex: Dict[str, Dict[str, Any]], out: Path) -> None:
    """Share of each information direction that is given to integral CL (sequential ramp: one hue)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#f0efec", "#86b6ef", "#2a78d6", "#0d366b"])
    names = ("hanging", "swing")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    im = None
    for ax, name in zip(axes, names):
        run_ = ex[name]["E"]
        W = run_["w_log"].T                                   # (directions, time), sorted within each stack
        im = ax.imshow(W, aspect="auto", origin="lower", cmap=cmap, vmin=0.0, vmax=1.0,
                       extent=(run_["t_log"][0], run_["t_log"][-1], 0, W.shape[0]), interpolation="nearest")
        n_A = ex[name]["n_A"]
        ax.axhline(n_A, color="#fcfcfb", linewidth=2)
        ax.set_yticks([n_A / 2, n_A + (W.shape[0] - n_A) / 2], ["cart row", "pendulum row"], rotation=90, va="center")
        ax.grid(False)
        ax.set_title(TITLES[name])
        ax.set_xlabel("time [s]")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("share given to integral CL (0: instantaneous law)")
    fig.suptitle("Information-weighted blend: one row per eigen-direction of each stack, weakest at the bottom (seed 0)",
                 fontsize=10, fontweight="bold", color=INK)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    res_dir, fig_dir = ROOT_DIR / "results" / "approach_e", ROOT_DIR / "figures" / "approach_e"
    res_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(s, m, seed, False) for s in SCENARIOS for m in MODELS for seed in range(args.seeds)]
    log_jobs = [] if args.no_plots else [(s, m, 0, True) for s in SCENARIOS for m in ("B", "D", "E")]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex_:
        logged = list(ex_.map(job, log_jobs))
        rows = list(ex_.map(job, jobs))
    print(f"{len(jobs)} runs in {time.time() - t0:.0f} s")
    df = pd.DataFrame(rows)
    df.to_csv(res_dir / "runs.csv", index=False)
    cols = [c for c in df.columns if c not in ("model", "seed", "scenario", "bumper")]
    med = df.groupby(["scenario", "model"], sort=False)[cols].median()
    med.to_csv(res_dir / "summary_median.csv")
    rng_ = pd.concat([df.groupby(["scenario", "model"], sort=False)[cols].min().add_suffix("__lo"),
                      df.groupby(["scenario", "model"], sort=False)[cols].max().add_suffix("__hi")], axis=1)
    rng_.to_csv(res_dir / "summary_range.csv")
    with pd.option_context("display.width", 250, "display.float_format", "{:.4g}".format, "display.max_columns", 30):
        print(med.to_string())
        print("\nbumper reached:", df.groupby("scenario")["bumper"].any().to_dict())

    if not args.no_plots:
        _style()
        fig_dir.mkdir(parents=True, exist_ok=True)
        ex: Dict[str, Dict[str, Any]] = {}
        for name in SCENARIOS:
            ex[name] = {"data": scenario(name, 0), "n_A": HybridObserver(HybridObserverConfig()).cols_A.size}
        for (name, m, *_), r in zip(log_jobs, logged):
            ex[name][m] = r
        plot_velocity(ex, fig_dir / "velocity_error.png")
        plot_parameters(ex, fig_dir / "parameter_tracking.png")
        plot_weights(ex, fig_dir / "information_weights.png")
        print(f"Figures: {fig_dir}")
    print(f"Results: {res_dir}")


if __name__ == "__main__":
    main()
