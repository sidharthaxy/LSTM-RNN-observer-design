#!/usr/bin/env python3
"""
Approach D validation: physics-structured integral-CL observer (PI-ICL) vs Approach B (PI-LSTM)
and Approach A (black-box Lb-LSTM) on Approach B's extreme-condition scenario.

Scenario (identical to experiments/run_pilstm_validation.py; its functions are reused):
    50 s at 1 kHz, open loop; theta(0) = 0.5 rad from upright (the pendulum falls and swings
    through most of the circle); U(+-1 deg) encoder noise plus cart noise and quantization;
    pendulum mass and inertia +50 % at t = 25 s; u = 2 sin 3t + 1.2 cos 6t.

Models:
    A          black-box Lb-LSTM (Approach A defaults, B's data normalization)
    B          PI-LSTM (Approach B defaults)
    D          PI-ICL (this branch's defaults)
    D, CL off  the same observer with integral CL and change detection disabled (ablation)

Reported:
    1. Velocity reconstruction per window (B's metric set).
    2. Identification: inertia error ||M_hat(q_hat) - M(q)|| / ||M(q)|| at 25 s and 50 s (B's
       metric), and D's physical parameters (M+m, ml, I+ml^2, mgl, b, d) against the truth
       before and after the mass step.
    3. Frozen models released from rest (u = 0) vs the true post-shift rigid body (B's free swing).
    4. --seeds N: medians and ranges over noise / initialization seeds.

Outputs: figures/approach_d/*.png, results/approach_d/*.csv
Usage:   python experiments/run_d_validation.py [--seeds 5] [--workers 8] [--no-plots]
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
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from experiments.run_pilstm_validation import (  # noqa: E402  (Approach B's protocol, reused unchanged)
    DT,
    LOG_EVERY,
    Scenario,
    Trajectory,
    _rk4,
    _x0,
    free_swing,
    lblstm_config,
    normalization,
    pilstm_config,
    run_lblstm,
    run_pilstm,
    simulate_scenario,
    velocity_metrics,
)
from src.observers.el_linear_model import LinearELModel  # noqa: E402
from src.observers.pi_icl_observer import PIICLObserver, PIICLObserverConfig  # noqa: E402
from src.plant.pendulum_plant import PendulumParameters, PendulumPlant  # noqa: E402

MODELS = ("A", "B", "D", "D, CL off")
PHYS = ("M+m", "ml", "I+ml^2", "mgl")


# ---------------------------------------------------------------------- D run
@dataclass
class DRun:
    velocity: np.ndarray        # (N, 2)
    phi: np.ndarray             # (N, 2)
    t_log: np.ndarray           # (K,)
    M_hat: np.ndarray           # (K, 2, 2) at q_hat
    phys: Dict[str, np.ndarray]  # physical reading of theta_hat, (K,) each
    lam_A: np.ndarray           # (K,) lambda_min, actuated stack
    lam_U: np.ndarray           # (K,) lambda_min, normalized unactuated regression
    n_A: np.ndarray             # (K,) actuated stack size
    purges: List[float]
    theta: np.ndarray
    diverged_at: float | None


def run_d(traj: Trajectory, cfg: PIICLObserverConfig) -> DRun:
    obs = PIICLObserver(cfg)
    obs.reset(_x0(traj))
    N = len(traj.t)
    K = (N + LOG_EVERY - 1) // LOG_EVERY
    vel, phi = np.full((N, 2), np.nan), np.full((N, 2), np.nan)
    M_hat = np.full((K, 2, 2), np.nan)
    phys = {k: np.full(K, np.nan) for k in PHYS}
    lam_A, lam_U, n_A = np.zeros(K), np.zeros(K), np.zeros(K, dtype=int)
    diverged = None
    for k in range(N):
        est = obs.update(traj.meas[k], traj.u[k], DT)
        if not np.all(np.isfinite(est)) or np.abs(est).max() > 1e3:
            diverged = float(traj.t[k])
            break
        vel[k] = est[[1, 3]]
        phi[k] = obs.phi_hat
        if k % LOG_EVERY == 0:
            j = k // LOG_EVERY
            M_hat[j] = obs.M_hat
            for key, v in obs.physical_parameters().items():
                if key in phys:
                    phys[key][j] = v
            if k % (10 * LOG_EVERY) == 0:           # rank diagnostics every 100 ms
                lm = obs.lambda_mins()
                lam_A[j], lam_U[j] = lm["actuated"], lm["row 1 (normalized)"]
            else:
                lam_A[j], lam_U[j] = lam_A[j - 1], lam_U[j - 1]
            n_A[j] = len(obs.stack_A)
    return DRun(vel, phi, traj.t[::LOG_EVERY], M_hat, phys, lam_A, lam_U, n_A, list(obs.purge_times),
                np.copy(obs.theta), diverged)


def d_free_swing(cfg: PIICLObserverConfig, theta: np.ndarray, t: np.ndarray, q0: np.ndarray) -> np.ndarray:
    """Frozen D model (rigid-body part, u = 0) released from rest, RK4; returns (N, 4) interleaved."""
    model = PIICLObserver(cfg).model
    u0 = np.zeros(1)

    def f(z: np.ndarray) -> np.ndarray:
        return np.concatenate((z[2:], model.conservative_acceleration(theta, z[:2], z[2:], u0)))

    z = np.concatenate((q0, np.zeros(2)))
    out = np.zeros((t.size, 4))
    for k in range(t.size):
        out[k] = z[[0, 2, 1, 3]]
        z = _rk4(f, z, DT)
    return out


def d_energy(cfg: PIICLObserverConfig, theta: np.ndarray, traj4: np.ndarray) -> np.ndarray:
    model = PIICLObserver(cfg).model
    return np.array([model.kinetic_energy(theta, z[[0, 2]], z[[1, 3]]) + model.potential_energy(theta, z[[0, 2]])
                     for z in traj4])


# ---------------------------------------------------------------------- metrics
def inertia_errors(traj: Trajectory, M_hat_log: np.ndarray) -> Dict[str, float]:
    M_true = traj.M_true[::LOG_EVERY]
    out = {}
    for label, tq in (("25 s", 25.0 - DT), ("50 s", float(traj.t[-1]))):
        j = min(round(tq / DT) // LOG_EVERY, len(M_hat_log) - 1)
        out[f"M_hat rel. error @ {label}"] = float(np.linalg.norm(M_hat_log[j] - M_true[j]) / np.linalg.norm(M_true[j]))
    return out


def phys_errors(run: DRun, t_log: np.ndarray, p_pre: PendulumParameters, p_post: PendulumParameters) -> Dict[str, float]:
    out = {}
    for label, tq, p in (("25 s", 25.0 - DT, p_pre), ("50 s", float(t_log[-1]), p_post)):
        j = min(int(np.searchsorted(t_log, tq)), len(t_log) - 1)
        truth = LinearELModel.physical_truth(p)
        for key in PHYS:
            out[f"{key} rel. err @ {label}"] = float(abs(run.phys[key][j] - truth[key]) / truth[key])
    return out


def online_nmse(traj: Trajectory, phi: np.ndarray) -> Dict[str, float]:
    fin = np.all(np.isfinite(phi), axis=1) & (traj.t >= 35.0)
    nmse = np.mean((phi[fin] - traj.accel[fin]) ** 2, axis=0) / np.var(traj.accel[fin], axis=0)
    return {"online accel NMSE x_ddot (steady)": float(nmse[0]), "online accel NMSE th_ddot (steady)": float(nmse[1])}


# ---------------------------------------------------------------------- worker
def d_config(**overrides: Any) -> PIICLObserverConfig:
    return replace(PIICLObserverConfig(), **overrides)


def job(args: Tuple[int, str]) -> Dict[str, Any]:
    seed, model = args
    t0 = time.time()
    sc = Scenario(noise_seed=42 + seed)
    traj = simulate_scenario(sc)
    norm = normalization(traj)
    row: Dict[str, Any] = {"seed": seed, "model": model}
    extra: Dict[str, Any] = {}
    if model == "A":
        run = run_lblstm(traj, lblstm_config(norm, seed))
        vel, phi, diverged = run.velocity, run.phi, run.diverged_at
        extra["theta"], extra["cfg"] = run.theta, lblstm_config(norm, seed)
    elif model == "B":
        cfg_b = pilstm_config(norm, seed)
        run = run_pilstm(traj, cfg_b)
        vel, phi, diverged = run.velocity, run.phi, run.diverged_at
        row.update(inertia_errors(traj, run.M_hat))
        extra["theta"], extra["cfg"] = run.theta, cfg_b
        if seed == 0:
            extra["M_hat"] = run.M_hat
    else:
        cfg_d = d_config(seed=seed) if model == "D" else d_config(seed=seed, cl_enabled=False, detect_changes=False)
        run_d_ = run_d(traj, cfg_d)
        vel, phi, diverged = run_d_.velocity, run_d_.phi, run_d_.diverged_at
        row.update(inertia_errors(traj, run_d_.M_hat))
        row.update(phys_errors(run_d_, run_d_.t_log, PendulumParameters(), traj.params_after))
        row["purges"] = ";".join(f"{t:.2f}" for t in run_d_.purges)
        row["first purge after step [s]"] = next((t - 25.0 for t in run_d_.purges if t >= 25.0), np.nan)
        row["false purges"] = sum(1 for t in run_d_.purges if t < 25.0 or t > 30.0)
        extra["theta"], extra["cfg"] = run_d_.theta, cfg_d
        if seed == 0:
            extra["drun"] = run_d_
    row["diverged"] = float(diverged is not None)
    if diverged is None:
        row.update(velocity_metrics(traj, vel))
        row.update(online_nmse(traj, phi))
    if seed == 0:
        extra["velocity"] = vel
        extra["traj_t"] = traj.t
        extra["v_true"] = traj.velocities
        extra["params_after"] = traj.params_after
        extra["M_true"] = traj.M_true[::LOG_EVERY]
    row["seconds"] = time.time() - t0
    return {"row": row, "extra": extra}


# ---------------------------------------------------------------------- plotting
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"A": "#2a78d6", "B": "#eb6834", "D": "#1baf7a", "D, CL off": "#eda100"}
LABELS = {"A": "A: Lb-LSTM", "B": "B: PI-LSTM", "D": "D: PI-ICL", "D, CL off": "D, CL off"}


def _style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 1.4, "font.family": "DejaVu Sans",
    })


def plot_parameters(ex: Dict[str, Dict[str, Any]], out: Path) -> None:
    import matplotlib.pyplot as plt
    d = ex["D"]["drun"]
    dn = ex["D, CL off"]["drun"]
    assert isinstance(d, DRun) and isinstance(dn, DRun)
    p0, p1 = LinearELModel.physical_truth(PendulumParameters()), LinearELModel.physical_truth(ex["D"]["params_after"])
    t = d.t_log
    M_B = ex["B"]["M_hat"]
    assert isinstance(M_B, np.ndarray)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2), sharex=True)
    for ax, key in zip(axes.ravel(), PHYS):
        truth = np.where(t < 25.0, p0[key], p1[key])
        ax.plot(t, truth, color=INK, lw=1.8, label="true")
        ax.plot(t, dn.phys[key], color=COLORS["D, CL off"], label=LABELS["D, CL off"])
        ax.plot(t, d.phys[key], color=COLORS["D"], label=LABELS["D"])
        if key == "M+m":
            ax.plot(t, M_B[:, 0, 0], color=COLORS["B"], lw=1.0, label=r"B: $\hat M_{11}(\hat q)$")
        if key == "I+ml^2":
            ax.plot(t, M_B[:, 1, 1], color=COLORS["B"], lw=1.0, label=r"B: $\hat M_{22}(\hat q)$")
        for tp in d.purges:
            ax.axvline(tp, color=INK_2, ls=":", lw=0.9)
        ax.set_title(key)
        lo, hi = min(p0[key], p1[key]), max(p0[key], p1[key])
        ax.set_ylim(max(0.0, lo - 0.6 * (hi - lo) - 0.2 * lo), hi + 0.6 * (hi - lo) + 0.2 * hi)
        ax.legend(loc="lower right", ncol=2)
    for ax in axes[1]:
        ax.set_xlabel("t [s]")
    fig.text(0.01, 0.005, "Pendulum mass and inertia +50 % at t = 25 s. Dotted: D's stack purges (change detection). Seed 0.",
             fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def plot_rank(ex: Dict[str, Dict[str, Any]], out: Path) -> None:
    import matplotlib.pyplot as plt
    d = ex["D"]["drun"]
    assert isinstance(d, DRun)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10.5, 4.8), sharex=True)
    ax.semilogy(d.t_log, np.maximum(d.lam_A, 1e-9), color=COLORS["D"], label=r"actuated rows: $\lambda_{\min}(\Omega_A)$")
    ax.semilogy(d.t_log, np.maximum(d.lam_U, 1e-9), color=INK_2, label=r"pendulum row, normalized: $\lambda_{\min}([z, Y_{own}]^T[z, Y_{own}])$")
    ax.axhline(PIICLObserverConfig().lambda_bar, color=INK_2, ls="--", lw=0.8)
    ax.text(0.3, PIICLObserverConfig().lambda_bar * 1.3, r"$\bar\lambda$ (CL gate)", fontsize=7.5, color=INK_2)
    for tp in d.purges:
        ax.axvline(tp, color=INK_2, ls=":", lw=0.9)
        ax2.axvline(tp, color=INK_2, ls=":", lw=0.9)
    ax.set_ylim(1e-8, 1e-1)
    ax.set_ylabel("excitation")
    ax.legend(loc="lower right")
    ax.set_title("Rank condition of the two integral-CL stacks (seed 0)")
    ax2.plot(d.t_log, d.n_A, color=COLORS["D"])
    ax2.set_ylabel("stored windows")
    ax2.set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_velocity(ex: Dict[str, Dict[str, Any]], out: Path) -> None:
    import matplotlib.pyplot as plt
    t = ex["D"]["traj_t"]
    v_true = ex["D"]["v_true"]
    assert isinstance(t, np.ndarray) and isinstance(v_true, np.ndarray)
    w = int(round(0.5 / DT))
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.0), sharex=True)
    for j, lab in enumerate((r"$\dot x$ error [m/s]", r"$\dot\theta$ error [rad/s]")):
        ax = axes[j]
        for m in ("A", "B", "D"):
            v = ex[m]["velocity"]
            assert isinstance(v, np.ndarray)
            e2 = np.nan_to_num(v[:, j] - v_true[:, j]) ** 2
            ax.semilogy(t, np.sqrt(np.convolve(e2, np.ones(w) / w, mode="same")), color=COLORS[m], label=LABELS[m])
        ax.axvline(25.0, color=INK_2, ls=":", lw=0.9)
        ax.set_ylabel("0.5 s RMS " + lab)
        ax.legend(loc="upper right", ncol=3)
    axes[0].set_title("Velocity reconstruction (seed 0); mass step at 25 s")
    axes[1].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_free_swing(fs_t: np.ndarray, truth: np.ndarray, twins: Dict[str, np.ndarray], out: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.plot(fs_t, truth[:, 2] - np.pi, color=INK, lw=1.8, label="true (post-shift rigid body)")
    for m, z in twins.items():
        ax.plot(fs_t, z[:, 2] - np.pi, color=COLORS[m], lw=1.1, label=LABELS[m])
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\theta - \pi$ [rad]")
    ax.set_title("Frozen models (t = 50 s) released from rest at theta = pi - 1, u = 0 (seed 0)")
    ax.legend(loc="lower left", ncol=4)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    jobs = [(s, m) for s in range(args.seeds) for m in MODELS]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        outs = list(ex.map(job, jobs))
    print(f"{len(jobs)} runs in {time.time() - t0:.0f} s")

    res_dir = ROOT_DIR / "results" / "approach_d"
    fig_dir = ROOT_DIR / "figures" / "approach_d"
    res_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([o["row"] for o in outs])
    df.to_csv(res_dir / "seeds.csv", index=False)

    cols = [
        "diverged", "RMSE x_dot pre-shift 15-25 s", "RMSE th_dot pre-shift 15-25 s", "RMSE th_dot post-shift 25-30 s",
        "RMSE x_dot steady 35-50 s", "RMSE th_dot steady 35-50 s", "peak |th_dot err| 0-2 s", "peak |th_dot err| 25-27 s",
        "online accel NMSE th_ddot (steady)", "M_hat rel. error @ 25 s", "M_hat rel. error @ 50 s",
    ]
    med = df.groupby("model", sort=False)[cols].median()
    lo = df.groupby("model", sort=False)[cols].min()
    hi = df.groupby("model", sort=False)[cols].max()
    summary = med.T
    summary.to_csv(res_dir / "summary_median.csv")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print("\n=== median over seeds ===")
        print(summary.to_string())
        print("\n=== min / max ===")
        print(pd.concat({"min": lo.T, "max": hi.T}, axis=1).to_string())
        d_rows = df[df.model == "D"]
        pcols = [c for c in df.columns if "rel. err @" in c]
        print("\n=== D physical parameters: relative error (median [min, max]) ===")
        for c in pcols:
            v = d_rows[c]
            print(f"  {c:28s} {v.median():.3f} [{v.min():.3f}, {v.max():.3f}]")
        print("  first purge after the step [s]:", d_rows["first purge after step [s]"].round(2).tolist())
        print("  false purges per seed:", d_rows["false purges"].tolist())
        dn = df[df.model == "D, CL off"]
        print("\n=== D, CL off: physical parameters at 50 s (median rel. err) ===")
        for c in pcols:
            print(f"  {c:28s} {dn[c].median():.3f}")

    # seed-0 extras
    ex0: Dict[str, Dict[str, Any]] = {str(o["row"]["model"]): o["extra"] for o in outs if o["row"]["seed"] == 0}

    # free swing (seed 0): B's function for A and B, D computed on the same grid and truth
    fs = free_swing(ex0["B"]["cfg"], ex0["B"]["theta"], ex0["A"]["cfg"], ex0["A"]["theta"],
                    ex0["D"]["params_after"])
    q0 = np.array([0.0, np.pi - 1.0])
    d_tw = d_free_swing(ex0["D"]["cfg"], ex0["D"]["theta"], fs.t, q0)
    true_p = replace(ex0["D"]["params_after"], b=0.0, d=0.0, x_limit=np.inf)
    plant = PendulumPlant(true_p)
    H_true_d = np.array([plant.total_energy(z) for z in d_tw])
    H_d = d_energy(ex0["D"]["cfg"], ex0["D"]["theta"], d_tw)
    fs_rows = []
    for m, z, Ht, Hl in (("A", fs.a_twin, fs.H_true_a, None), ("B", fs.pi_twin, fs.H_true_pi, fs.H_pi), ("D", d_tw, H_true_d, H_d)):
        ok = np.all(np.isfinite(z), axis=1)
        err = np.sqrt(np.mean((z[ok, 2] - fs.truth[ok, 2]) ** 2)) if ok.any() else np.nan
        row = {"model": m, "theta RMSE [rad]": err,
               "true energy drift [J]": float(np.nanmax(np.abs(Ht - Ht[0]))),
               "learned-energy drift [J]": float(np.max(np.abs(Hl - Hl[0]))) if Hl is not None else np.nan,
               "swing energy scale [J]": fs.T_peak}
        fs_rows.append(row)
    fs_df = pd.DataFrame(fs_rows)
    fs_df.to_csv(res_dir / "free_swing.csv", index=False)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print("\n=== frozen-model free swing (seed 0) ===")
        print(fs_df.to_string(index=False))

    if not args.no_plots:
        _style()
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_parameters(ex0, fig_dir / "parameter_tracking.png")
        plot_rank(ex0, fig_dir / "rank_condition.png")
        plot_velocity(ex0, fig_dir / "velocity_error.png")
        plot_free_swing(fs.t, fs.truth, {"A": fs.a_twin, "B": fs.pi_twin, "D": d_tw}, fig_dir / "free_swing.png")
        print(f"\nFigures: {fig_dir}")
    print(f"Results: {res_dir}")


if __name__ == "__main__":
    main()
