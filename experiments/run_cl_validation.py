#!/usr/bin/env python3
"""
Approach C validation: concurrent-learning Lb-LSTM under weak (non-PE) excitation.

Stage 1 (parameter convergence, seed 0):
    Two non-PE training commands on the open-loop rig (encoder noise and quantization on):
        decaying      u(t) = 3 exp(-t / 8) sin(1.6 t)
        single_freq   u(t) = 1.5 env(t) sin(2 t)
    Compared: Approach C (CL loop on), the same observer with the CL loop off (standard
    instantaneous law, "Std law"), and Approach A. Logged: lambda_min(Omega) of the history
    stack vs the sliding-window excitation lambda_min of the instantaneous regressor h(t),
    the distance of the readout to the CL fixed point (stack least squares), the weight rate,
    and the online acceleration error.

Stage 2 (open-loop predictor benchmark, seeds 0..S-1, decaying training command):
    Every model is frozen at the end of training and driven open loop by four unseen inputs
    (step doublet, chirp, multisine, 1.2 rad/s sine). Primary metrics: one-step NMSE of the
    acceleration model and short-horizon (0.5 s re-initialized) NMSE of the positions.

Outputs:
    figures/approach_c/rank_condition.png        lambda_min(Omega) and parameter convergence
    figures/approach_c/prediction_benchmark.png  Approach A vs Approach C bar chart
    figures/approach_c/short_horizon_chirp.png   0.5 s predictions on the chirp test
    results/approach_c/predictor_benchmark.csv   per-seed metrics
    results/approach_c/predictor_summary.csv     median over seeds

Usage:
    python experiments/run_cl_validation.py [--seeds 3] [--workers 8] [--no-plots]
"""

from __future__ import annotations

import os

# One BLAS thread per worker: the per-step linear algebra is tiny and parallelism comes from
# the process pool; without this, BLAS oversubscribes the cores.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.concurrent_learning.scenarios import DT, simulate_weak_excitation
from src.identification.benchmark_predictor import (
    MODEL_A,
    MODEL_C,
    MODEL_C_FROZEN,
    MODEL_STD,
    PRIMARY_METRICS,
    evaluate_models,
    results_frame,
    simulate_test_suite,
    summary_table,
    train_models,
)
from src.observers.cl_lstm_observer import CLLbLSTMObserver

PE_WINDOW = 2.0          # [s] sliding window of the instantaneous excitation Gramian
DRIFT_LAG = 5.0          # [s] lag of the weight drift rate (averages out the noise jitter of the fast readout law)
TEST_ORDER = ("chirp", "multisine", "step_doublet", "sine_1.2rad/s")
TEST_LABELS = {"chirp": "Chirp", "multisine": "Multisine", "step_doublet": "Step doublet", "sine_1.2rad/s": "Sine 1.2 rad/s"}


# ---------------------------------------------------------------------- worker
@dataclass
class ConvergenceTrace:
    """Decimated (10 ms) adaptation history of one model on one training trajectory."""
    t: np.ndarray
    lambda_min: np.ndarray        # stack lambda_min(Omega) (zeros for Approach A)
    pe_lambda_min: np.ndarray     # lambda_min of int_{t-T}^{t} h h^T (CL observers only)
    fixed_point_dist: np.ndarray  # ||W_h - W_H||_F
    weight_rate: np.ndarray       # ||theta_h(t) - theta_h(t - DRIFT_LAG)|| / DRIFT_LAG
    accel_err: np.ndarray         # (K, 2) rolling RMS of Phi_hat - x_ddot over 2 s
    accel_rms: np.ndarray         # (2,) RMS of the true acceleration after t = 2 s
    rank_time: float              # first time lambda_min(Omega) > 0 (nan if never)


def sliding_pe_lambda_min(features: np.ndarray, dt: float, window: float, decim: int) -> np.ndarray:
    """lambda_min( int_{t - window}^{t} h h^T dtau ) sampled every `decim` samples."""
    N, L = features.shape
    K = N // decim
    blocks = np.einsum("kbi,kbj->kij", features[: K * decim].reshape(K, decim, L),
                       features[: K * decim].reshape(K, decim, L)) * dt
    csum = np.concatenate((np.zeros((1, L, L)), np.cumsum(blocks, axis=0)))
    w = max(1, int(round(window / (decim * dt))))
    out = np.zeros(K)
    for k in range(K):
        G = csum[k + 1] - csum[max(0, k + 1 - w)]
        out[k] = max(float(np.linalg.eigvalsh(G)[0]), 0.0)
    return out


def _rolling_rms(x: np.ndarray, window: int) -> np.ndarray:
    c = np.concatenate(([0.0], np.cumsum(x ** 2)))
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(0, idx - window)
    return np.sqrt((c[idx] - c[lo]) / (idx - lo))


def job(args: Tuple[str, int, str, bool]) -> Dict[str, object]:
    scenario, seed, model, evaluate = args
    t0 = time.time()
    traj, _ = simulate_weak_excitation(scenario, 60.0, noise_seed=42 + seed)
    trained = train_models(traj, seed, include=(model,))[model]
    tr = trained.trace
    out: Dict[str, object] = {"scenario": scenario, "seed": seed, "model": model}

    if seed == 0:
        decim = 10
        K = len(tr.t)
        lag = int(round(DRIFT_LAG / (decim * DT)))
        rate = np.full(K, np.nan)
        rate[lag:] = np.linalg.norm(tr.theta_h[lag:] - tr.theta_h[:-lag], axis=1) / DRIFT_LAG
        err = trained.trace.phi_hat - traj.accel
        w = int(round(2.0 / DT))
        accel_err = np.column_stack([_rolling_rms(err[:, j], w)[::decim][:K] for j in range(2)])
        is_cl = isinstance(trained.observer, CLLbLSTMObserver)
        pe = sliding_pe_lambda_min(tr.features, DT, PE_WINDOW, decim)[:K] if is_cl else np.zeros(K)
        pos = np.nonzero(tr.lambda_min > 0.0)[0]
        out["trace"] = ConvergenceTrace(
            t=tr.t, lambda_min=tr.lambda_min, pe_lambda_min=pe, fixed_point_dist=tr.fixed_point_dist,
            weight_rate=rate, accel_err=accel_err,
            accel_rms=np.sqrt(np.mean(traj.accel[traj.t >= 2.0] ** 2, axis=0)),
            rank_time=float(tr.t[pos[0]]) if len(pos) else float("nan"),
        )
        if is_cl:
            assert isinstance(trained.observer, CLLbLSTMObserver)
            s = trained.observer.stack
            out["stack"] = {
                "lambda_min": s.lambda_min(), "lambda_max": s.lambda_max(), "rank": s.rank(),
                **{k: getattr(s.stats, k) for k in ("considered", "appended", "replaced", "rejected_novelty", "rejected_no_gain")},
            }

    if evaluate:
        results = evaluate_models({model: trained.frozen}, simulate_test_suite())
        out["metrics"] = results_frame(results, seed)
        if seed == 0:
            out["chirp"] = next(
                {"t": r.truth.t, "theta_true": r.truth.state[:, 2], "theta_pred": r.short_horizon_state[:, 2]}
                for r in results if r.test == "chirp"
            )
    out["seconds"] = time.time() - t0
    return out


# ---------------------------------------------------------------------- plotting
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
# Reference categorical palette, fixed order (validated: scripts/validate_palette.js, light mode).
COLORS = {MODEL_A: "#2a78d6", MODEL_C: "#eb6834", MODEL_STD: "#1baf7a", MODEL_C_FROZEN: "#eda100"}
LABELS = {MODEL_A: "A: Lb-LSTM", MODEL_C: "C: CL-Lb-LSTM", MODEL_STD: "Std law (C, CL off)",
          MODEL_C_FROZEN: "C, frozen gates"}


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


def plot_rank_condition(traces: Dict[Tuple[str, str], ConvergenceTrace], out: Path) -> None:
    import matplotlib.pyplot as plt

    scenarios = ("decaying", "single_freq")
    titles = {"decaying": "Decaying command  3e^(-t/8) sin(1.6t)", "single_freq": "Single frequency  1.5 sin(2t)"}
    fig, axes = plt.subplots(4, 2, figsize=(10.5, 10.5), sharex=True)
    for col, sc in enumerate(scenarios):
        c = traces[(sc, MODEL_C)]
        cf = traces[(sc, MODEL_C_FROZEN)]
        std = traces[(sc, MODEL_STD)]
        a = traces[(sc, MODEL_A)]

        ax = axes[0, col]
        met = f", > 0 from t = {c.rank_time:.1f} s" if np.isfinite(c.rank_time) else ""
        ax.semilogy(c.t, np.maximum(c.lambda_min, 1e-8), color=COLORS[MODEL_C],
                    label=rf"history stack $\lambda_{{\min}}(\Omega)${met}")
        ax.semilogy(c.t, np.maximum(c.pe_lambda_min, 1e-8), color=INK_2, lw=1.0,
                    label=rf"instantaneous $\lambda_{{\min}}(\int_{{t-{PE_WINDOW:g}}}^{{t}} h h^T)$")
        if np.isfinite(c.rank_time):
            ax.axvline(c.rank_time, color=INK_2, ls=":", lw=0.9)
        ax.set_ylim(1e-8, 10.0)
        ax.set_title(titles[sc])
        ax.set_ylabel("excitation")
        ax.legend(loc="upper right")

        ax = axes[1, col]
        for m, tr in ((MODEL_STD, std), (MODEL_C, c), (MODEL_C_FROZEN, cf)):
            ax.semilogy(tr.t, tr.fixed_point_dist, color=COLORS[m], label=LABELS[m])
        ax.set_ylabel(r"$\|\hat W_h - W_{\mathcal{H}}\|_F$")
        ax.set_ylim(1e-3, 1e6)
        ax.legend(loc="upper right", ncol=3)

        ax = axes[2, col]
        for m, tr in ((MODEL_A, a), (MODEL_STD, std), (MODEL_C, c), (MODEL_C_FROZEN, cf)):
            ax.semilogy(tr.t, tr.weight_rate, color=COLORS[m], label=LABELS[m])
        ax.set_ylabel(rf"readout drift over {DRIFT_LAG:g} s  [1/s]")
        ax.set_ylim(1e-4, 10.0)
        ax.legend(loc="upper right", ncol=2)

        ax = axes[3, col]
        for m, tr in ((MODEL_A, a), (MODEL_STD, std), (MODEL_C, c)):
            ax.plot(tr.t, tr.accel_err[:, 1] / tr.accel_rms[1], color=COLORS[m], label=LABELS[m])
        ax.set_ylabel(r"rolling RMS($\hat\Phi_2-\ddot\theta$) / RMS($\ddot\theta$)")
        ax.set_xlabel("t [s]")
        ax.set_ylim(0, 1.2)
        ax.legend(loc="upper right")

    fig.text(0.01, 0.005, r"$W_{\mathcal{H}}$: least-squares readout on the observer's own history stack (the CL fixed point); "
             "both observers record a stack, only C adapts on it.", fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def plot_benchmark(df: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    panels = [
        ("one-step NMSE theta_ddot", r"one-step NMSE  $\ddot\theta$"),
        ("one-step NMSE x_ddot", r"one-step NMSE  $\ddot x$"),
        ("short-horizon NMSE theta", r"0.5 s horizon NMSE  $\theta$"),
        ("short-horizon NMSE x", r"0.5 s horizon NMSE  $x$"),
    ]
    models = (MODEL_A, MODEL_C)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6))
    x = np.arange(len(TEST_ORDER))
    width = 0.36
    for ax, (metric, title) in zip(axes.ravel(), panels):
        for i, m in enumerate(models):
            g = df[df.model == m].groupby("test")[metric]
            med = np.array([g.median()[t] for t in TEST_ORDER])
            lo = np.array([g.min()[t] for t in TEST_ORDER])
            hi = np.array([g.max()[t] for t in TEST_ORDER])
            pos = x + (i - 0.5) * (width + 0.02)
            ax.bar(pos, med, width, color=COLORS[m], label=LABELS[m], edgecolor="white", linewidth=2)
            ax.errorbar(pos, med, yerr=[med - lo, hi - med], fmt="none", ecolor=INK_2, elinewidth=0.8, capsize=2)
            for p, v, top in zip(pos, med, hi):
                ax.annotate(f"{v:.2g}", (p, top), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_yscale("log")
        ax.margins(y=0.15)
        ax.axhline(1.0, color=INK_2, lw=0.8, ls="--")
        ax.set_xticks(x, [TEST_LABELS[t] for t in TEST_ORDER])
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0, 0].legend(loc="upper left")
    fig.text(0.01, 0.005, "Bars: median over seeds; whiskers: min-max. Dashed line: NMSE = 1 (predicting the mean). "
             "Lower is better. Training: decaying non-PE command only.", fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def plot_chirp(chirp: Dict[str, Dict[str, np.ndarray]], out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    ref = chirp[MODEL_A]
    ax.plot(ref["t"], ref["theta_true"] - np.pi, color=INK, lw=1.8, label="true plant")
    for m in (MODEL_A, MODEL_C):
        ax.plot(chirp[m]["t"], chirp[m]["theta_pred"] - np.pi, color=COLORS[m], lw=1.1, label=LABELS[m])
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\theta - \pi$  [rad]")
    ax.set_title("Chirp test (unseen): open-loop predictions restarted from the true state every 0.5 s")
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    models = (MODEL_A, MODEL_C, MODEL_STD)
    jobs = [("decaying", s, m, True) for s in range(args.seeds) for m in models]
    jobs += [("single_freq", 0, m, False) for m in models]
    jobs += [(sc, 0, MODEL_C_FROZEN, False) for sc in ("decaying", "single_freq")]

    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        outs = list(ex.map(job, jobs))
    print(f"{len(jobs)} runs in {time.time() - t0:.0f} s")

    res_dir = ROOT_DIR / "results" / "approach_c"
    fig_dir = ROOT_DIR / "figures" / "approach_c"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1 summary
    traces: Dict[Tuple[str, str], ConvergenceTrace] = {}
    print("\n=== Stage 1: rank condition and parameter convergence (seed 0) ===")
    for o in outs:
        if "trace" not in o:
            continue
        tr = o["trace"]
        assert isinstance(tr, ConvergenceTrace)
        traces[(str(o["scenario"]), str(o["model"]))] = tr
        late = tr.t >= 40.0
        line = (f"{o['scenario']:<12}{o['model']:<20} final ||W_h - W_H|| = {tr.fixed_point_dist[-1]:8.3g}   "
                f"median drift rate (t>=40 s) = {np.nanmedian(tr.weight_rate[late]):8.3g}   "
                f"theta_ddot rel. error (t>=40 s) = {np.median(tr.accel_err[late, 1]) / tr.accel_rms[1]:.3f}")
        if o["model"] in (MODEL_C, MODEL_C_FROZEN):
            st = o["stack"]
            assert isinstance(st, dict)
            line += (f"\n{'':<32} rank time {tr.rank_time:.2f} s, lambda_min = {st['lambda_min']:.3g}, "
                     f"lambda_max = {st['lambda_max']:.3g}, rank {st['rank']}, appended {st['appended']}, "
                     f"replaced {st['replaced']}, rejected {st['rejected_novelty']} (novelty) / {st['rejected_no_gain']} (no gain); "
                     f"instantaneous lambda_min at t=60 s: {tr.pe_lambda_min[-1]:.3g}")
        print(line)

    # Stage 2 summary
    df = pd.concat([o["metrics"] for o in outs if "metrics" in o], ignore_index=True)  # type: ignore[misc]
    df.to_csv(res_dir / "predictor_benchmark.csv", index=False)
    summ = summary_table(df)
    summ.to_csv(res_dir / "predictor_summary.csv", index=False)
    print("\n=== Stage 2: open-loop predictor benchmark (median over seeds) ===")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(summ.to_string(index=False))
        wins = 0
        cells = 0
        for t in TEST_ORDER:
            for m in PRIMARY_METRICS:
                a = summ[(summ.test == t) & (summ.model == MODEL_A)][m].iloc[0]
                c = summ[(summ.test == t) & (summ.model == MODEL_C)][m].iloc[0]
                wins += int(c < a)
                cells += 1
        print(f"\nApproach C better than Approach A in {wins}/{cells} (test, metric) cells.")
        print("\nSecondary: full-length free-run NMSE (median)")
        print(df.groupby(["test", "model"], sort=False)[["rollout NMSE x", "rollout NMSE theta"]].median().to_string())

    if not args.no_plots:
        _style()
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_rank_condition(traces, fig_dir / "rank_condition.png")
        plot_benchmark(df[df.model.isin([MODEL_A, MODEL_C])], fig_dir / "prediction_benchmark.png")
        chirp = {str(o["model"]): o["chirp"] for o in outs if "chirp" in o}
        plot_chirp(chirp, fig_dir / "short_horizon_chirp.png")  # type: ignore[arg-type]
        print(f"\nFigures: {fig_dir}")
    print(f"Results: {res_dir}")


if __name__ == "__main__":
    main()
