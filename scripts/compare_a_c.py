#!/usr/bin/env python3
"""
Approach A (black-box Lb-LSTM) against Approach C (the same observer with concurrent learning),
on the shared benchmark: the same scenarios, seeds, normalization and metrics for both.

The x-free variants (A-x, C-x: cart position removed from the network input) are shown alongside,
because C's advantage on held-out prediction is largest when it is combined with that change.

Reads   results/shared_benchmark/velocity_runs.csv and held_out_twins.csv
        (written by experiments/run_shared_benchmark.py)
Writes  figures/comparisons/a_vs_c.png and prints the median table used in the README.
        figures/comparisons/a_vs_c_velocity_error.png: A and C rerun on S1-S3 (seed 0, the
        benchmark's harness and settings) and their angular-velocity error plotted over time.

Usage:  python scripts/compare_a_c.py [--bars-only]
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "results" / "shared_benchmark"
OUT = ROOT / "figures" / "comparisons" / "a_vs_c.png"
OUT_TS = ROOT / "figures" / "comparisons" / "a_vs_c_velocity_error.png"
TS_MODELS = ("A", "C")
TS_TITLES = {"S1": "S1: hanging, small swings", "S2": "S2: swing-through, +50 % mass at 25 s",
             "S3": "S3: weak, fading input"}
TS_LABELS = {"A": "A: Lb-LSTM", "C": "C: CL-Lb-LSTM"}

MODELS = ("A", "C", "A-x", "C-x")
SCENARIOS = ("S1", "S2", "S3")
# Shared-benchmark colours (reference categorical palette, fixed slots): A 1, C 3, A-x 5, C-x 6.
COLORS = {"A": "#2a78d6", "C": "#1baf7a", "A-x": "#e87ba4", "C-x": "#008300"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

PANELS = (
    ("velocity", "RMSE th_dot steady", r"steady $\dot\theta$ RMSE [rad/s]"),
    ("velocity", "RMSE th_dot transient", r"$\dot\theta$ RMSE, 0.1-5 s [rad/s]"),
    ("held-out", "one-step NMSE theta_ddot", r"held-out: one-step NMSE $\ddot\theta$"),
    ("held-out", "0.5 s NMSE theta", r"held-out: 0.5 s prediction NMSE $\theta$"),
)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(RES / "velocity_runs.csv")
    runs = runs[~runs["warm start"].astype(bool) & runs.model.isin(MODELS)]
    twins = pd.read_csv(RES / "held_out_twins.csv")
    return runs, twins[twins.model.isin(MODELS)]


def summary(runs: pd.DataFrame, twins: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind, col, _ in PANELS:
        df = runs if kind == "velocity" else twins
        med = df.groupby(["scenario", "model"])[col].median().unstack("model")[list(MODELS)]
        for s in SCENARIOS:
            rows.append({"metric": col, "scenario": s, **med.loc[s].to_dict()})
    return pd.DataFrame(rows)


def _style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": INK_2, "ytick.color": INK_2,
        "legend.frameon": False, "legend.fontsize": 8, "font.family": "DejaVu Sans",
    })


def plot(runs: pd.DataFrame, twins: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(16.5, 3.9))
    x = np.arange(len(SCENARIOS))
    width = 0.8 / len(MODELS)
    for ax, (kind, col, title) in zip(axes, PANELS):
        df = runs if kind == "velocity" else twins
        g = df.groupby(["scenario", "model"])[col]
        med, lo, hi = g.median(), g.min(), g.max()
        for i, m in enumerate(MODELS):
            md = np.array([med.get((s, m), np.nan) for s in SCENARIOS])
            pos = x - 0.4 + (i + 0.5) * width
            ax.bar(pos, md, width * 0.92, color=COLORS[m], label=m, edgecolor="white", linewidth=1)
            ax.errorbar(pos, md, yerr=[md - np.array([lo.get((s, m), np.nan) for s in SCENARIOS]),
                                       np.array([hi.get((s, m), np.nan) for s in SCENARIOS]) - md],
                        fmt="none", ecolor=INK_2, elinewidth=0.7, capsize=1.5)
        ax.set_yscale("log")
        ax.set_xticks(x, list(SCENARIOS))
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(MODELS), bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("Approach A vs Approach C on the shared benchmark (cold start)", fontsize=10, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "Bars: median over 5 seeds (and over the 4 held-out inputs for prediction); whiskers: min-max. "
             "S1 = hanging two-tone (A's home), S2 = swing-through with mass step, S3 = weak non-PE input (C's home).",
             fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.88))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)


def velocity_error(args: Tuple[str, str, int]) -> Tuple[str, str, np.ndarray, np.ndarray, float | None]:
    """One online run of the benchmark harness (cold start); returns the theta_dot error over time."""
    from src.benchmark.models import make_observer
    from src.benchmark.scenarios import DT, TRAINING
    scenario, model, seed = args
    data = TRAINING[scenario](seed)
    obs = make_observer(model, data, seed)
    obs.reset(np.array([data.meas[0, 0], 0.0, data.meas[0, 1], 0.0]))
    err = np.full(len(data.t), np.nan)
    for k in range(len(data.t)):
        est = obs.update(data.meas[k], data.u[k], DT)
        err[k] = est[3] - data.velocities[k, 1]
    return scenario, model, data.t, err, data.shift_time


def _moving_rms(x: np.ndarray, window: int) -> np.ndarray:
    return np.sqrt(np.convolve(x ** 2, np.ones(window) / window, mode="same"))


def plot_velocity_error(seed: int = 0) -> None:
    import matplotlib.pyplot as plt
    jobs = [(s, m, seed) for s in SCENARIOS for m in TS_MODELS]
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        results = list(pool.map(velocity_error, jobs))
    _style()
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(13.5, 3.6))
    for ax, s in zip(axes, SCENARIOS):
        for scenario, model, t, err, shift in results:
            if scenario != s:
                continue
            ax.plot(t, _moving_rms(np.nan_to_num(err), 500), color=COLORS[model], label=TS_LABELS[model], linewidth=1.6)
            if shift is not None:
                ax.axvline(shift, color=INK_2, linewidth=0.8, linestyle=":")
        ax.set_yscale("log")
        ax.set_title(TS_TITLES[s])
        ax.set_xlabel("time [s]")
    axes[0].set_ylabel(r"$\dot\theta$ error, 0.5 s RMS [rad/s]")
    axes[0].legend(loc="upper right")
    fig.suptitle(f"Angular-velocity estimation error over time, Approach A vs Approach C (seed {seed})",
                 fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT_TS.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_TS)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bars-only", action="store_true", help="skip the six simulations behind the error-over-time figure")
    args = parser.parse_args()
    runs, twins = load()
    tab = summary(runs, twins)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(tab.to_string(index=False))
    plot(runs, twins)
    print(f"Figure: {OUT}")
    if not args.bars_only:
        plot_velocity_error()
        print(f"Figure: {OUT_TS}")


if __name__ == "__main__":
    main()
