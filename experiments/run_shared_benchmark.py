#!/usr/bin/env python3
"""
Shared benchmark of Approaches A, B, C, D and E (improvements 1-3).

1. One harness: every approach (branch defaults) is trained on every branch's home scenario
   (S1 hanging two-sine = A's, S2 swing-through + mass step = B's, S3 weak excitation = C's),
   5 seeds each, and scored with the same metrics:
     * velocity reconstruction: RMSE of x_dot and theta_dot over the transient (0.1-5 s) and the
       steady window, peak error 0.1-2 s, post-shift RMSE (S2);
     * held-out prediction: the frozen model (twin) on four inputs never used by any branch,
       one-step NMSE of the accelerations and 0.5 s short-horizon NMSE of the positions.
2. Cart position removed from the inputs of the black-box models: A-x, C-x.
3. Model-free warm start (WarmStartObserver) for every model: cold vs warm transient metrics.

Usage:  python experiments/run_shared_benchmark.py [--seeds 5] [--workers 8] [--no-plots]
        python experiments/run_shared_benchmark.py --only E     # run only the listed models and merge them into the saved CSVs
Outputs: results/shared_benchmark/*.csv, figures/shared_benchmark/*.png
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.benchmark.models import MODELS, make_observer, make_twin, twin_metrics  # noqa: E402
from src.benchmark.scenarios import DT, STEADY_WINDOW, TRAINING, held_out_tests  # noqa: E402
from src.benchmark.warm_start import WarmStartObserver  # noqa: E402

SCENARIOS = tuple(TRAINING)
T_SKIP = 0.1            # transient metrics start after the warm-start window (same for cold and warm runs)
TWIN_METRICS = ("one-step NMSE x_ddot", "one-step NMSE theta_ddot", "0.5 s NMSE x", "0.5 s NMSE theta")


def job(args: Tuple[str, str, int, bool]) -> List[Dict[str, Any]]:
    scenario, model, seed, warm = args
    t0 = time.time()
    data = TRAINING[scenario](seed)
    obs = make_observer(model, data, seed)
    runner = WarmStartObserver(obs, DT) if warm else obs
    runner.reset(np.array([data.meas[0, 0], 0.0, data.meas[0, 1], 0.0]))
    N = len(data.t)
    vel = np.full((N, 2), np.nan)
    diverged = None
    for k in range(N):
        est = runner.update(data.meas[k], data.u[k], DT)
        if not np.all(np.isfinite(est)) or np.abs(est).max() > 1e3:
            diverged = float(data.t[k])
            break
        vel[k] = est[[1, 3]]

    base = {"scenario": scenario, "model": model, "seed": seed, "warm start": warm}
    row: Dict[str, Any] = {**base, "diverged": float(diverged is not None)}
    err = vel - data.velocities
    t = data.t
    windows = {"transient": (T_SKIP, 5.0), "steady": STEADY_WINDOW[scenario]}
    if data.shift_time is not None:
        windows["post-shift"] = (data.shift_time, data.shift_time + 5.0)
    for wname, (a, b) in windows.items():
        m = (t >= a) & (t < b)
        rmse = np.sqrt(np.nanmean(err[m] ** 2, axis=0))
        row[f"RMSE x_dot {wname}"], row[f"RMSE th_dot {wname}"] = float(rmse[0]), float(rmse[1])
    m = (t >= T_SKIP) & (t < 2.0)
    row["peak x_dot err 0.1-2 s"] = float(np.nanmax(np.abs(err[m, 0])))
    row["peak th_dot err 0.1-2 s"] = float(np.nanmax(np.abs(err[m, 1])))
    rows = [row]

    if not warm and diverged is None:
        twin = make_twin(obs)
        for test in held_out_tests(data.params_final):
            r = {**base, "test": test.name}
            try:
                r.update(twin_metrics(twin, test))
            except (FloatingPointError, np.linalg.LinAlgError):
                r.update({k: np.nan for k in TWIN_METRICS})
            rows.append(r)
    rows[0]["seconds"] = time.time() - t0
    return rows


# ---------------------------------------------------------------------- plotting
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Reference categorical palette, validated order (scripts/validate_palette.js): slots 1-6.
COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100", "A-x": "#e87ba4", "C-x": "#008300", "E": "#4a3aa7"}
ORDER = ("A", "B", "C", "D", "A-x", "C-x", "E")


def _style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "legend.frameon": False, "legend.fontsize": 8, "font.family": "DejaVu Sans",
    })


def bar_panels(table: pd.DataFrame, metrics: List[Tuple[str, str]], out: Path, title: str, log: bool = True) -> None:
    """table: index (scenario, model) -> columns metric (median), plus <metric>__lo / __hi."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6), squeeze=False)
    x = np.arange(len(SCENARIOS))
    width = 0.8 / len(ORDER)
    for ax, (metric, label) in zip(axes[0], metrics):
        for i, m in enumerate(ORDER):
            med = np.array([table.loc[(s, m), metric] if (s, m) in table.index else np.nan for s in SCENARIOS])
            lo = np.array([table.loc[(s, m), metric + "__lo"] if (s, m) in table.index else np.nan for s in SCENARIOS])
            hi = np.array([table.loc[(s, m), metric + "__hi"] if (s, m) in table.index else np.nan for s in SCENARIOS])
            pos = x - 0.4 + (i + 0.5) * width
            ax.bar(pos, med, width * 0.92, color=COLORS[m], label=m, edgecolor="white", linewidth=1)
            ax.errorbar(pos, med, yerr=[med - lo, hi - med], fmt="none", ecolor=INK_2, elinewidth=0.7, capsize=1.5)
        if log:
            ax.set_yscale("log")
        ax.set_xticks(x, list(SCENARIOS))
        ax.set_title(label)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ORDER), bbox_to_anchor=(0.5, 0.94))
    fig.suptitle(title, fontsize=10, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, "Bars: median over seeds (and over held-out tests where applicable); whiskers: min-max. "
             "S1 = A's scenario, S2 = B's, S3 = C's.", fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    fig.savefig(out)
    plt.close(fig)


def warm_panels(df: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    metrics = [("RMSE x_dot transient", r"$\dot x$ RMSE, 0.1-5 s [m/s]"), ("RMSE th_dot transient", r"$\dot\theta$ RMSE, 0.1-5 s [rad/s]")]
    fig, axes = plt.subplots(len(metrics), len(SCENARIOS), figsize=(12, 5.6), squeeze=False)
    for r, (metric, label) in enumerate(metrics):
        for c, s in enumerate(SCENARIOS):
            ax = axes[r][c]
            sub = df[df.scenario == s]
            x = np.arange(len(ORDER))
            for j, (warm, alpha) in enumerate(((False, 0.45), (True, 1.0))):
                med = [sub[(sub.model == m) & (sub["warm start"] == warm)][metric].median() for m in ORDER]
                ax.bar(x + (j - 0.5) * 0.38, med, 0.36, color=[COLORS[m] for m in ORDER], alpha=alpha,
                       edgecolor="white", linewidth=1, label="warm start" if warm else "cold start")
            ax.set_yscale("log")
            ax.set_xticks(x, list(ORDER))
            ax.grid(axis="x", visible=False)
            if r == 0:
                ax.set_title(s)
            if c == 0:
                ax.set_ylabel(label)
    handles = [plt.Rectangle((0, 0), 1, 1, color=INK_2, alpha=0.45), plt.Rectangle((0, 0), 1, 1, color=INK_2)]
    fig.legend(handles, ["cold start (left, light)", "warm start (right)"], loc="upper right", ncol=2, frameon=False)
    fig.suptitle("Improvement 3: model-free warm start (median over seeds)", fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def _med_lo_hi(df: pd.DataFrame, keys: List[str], cols: List[str]) -> pd.DataFrame:
    g = df.groupby(keys)[cols]
    med, lo, hi = g.median(), g.min(), g.max()
    return pd.concat([med, lo.add_suffix("__lo"), hi.add_suffix("__hi")], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plots-only", action="store_true", help="redraw the figures from the saved CSVs")
    parser.add_argument("--only", nargs="+", metavar="MODEL", choices=MODELS,
                        help="run only these models (cold and warm) and merge them into the saved CSVs")
    parser.add_argument("--warm-only", action="store_true",
                        help="rerun only the warm-start runs and merge them into the saved CSVs (after a warm-start change)")
    args = parser.parse_args()
    res_dir = ROOT_DIR / "results" / "shared_benchmark"
    fig_dir = ROOT_DIR / "figures" / "shared_benchmark"
    if args.plots_only:
        runs = pd.read_csv(res_dir / "velocity_runs.csv")
        twins = pd.read_csv(res_dir / "held_out_twins.csv")
        make_figures(runs, twins, fig_dir)
        return
    if args.warm_only:
        rerun_warm(args.seeds, args.workers, res_dir, fig_dir, plots=not args.no_plots)
        return
    if args.only:
        rerun_models(tuple(args.only), args.seeds, args.workers, res_dir, fig_dir, plots=not args.no_plots)
        return

    jobs = [(s, m, seed, w) for s in SCENARIOS for m in MODELS for seed in range(args.seeds) for w in (False, True)]
    jobs.sort(key=lambda j: (j[1] not in ("B", "D", "E"), j))  # slow models first
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        results = [r for rows in ex.map(job, jobs) for r in rows]
    print(f"{len(jobs)} runs in {time.time() - t0:.0f} s")

    res_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame([r for r in results if "test" not in r])
    twins = pd.DataFrame([r for r in results if "test" in r])
    runs.to_csv(res_dir / "velocity_runs.csv", index=False)
    twins.to_csv(res_dir / "held_out_twins.csv", index=False)

    cold = runs[~runs["warm start"]]
    vcols = ["RMSE x_dot steady", "RMSE th_dot steady", "RMSE th_dot transient", "peak th_dot err 0.1-2 s", "RMSE th_dot post-shift", "diverged"]
    vtab = _med_lo_hi(cold, ["scenario", "model"], vcols)
    ttab = _med_lo_hi(twins, ["scenario", "model"], list(TWIN_METRICS))
    vtab.to_csv(res_dir / "velocity_summary.csv")
    ttab.to_csv(res_dir / "held_out_summary.csv")

    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format, "display.max_rows", 100):
        print("\n=== 1. Velocity reconstruction, cold start (median over seeds) ===")
        print(cold.groupby(["scenario", "model"], sort=False)[vcols].median().to_string())
        print("\n=== 1. Held-out prediction (median over 4 tests x seeds) ===")
        print(twins.groupby(["scenario", "model"], sort=False)[list(TWIN_METRICS)].median().to_string())
        print("\n=== 1. Held-out prediction per test (median over seeds), theta_ddot one-step NMSE ===")
        print(twins.pivot_table(index=["scenario", "model"], columns="test", values="one-step NMSE theta_ddot", aggfunc="median").to_string())

        print("\n=== 2. Cart position removed (x-free minus original; negative = improvement) ===")
        for base, var in (("A", "A-x"), ("C", "C-x")):
            for s in SCENARIOS:
                vb = cold[(cold.model == base) & (cold.scenario == s)].set_index("seed")
                vv = cold[(cold.model == var) & (cold.scenario == s)].set_index("seed")
                tb = twins[(twins.model == base) & (twins.scenario == s)].groupby("seed")[list(TWIN_METRICS)].median()
                tv = twins[(twins.model == var) & (twins.scenario == s)].groupby("seed")[list(TWIN_METRICS)].median()
                d_vel = (vv["RMSE th_dot steady"] - vb["RMSE th_dot steady"]).median()
                d_one = (tv["one-step NMSE theta_ddot"] - tb["one-step NMSE theta_ddot"]).median()
                d_sh = (tv["0.5 s NMSE theta"] - tb["0.5 s NMSE theta"]).median()
                better = int(((tv["one-step NMSE theta_ddot"] < tb["one-step NMSE theta_ddot"])).sum())
                print(f"  {var} vs {base}, {s}: d steady th_dot RMSE {d_vel:+.4f}; d one-step th_ddot NMSE {d_one:+.3f} "
                      f"(better in {better}/{len(tb)} seeds); d 0.5 s theta NMSE {d_sh:+.3f}")

        print("\n=== 3. Warm start (median over seeds): cold -> warm ===")
        wcols = ["RMSE x_dot transient", "RMSE th_dot transient", "peak x_dot err 0.1-2 s", "peak th_dot err 0.1-2 s", "RMSE th_dot steady"]
        wt = runs.groupby(["scenario", "model", "warm start"])[wcols].median().unstack("warm start")
        for c in wcols:
            wt[(c, "ratio")] = wt[(c, True)] / wt[(c, False)]
        print(wt.to_string())
        wt.to_csv(res_dir / "warm_start_summary.csv")

    if not args.no_plots:
        make_figures(runs, twins, fig_dir)
    print(f"Results: {res_dir}")


WARM_COLS = ["RMSE x_dot transient", "RMSE th_dot transient", "peak x_dot err 0.1-2 s", "peak th_dot err 0.1-2 s", "RMSE th_dot steady"]


def warm_summary(runs: pd.DataFrame) -> pd.DataFrame:
    wt = runs.groupby(["scenario", "model", "warm start"])[WARM_COLS].median().unstack("warm start")
    for c in WARM_COLS:
        wt[(c, "ratio")] = wt[(c, True)] / wt[(c, False)]
    return wt


def rerun_warm(seeds: int, workers: int, res_dir: Path, fig_dir: Path, plots: bool) -> None:
    jobs = [(s, m, seed, True) for s in SCENARIOS for m in MODELS for seed in range(seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for out in ex.map(job, jobs) for r in out]
    print(f"{len(jobs)} warm-start runs in {time.time() - t0:.0f} s")
    runs = pd.read_csv(res_dir / "velocity_runs.csv")
    runs["warm start"] = runs["warm start"].astype(bool)
    runs = pd.concat([runs[~runs["warm start"]], pd.DataFrame(rows)], ignore_index=True)
    runs.to_csv(res_dir / "velocity_runs.csv", index=False)
    wt = warm_summary(runs)
    wt.to_csv(res_dir / "warm_start_summary.csv")
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        print(wt[[(c, "ratio") for c in WARM_COLS]].to_string())
    if plots:
        make_figures(runs, pd.read_csv(res_dir / "held_out_twins.csv"), fig_dir)


VEL_COLS = ["RMSE x_dot steady", "RMSE th_dot steady", "RMSE th_dot transient", "peak th_dot err 0.1-2 s", "RMSE th_dot post-shift", "diverged"]


def rerun_models(models: Tuple[str, ...], seeds: int, workers: int, res_dir: Path, fig_dir: Path, plots: bool) -> None:
    """Run only `models` (cold and warm start) and replace their rows in the saved CSVs."""
    jobs = [(s, m, seed, w) for s in SCENARIOS for m in models for seed in range(seeds) for w in (False, True)]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        results = [r for rows in ex.map(job, jobs) for r in rows]
    print(f"{len(jobs)} runs in {time.time() - t0:.0f} s")
    runs = pd.read_csv(res_dir / "velocity_runs.csv")
    twins = pd.read_csv(res_dir / "held_out_twins.csv")
    runs["warm start"] = runs["warm start"].astype(bool)
    runs = pd.concat([runs[~runs.model.isin(models)], pd.DataFrame([r for r in results if "test" not in r])], ignore_index=True)
    twins = pd.concat([twins[~twins.model.isin(models)], pd.DataFrame([r for r in results if "test" in r])], ignore_index=True)
    runs.to_csv(res_dir / "velocity_runs.csv", index=False)
    twins.to_csv(res_dir / "held_out_twins.csv", index=False)
    cold = runs[~runs["warm start"]]
    _med_lo_hi(cold, ["scenario", "model"], VEL_COLS).to_csv(res_dir / "velocity_summary.csv")
    _med_lo_hi(twins, ["scenario", "model"], list(TWIN_METRICS)).to_csv(res_dir / "held_out_summary.csv")
    wt = warm_summary(runs)
    wt.to_csv(res_dir / "warm_start_summary.csv")
    with pd.option_context("display.width", 250, "display.float_format", "{:.4g}".format, "display.max_rows", 100):
        print(cold.groupby(["scenario", "model"])[VEL_COLS].median().to_string())
        print(twins.groupby(["scenario", "model"])[list(TWIN_METRICS)].median().to_string())
        print(wt[[(c, "ratio") for c in WARM_COLS]].to_string())
    if plots:
        make_figures(runs, twins, fig_dir)


def make_figures(runs: pd.DataFrame, twins: pd.DataFrame, fig_dir: Path) -> None:
    cold = runs[~runs["warm start"].astype(bool)]
    vcols = ["RMSE x_dot steady", "RMSE th_dot steady", "RMSE th_dot transient", "peak th_dot err 0.1-2 s"]
    vtab = _med_lo_hi(cold, ["scenario", "model"], vcols)
    ttab = _med_lo_hi(twins, ["scenario", "model"], list(TWIN_METRICS))
    _style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    bar_panels(vtab, [("RMSE th_dot steady", r"steady $\dot\theta$ RMSE [rad/s]"),
                      ("RMSE x_dot steady", r"steady $\dot x$ RMSE [m/s]"),
                      ("peak th_dot err 0.1-2 s", r"peak $\dot\theta$ error, 0.1-2 s [rad/s]")],
               fig_dir / "velocity.png", "Velocity reconstruction on every home scenario (cold start)")
    bar_panels(ttab, [("one-step NMSE theta_ddot", r"one-step NMSE $\ddot\theta$"),
                      ("one-step NMSE x_ddot", r"one-step NMSE $\ddot x$"),
                      ("0.5 s NMSE theta", r"0.5 s horizon NMSE $\theta$")],
               fig_dir / "held_out.png", "Frozen models on the held-out test inputs")
    warm_panels(runs.assign(**{"warm start": runs["warm start"].astype(bool)}), fig_dir / "warm_start.png")
    print(f"\nFigures: {fig_dir}")


if __name__ == "__main__":
    main()
