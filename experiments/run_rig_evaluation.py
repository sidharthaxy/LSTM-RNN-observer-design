#!/usr/bin/env python3
"""
Evaluation of the observers on recorded data from the Feedback 33-936S rig.

The observers only need the encoder readings and the commanded input, and they are causal, so a
recorded run replayed through them gives the estimates they would have produced live.

Folder layout and file format: see src/rig/rig_data.py. Run types (keyword in the file name):
    rest           zero input, pendulum at rest: noise level and resolution
    deadzone       slow command ramp: where the cart starts to move
    freeswing, twotone, fading, mass_step    identification runs (each model learns on each run)
    heldout        held-out runs (each frozen model predicts them; never used for learning)

What is reported (the rig has no velocity sensor, so there is no true velocity):
    1. Velocity estimate against a zero-phase reference built from the recording itself.
    2. Held-out prediction: each model, frozen after an identification run, predicts the measured
       positions of the test runs 0.5 s ahead. This needs no velocity truth.
    3. Physical parameters identified by Approaches D and E, against the datasheet values.

Usage:
    python experiments/run_rig_evaluation.py --template > data/rig/metadata.txt
    python experiments/run_rig_evaluation.py --data data/rig [--models B D E] [--seeds 1]
    python experiments/run_rig_evaluation.py --make-demo data/rig_demo    # simulated runs in the rig format
Outputs: results/rig/<folder name>/*.csv, figures/rig/<folder name>/summary.png
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.benchmark.models import MODELS, make_observer, make_twin, twin_metrics  # noqa: E402
from src.benchmark.scenarios import DT, HELD_OUT, drift_cancelling_state, simulate  # noqa: E402
from src.observers.el_linear_model import LinearELModel  # noqa: E402
from src.observers.pi_icl_observer import PIICLObserver  # noqa: E402
from src.plant.pendulum_plant import PendulumParameters  # noqa: E402
from src.rig.rig_data import (  # noqa: E402
    METADATA_TEMPLATE, RigRun, as_dataset, is_mass_step, load_folder, metadata_text, reference_derivatives, write_run,
)

PHYS = ("M+m", "ml", "I+ml^2", "mgl")
TWIN_METRICS = ("one-step NMSE x_ddot", "one-step NMSE theta_ddot", "0.5 s NMSE x", "0.5 s NMSE theta")
T_SKIP = 0.1


# ---------------------------------------------------------------------- checks on the recording
def check_signs(run: RigRun) -> Dict[str, float]:
    """
    Sign conventions, from the physics. Near the hanging position the model gives
        x_ddot ~ g_u * u  with g_u > 0,   theta_ddot ~ a * x_ddot - w^2 (theta - pi)  with a > 0.
    A negative g_u means cart_sign is flipped; a negative a means angle_sign is flipped.
    """
    pos, _, acc = reference_derivatives(run.meas)
    out = {"g_u": float("nan"), "a": float("nan")}
    if np.std(run.u) > 1e-3:
        out["g_u"] = float(np.dot(run.u, acc[:, 0]) / np.dot(run.u, run.u))
    if np.std(acc[:, 0]) > 1e-3 and np.abs(pos[:, 1] - np.pi).max() < 1.2:
        X = np.column_stack((acc[:, 0], -(pos[:, 1] - np.pi)))
        out["a"] = float(np.linalg.lstsq(X, acc[:, 1], rcond=None)[0][0])
    return out


def rest_statistics(run: RigRun) -> Dict[str, float]:
    d = run.meas - run.meas.mean(axis=0)
    steps = [np.unique(np.abs(np.diff(run.meas[:, j]))) for j in range(2)]
    res = [float(s[s > 1e-9].min()) if np.any(s > 1e-9) else 0.0 for s in steps]   # ignore float round-off
    _, vel, _ = reference_derivatives(run.meas)
    core = vel[200:-200]                               # the pendulum is at rest: all of this is reference noise
    return {"x std [m]": float(d[:, 0].std()), "theta std [rad]": float(d[:, 1].std()),
            "x resolution [m]": res[0], "theta resolution [rad]": res[1],
            "reference x_dot noise floor [m/s]": float(core[:, 0].std()), "reference th_dot noise floor [rad/s]": float(core[:, 1].std())}


def deadzone_thresholds(run: RigRun, v_min: float = 0.01) -> Dict[str, float]:
    """Smallest |command| at which the cart moves faster than v_min, for each direction [N]."""
    _, vel, _ = reference_derivatives(run.meas)
    moving = np.abs(vel[:, 0]) > v_min
    out = {}
    for name, sel in (("positive", run.u > 0), ("negative", run.u < 0)):
        m = moving & sel
        out[f"breakaway {name} [N]"] = float(np.abs(run.u[m]).min()) if m.any() else float("nan")
    return out


# ---------------------------------------------------------------------- replay
def job(args: Tuple[str, str, str, int]) -> List[Dict[str, Any]]:
    folder, run_name, model, seed = args
    _, runs = load_folder(Path(folder))
    run = next(r for r in runs if r.name == run_name)
    data = as_dataset(run)
    obs = make_observer(model, data, seed)
    obs.reset(np.array([data.meas[0, 0], 0.0, data.meas[0, 1], 0.0]))
    N = len(data.t)
    vel = np.full((N, 2), np.nan)
    diverged = None
    for k in range(N):
        est = obs.update(data.meas[k], data.u[k], DT)
        if not np.all(np.isfinite(est)) or np.abs(est).max() > 1e3:
            diverged = float(data.t[k])
            break
        vel[k] = est[[1, 3]]
    base = {"run": run_name, "model": model, "seed": seed}
    row: Dict[str, Any] = {**base, "kind": "velocity", "diverged": float(diverged is not None)}
    err = vel - data.velocities                      # against the zero-phase reference
    t = data.t
    half = (t >= 0.5 * t[-1])
    early = (t >= T_SKIP) & (t < min(5.0, 0.5 * t[-1]))
    for name, m in (("second half", half), ("0.1-5 s", early)):
        rmse = np.sqrt(np.nanmean(err[m] ** 2, axis=0))
        row[f"x_dot RMSE vs ref, {name}"], row[f"th_dot RMSE vs ref, {name}"] = float(rmse[0]), float(rmse[1])
    rows = [row]
    if diverged is None:
        if isinstance(obs, PIICLObserver):
            truth = LinearELModel.physical_truth(PendulumParameters())
            pp = obs.physical_parameters()
            rows.append({**base, "kind": "parameters", **{n: pp[n] for n in PHYS},
                         **{f"{n} datasheet": truth[n] for n in PHYS}})
        if not is_mass_step(run_name):               # the held-out runs use the standard pendulum
            twin = make_twin(obs)
            for test in (r for r in runs if r.role == "test"):
                r = {**base, "kind": "prediction", "test": test.name}
                try:
                    r.update(twin_metrics(twin, as_dataset(test)))
                except (FloatingPointError, np.linalg.LinAlgError):
                    r.update({k: np.nan for k in TWIN_METRICS})
                rows.append(r)
    return rows


# ---------------------------------------------------------------------- demo data (simulation in the rig format)
def make_demo(folder: Path, repeats: int) -> None:
    """Simulated runs written exactly as the rig would log them (raw counts, volts). For rehearsing the pipeline."""
    meta = dict(METADATA_TEMPLATE, notes="Simulated demo data, not a rig recording.", mass_step_pendulum="pendulum mass and inertia +50 %")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metadata.txt").write_text(metadata_text(meta), encoding="utf-8")
    p = PendulumParameters()
    alt = replace(p, m=1.5 * p.m, I=1.5 * p.I)
    two_tone = lambda t: 2.0 * np.sin(1.5 * t) + 1.2 * np.cos(3.0 * t)
    fading = lambda t: 3.0 * np.exp(-t / 8.0) * np.sin(1.6 * t)
    ramp = lambda t: 0.05 * np.cos(2.0 * np.pi * t / 20.0)         # one slow cycle; small, the simulated cart has no stiction
    v_two = -2.0 / (1.5 * p.total_cart_mass)

    def save(name: str, u_fn, t_final: float, x0, seed: int, noise_deg: float = 0.5, params=None) -> None:
        d = simulate(name, u_fn, t_final, np.asarray(x0, dtype=float), noise_deg, 42 + seed, params=params)
        write_run(folder / f"{name}.csv", d.t, d.meas[:, 0], d.meas[:, 1], d.u, meta)

    save("run1_rest_trial1", lambda t: 0.0, 20.0, [0.0, 0.0, np.pi, 0.0], 0)
    save("run3_deadzone_trial1", ramp, 20.0, [-0.05 / ((2.0 * np.pi / 20.0) ** 2 * p.total_cart_mass), 0.0, np.pi, 0.0], 0)
    for k in range(1, repeats + 1):
        save(f"run2_freeswing_trial{k}", lambda t: 0.0, 20.0, [0.0, 0.0, np.pi - 0.4, 0.0], k)
        save(f"run4_twotone_trial{k}", two_tone, 50.0, [0.0, v_two, np.pi - 0.4, 0.0], k)
        x_c, v0 = drift_cancelling_state(fading, 60.0)
        save(f"run5_fading_trial{k}", fading, 60.0, [x_c, v0, np.pi - 0.3, 0.0], k)
    save("run7_mass_step_twotone", two_tone, 50.0, [0.0, -2.0 / (1.5 * alt.total_cart_mass), np.pi - 0.4, 0.0], 7, params=alt)
    for name, u_fn in zip(("square", "multisine", "chirp", "pulse"), HELD_OUT.values()):
        x_c, v0 = drift_cancelling_state(u_fn, 12.0)
        save(f"run6_heldout_{name}", u_fn, 12.0, [x_c, v0, np.pi, 0.0], 99)
    print(f"Demo recording written to {folder} ({len(list(folder.glob('*.csv')))} CSV files + metadata.txt)")


# ---------------------------------------------------------------------- plotting
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100", "A-x": "#e87ba4", "C-x": "#008300", "E": "#4a3aa7"}


def plot_summary(vel: pd.DataFrame, pred: pd.DataFrame, models: List[str], out: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "xtick.color": INK_2, "ytick.color": INK_2, "font.family": "DejaVu Sans",
    })
    panels = [(vel, "th_dot RMSE vs ref, second half", r"$\dot\theta$ estimate vs zero-phase reference [rad/s]")]
    if len(pred):
        panels.append((pred, "0.5 s NMSE theta", r"held-out runs: 0.5 s prediction NMSE of $\theta$"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.4), squeeze=False)
    for ax, (df, col, title) in zip(axes[0], panels):
        g = df.groupby("model")[col]
        med, lo, hi = g.median().reindex(models), g.min().reindex(models), g.max().reindex(models)
        x = np.arange(len(models))
        ax.bar(x, med, 0.6, color=[COLORS[m] for m in models], edgecolor="white", linewidth=1)
        ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="none", ecolor=INK_2, elinewidth=0.8, capsize=2)
        for xi, v in zip(x, med):
            ax.annotate(f"{v:.3g}", (xi, v), ha="center", va="bottom", fontsize=8, color=INK, xytext=(0, 2), textcoords="offset points")
        ax.set_yscale("log")
        ax.set_xticks(x, models)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    fig.text(0.01, 0.01, "Bars: median over runs and seeds; whiskers: min-max.", fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT_DIR / "data" / "rig")
    parser.add_argument("--models", nargs="+", default=["B", "D", "E"], choices=MODELS)
    parser.add_argument("--seeds", type=int, default=1, help="initialization seeds per (model, run)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--template", action="store_true", help="print a metadata.txt template and exit")
    parser.add_argument("--make-demo", type=Path, metavar="FOLDER", help="write simulated runs in the rig format and exit")
    parser.add_argument("--demo-repeats", type=int, default=1)
    args = parser.parse_args()
    if args.template:
        print(metadata_text(), end="")
        return
    if args.make_demo:
        make_demo(args.make_demo, args.demo_repeats)
        return

    folder = args.data.resolve()
    meta, runs = load_folder(folder)
    res_dir = ROOT_DIR / "results" / "rig" / folder.name
    res_dir.mkdir(parents=True, exist_ok=True)
    print(f"Recording: {folder}  ({len(runs)} runs; pendulum: {meta['pendulum']})\n")

    # 1. What is in the recording, and whether it is usable
    report, problems, signs = [], [], []
    for r in runs:
        driven = r.role in ("train", "test") and float(np.std(r.u)) > 0.05 and not r.near_stop
        s = check_signs(r) if driven else {"g_u": np.nan, "a": np.nan}     # only runs where the motor drives the cart
        if driven:
            signs.append(s)
        report.append({"run": r.name, "role": r.role, "duration [s]": round(float(r.t[-1]), 1), "max |x| [m]": round(float(np.abs(r.meas[:, 0]).max()), 3),
                       "near end stop": r.near_stop, "sample-time jitter [ms]": round(1e3 * r.max_dt_error, 3),
                       "cart gain g_u": s["g_u"], "coupling a": s["a"]})
        if r.near_stop:
            problems.append(f"{r.name}: the cart came within 2 cm of an end stop; impacts are not modelled, shorten or repeat the run.")
        if r.max_dt_error > 0.2 * DT:
            problems.append(f"{r.name}: sample time is irregular (jitter {1e3 * r.max_dt_error:.2f} ms).")
    if signs and np.nanmedian([s["g_u"] for s in signs]) < 0:
        problems.append("A positive command accelerates the cart backwards in the driven runs: flip cart_sign in metadata.txt.")
    if signs and np.nanmedian([s["a"] for s in signs]) < 0:
        problems.append("The pendulum reacts to cart acceleration with the wrong sign in the driven runs: flip angle_sign in metadata.txt.")
    rep = pd.DataFrame(report)
    rep.to_csv(res_dir / "run_report.csv", index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.3g}".format):
        print(rep.to_string(index=False))
        for r in runs:
            if r.role == "rest":
                print(f"\n{r.name} (noise at rest):", {k: float(f"{v:.3g}") for k, v in rest_statistics(r).items()})
            if r.role == "deadzone":
                print(f"{r.name}:", {k: float(f"{v:.3g}") for k, v in deadzone_thresholds(r).items()})
    if problems:
        print("\nProblems found:\n  " + "\n  ".join(problems))
    if any("flip" in p for p in problems):
        sys.exit("\nFix the sign convention in metadata.txt and run again.")

    # 2. Replay every model on every identification run
    train = [r for r in runs if r.role == "train"]
    if not train:
        sys.exit("No identification runs (freeswing, twotone, fading, mass_step) in the folder.")
    jobs = [(str(folder), r.name, m, seed) for r in train for m in args.models for seed in range(args.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for out in ex.map(job, jobs) for r in out]
    print(f"\n{len(jobs)} replays in {time.time() - t0:.0f} s")
    df = pd.DataFrame(rows)
    vel = df[df.kind == "velocity"].dropna(axis=1, how="all").drop(columns="kind")
    pred = df[df.kind == "prediction"].dropna(axis=1, how="all").drop(columns="kind")
    par = df[df.kind == "parameters"].dropna(axis=1, how="all").drop(columns="kind")
    vel.to_csv(res_dir / "velocity.csv", index=False)
    pred.to_csv(res_dir / "prediction.csv", index=False)
    par.to_csv(res_dir / "parameters.csv", index=False)

    with pd.option_context("display.width", 220, "display.float_format", "{:.4g}".format):
        vcols = [c for c in vel.columns if "RMSE" in c] + ["diverged"]
        print("\n=== Velocity estimate against the zero-phase reference (median over runs and seeds) ===")
        print(vel.groupby("model")[vcols].median().reindex(args.models).to_string())
        if len(pred):
            print("\n=== Held-out prediction of the test runs (median over identification runs, tests and seeds) ===")
            print(pred.groupby("model")[list(TWIN_METRICS)].median().reindex(args.models).to_string())
        else:
            print("\nNo heldout runs in the folder: held-out prediction skipped.")
        if len(par):
            print("\n=== Identified physical parameters (median over runs), with the datasheet value ===")
            std = par[~par.run.map(is_mass_step)]
            tab = std.groupby("model")[list(PHYS)].median()
            tab.loc["datasheet"] = [LinearELModel.physical_truth(PendulumParameters())[n] for n in PHYS]
            print(tab.to_string())
    if not args.no_plots:
        fig = ROOT_DIR / "figures" / "rig" / folder.name / "summary.png"
        plot_summary(vel, pred, list(args.models), fig)
        print(f"\nFigure: {fig}")
    print(f"Results: {res_dir}")


if __name__ == "__main__":
    main()
