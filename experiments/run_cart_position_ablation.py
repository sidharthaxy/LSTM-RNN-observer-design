#!/usr/bin/env python3
"""
Cart-position ablation for the black-box observers A (Lb-LSTM) and C (Lb-LSTM + concurrent learning).

Does the cart position x as a network input help or hurt?

The default A and C already take x: their LSTM input is
    zeta = [s * ([x_hat, theta_hat, x_dot_hat, theta_dot_hat, u] - mu), h_hat, 1]
(src/observers/blackbox_lstm.py, build_zeta). The x-free variants A-x and C-x
(src/benchmark/models.py, _lstm_scales) set the x input scale to 0 and change nothing else. So this
experiment compares the existing variants:

    A-x, C-x   no cart position in the network input
    A,   C     cart position in the network input (the defaults)

Every gain, learning rate, initialization rule, stack size (C: N = 48) and rank gate is that of
make_observer. Both variants have the same parameter count (the x column stays in the network and
receives a zero input in A-x / C-x).

Datasets:
    simulation S1, S2, S3 (src/benchmark/scenarios.py), scored against the true velocity over the
        steady window; the seed changes the noise realization and the LSTM initialization, as in the
        shared benchmark;
    rig runs run2_freeswing_trial2, run4_twotone_trial2, run5_fading_trial2, scored against the
        zero-phase Savitzky-Golay reference (cubic, 0.301 s) over the second half; the seed changes
        the LSTM initialization only.

Per run: theta_dot RMSE; the error split by a zero-phase 4th-order Butterworth low-pass at 10 Hz into
a part below 10 Hz (lag / gain / model error) and above (noise); gain (slope of the estimate on the
reference) and lag (cross-correlation peak within +-200 ms); the README's noise metric (RMS of the
10 Hz high-passed estimate from 0.1 s); the cart-velocity RMSE.

Cross-run prediction (rig): the model frozen after the free swing predicts the two-tone run and vice
versa (one-step theta_ddot NMSE and 0.5 s theta NMSE, src/benchmark/models.twin_metrics).

Invariance (A and C, which use x): the same run with the network's x input shifted by +0.1 m and
-0.1 m, and with it replaced by a time-shuffled copy of the measured x. Only the network input is
altered; the observer's own position tracking still uses the true measurement. Reported as the RMS
change of the theta_dot estimate.

Usage:  python experiments/run_cart_position_ablation.py [--seeds 3] [--workers 8]
        python experiments/run_cart_position_ablation.py --summary      # tables from the saved CSVs
Outputs: results/ablation_cart_position/{runs.csv, invariance.csv, summary.md}
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import importlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.benchmark.models import make_observer, make_twin, twin_metrics  # noqa: E402
from src.benchmark.scenarios import DT, STEADY_WINDOW, TRAINING  # noqa: E402
from src.rig.rig_data import as_dataset, load_folder  # noqa: E402

OUT_DIR = ROOT_DIR / "results" / "ablation_cart_position"
RIG_RUNS = {"free swing": "run2_freeswing_trial2", "two-tone": "run4_twotone_trial2", "fading": "run5_fading_trial2"}
CROSS = {"free swing": "two-tone", "two-tone": "free swing"}
DATASETS = ("S1", "S2", "S3", "free swing", "two-tone", "fading")
PERTURB = {"x + 0.1 m": 0.1, "x - 0.1 m": -0.1, "x shuffled": "shuffle"}
B_LO = butter(4, 10.0, btype="low", fs=1000.0)      # (b, a)
B_HI = butter(4, 10.0, btype="high", fs=1000.0)


def zero_phase(ba, s: np.ndarray) -> np.ndarray:
    return filtfilt(ba[0], ba[1], s)


def load_rig() -> Dict[str, Any]:
    _, runs = load_folder(ROOT_DIR / "data" / "rig")
    return {r.name: r for r in runs}


def dataset(name: str, seed: int):
    if name in TRAINING:
        d = TRAINING[name](seed)
        a, b = STEADY_WINDOW[name]
        return d, (d.t >= a) & (d.t < b)
    d = as_dataset(load_rig()[RIG_RUNS[name]])
    return d, d.t >= 0.5 * d.t[-1]


def perturb(obs, how, x_meas: np.ndarray, seed: int):
    """Alters only the network's x input: obs.build_zeta sees a shifted or shuffled cart position."""
    if how is None:
        return obs
    base = obs.__class__
    sub = (np.random.default_rng(1000 + seed).permutation(x_meas) if how == "shuffle" else None)

    class Perturbed(base):
        def build_zeta(self, x1_hat, x2_hat, u, h_hat):
            x1 = np.array(x1_hat, dtype=np.float64)
            x1[0] = sub[min(self._k, len(sub) - 1)] if sub is not None else x1[0] + how
            return super().build_zeta(x1, x2_hat, u, h_hat)

        def update(self, y, u, dt):
            est = super().update(y, u, dt)
            self._k += 1
            return est

    obs.__class__ = Perturbed
    obs._k = 0
    return obs


def run_observer(model: str, d, seed: int, how=None) -> Tuple[Any, np.ndarray]:
    obs = make_observer(model, d, seed)
    perturb(obs, how, d.meas[:, 0], seed)
    obs.reset(np.array([d.meas[0, 0], 0.0, d.meas[0, 1], 0.0]))
    N = len(d.t)
    vel = np.full((N, 2), np.nan)
    for k in range(N):
        est = obs.update(d.meas[k], d.u[k], DT)
        if not np.all(np.isfinite(est)) or np.abs(est).max() > 1e3:
            break
        vel[k] = est[[1, 3]]
    return obs, vel


def lag_ms(est: np.ndarray, ref: np.ndarray, maxlag: int = 200) -> float:
    e, r = est - est.mean(), ref - ref.mean()
    n = len(e)
    c = [np.dot(e[L:], r[:n - L]) if L >= 0 else np.dot(e[:n + L], r[-L:]) for L in range(-maxlag, maxlag + 1)]
    return float(np.argmax(c) - maxlag)


def score(d, vel: np.ndarray, m: np.ndarray) -> Dict[str, float]:
    ref = d.velocities
    if not np.all(np.isfinite(vel[m])):
        return {"diverged": 1.0}
    e = vel[:, 1] - ref[:, 1]
    lo = zero_phase(B_LO, e)
    whole = d.t >= 0.1
    return {
        "diverged": 0.0,
        "th RMSE [mrad/s]": 1e3 * float(np.sqrt(np.mean(e[m] ** 2))),
        "th <10Hz [mrad/s]": 1e3 * float(np.sqrt(np.mean(lo[m] ** 2))),
        "th >10Hz [mrad/s]": 1e3 * float(np.sqrt(np.mean((e[m] - lo[m]) ** 2))),
        "th gain": float(np.polyfit(ref[m, 1], vel[m, 1], 1)[0]),
        "th lag [ms]": lag_ms(vel[m, 1], ref[m, 1]),
        "th estimate >10Hz from 0.1 s [mrad/s]": 1e3 * float(np.std(zero_phase(B_HI, vel[whole, 1]))),
        "x_dot RMSE [mm/s]": 1e3 * float(np.sqrt(np.mean((vel[m, 0] - ref[m, 0]) ** 2))),
    }


def job(args: Tuple[str, str, int]) -> Dict[str, List[Dict[str, Any]]]:
    """One (dataset, model, seed). Models with x also run the three perturbations of their x input."""
    name, model, seed = args
    t0 = time.time()
    d, m = dataset(name, seed)
    base = {"dataset": name, "model": model, "seed": seed}
    obs, vel = run_observer(model, d, seed)
    row = {**base, **score(d, vel, m)}
    if name in CROSS and row["diverged"] == 0.0:
        test = as_dataset(load_rig()[RIG_RUNS[CROSS[name]]])
        try:
            tm = twin_metrics(make_twin(obs), test)
            row.update({f"cross {k}": v for k, v in tm.items()})
        except (FloatingPointError, np.linalg.LinAlgError):
            row["cross failed"] = 1.0
    inv = []
    if not model.endswith("-x"):
        for label, how in PERTURB.items():
            _, vp = run_observer(model, d, seed, how)
            ok = np.all(np.isfinite(vp[m]))
            diff = vp[m] - vel[m]
            sp = score(d, vp, m) if ok else {"diverged": 1.0}
            inv.append({**base, "perturbation": label, "diverged": float(not ok),
                        "th estimate change RMS [mrad/s]": 1e3 * float(np.sqrt(np.mean(diff[:, 1] ** 2))) if ok else np.nan,
                        "x_dot estimate change RMS [mm/s]": 1e3 * float(np.sqrt(np.mean(diff[:, 0] ** 2))) if ok else np.nan,
                        "th RMSE perturbed [mrad/s]": sp.get("th RMSE [mrad/s]", np.nan),
                        "th RMSE unperturbed [mrad/s]": row.get("th RMSE [mrad/s]", np.nan)})
    row["seconds"] = time.time() - t0
    return {"runs": [row], "invariance": inv}


# ---------------------------------------------------------------------- physics check
def position_force_check() -> str:
    """
    Is there a real position-dependent force on the rig cart (cable drag, a spring-like belt, a bowed
    track)? The cart model of README 12.5.3 (scripts/rig/characterise_rig.cart_fit_with_drive) is refitted
    with terms k1 x and k2 x^2 added, on the moving samples of the driven runs. A constant tilt is a
    constant force and already in the model (its offset term); it does not depend on x.
    """
    sys.path.insert(0, str(ROOT_DIR / "scripts" / "rig"))
    cr = importlib.import_module("characterise_rig")
    folder = ROOT_DIR / "data" / "rig"
    names = [f"run4_twotone_trial{i}" for i in (1, 2, 3)] + [f"run5_fading_trial{i}" for i in (1, 2, 3)]
    lines = ["## Physics check: a position-dependent cart force on the rig", "",
             "Cart model of README §12.5.3 (with the drive-compensation term) refitted on the moving samples",
             "(|ẋ| > 0.02 m/s) of the two-tone and fading runs, with k₁x and k₂x² added. In-sample R², and the",
             "R² of the model fitted on the two-tone runs and tested on the fading runs (and vice versa).", "",
             "| Terms | k₁ [m/s² per m] | k₂ [m/s² per m²] | R² in-sample | R² two-tone → fading | R² fading → two-tone |",
             "|:---|---:|---:|---:|---:|---:|"]

    def design(ns, extra):
        Xs, ys = [], []
        for n in ns:
            t, x, a, c = cr.load(folder, n)
            phr = -(a - np.pi)
            xd, xdd, pdd = cr.sg(x, 1), cr.sg(x, 2), cr.sg(phr, 2)
            m = np.abs(xd) > 0.02
            m[:300] = m[-300:] = False
            cols = [c, -xd, -(xd > 0).astype(float), (xd < 0).astype(float), pdd, np.ones_like(c), np.tanh(c / 0.01)]
            cols += [x ** (p + 1) for p in range(extra)]
            Xs.append(np.column_stack(cols)[m])
            ys.append(xdd[m])
        return np.vstack(Xs), np.concatenate(ys)

    def r2(X, y, th):
        return 1 - np.var(y - X @ th) / np.var(y)

    tt, ff = names[:3], names[3:]
    for extra, lab in ((0, "model of §12.5.3"), (1, "+ k₁x"), (2, "+ k₁x + k₂x²")):
        X, y = design(names, extra)
        th, *_ = np.linalg.lstsq(X, y, rcond=None)
        Xt, yt = design(tt, extra)
        Xf, yf = design(ff, extra)
        tht, *_ = np.linalg.lstsq(Xt, yt, rcond=None)
        thf, *_ = np.linalg.lstsq(Xf, yf, rcond=None)
        k1 = f"{th[7]:+.3f}" if extra >= 1 else "–"
        k2 = f"{th[8]:+.3f}" if extra >= 2 else "–"
        lines.append(f"| {lab} | {k1} | {k2} | {r2(X, y, th):.4f} | {r2(Xf, yf, tht):.4f} | {r2(Xt, yt, thf):.4f} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------- full rig evaluation
def rig_eval(seeds: int, workers: int) -> str:
    """
    The rig evaluation of README 12.5.4 for A, A-x, C and C-x on all ten identification recordings, with
    held-out prediction of the four held-out recordings. It calls experiments/run_rig_evaluation.job
    unchanged, so every replay is the one of the main evaluation, and summarizes per run type over seeds.
    """
    sys.path.insert(0, str(ROOT_DIR / "experiments"))
    rr = importlib.import_module("run_rig_evaluation")
    folder = ROOT_DIR / "data" / "rig"
    _, runs = load_folder(folder)
    train = [r.name for r in runs if r.role == "train"]
    jobs = [(str(folder), n, m, s) for m in ("A", "A-x", "C", "C-x") for n in train for s in range(seeds)]
    out_dir = OUT_DIR / "rig_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for res in ex.map(rr.job, jobs) for r in res]
    df = pd.DataFrame(rows)
    vel, pred = df[df.kind == "velocity"].drop(columns="kind"), df[df.kind == "prediction"].drop(columns="kind")
    vel.to_csv(out_dir / "velocity.csv", index=False)
    pred.to_csv(out_dir / "prediction.csv", index=False)
    return rig_eval_summary(vel, pred)


def _run_type(run: str) -> str:
    for k, lab in (("mass_step", "mass step"), ("freeswing", "free swing"), ("twotone", "two-tone"), ("fading", "fading")):
        if k in run:
            return lab
    return run


def rig_eval_summary(vel: pd.DataFrame, pred: pd.DataFrame) -> str:
    """Per run type: median over the runs of that type for each seed, then mean [min–max] over seeds."""
    types = ("free swing", "two-tone", "fading", "mass step")
    vel = vel.assign(type=vel.run.map(_run_type))
    pred = pred.assign(type=pred.run.map(_run_type))
    col = "th_dot RMSE vs ref, second half"
    out = ["## Full rig evaluation (all ten identification recordings, README §12.5.4 protocol)", "",
           "θ̇ RMSE against the reference, second half, in mrad/s. For each seed, the median over the runs of",
           "that type; then mean [min–max] over the seeds.", "",
           "| | " + " | ".join(types) + " | worst run |", "|:---|" + "---:|" * (len(types) + 1)]
    for mdl in ("A-x", "A", "C-x", "C"):
        v = vel[vel.model == mdl]
        cells = []
        for t in types:
            per_seed = v[v.type == t].groupby("seed")[col].median() * 1e3
            cells.append(ms(per_seed, "{:.1f}"))
        worst = v.loc[v[col].idxmax()]
        cells.append(f"{1e3 * worst[col]:.0f} ({worst.run}, seed {int(worst.seed)})")
        out.append(f"| {LABEL[mdl]} | " + " | ".join(cells) + " |")
    out += ["", "Held-out prediction of the four held-out recordings, 0.5 s θ NMSE (1 = no better than the mean):",
            "for each seed, the median over the training runs of that type and the four tests; then mean [min–max].", "",
            "| | trained on free swing | two-tone | fading | all |", "|:---|---:|---:|---:|---:|"]
    for mdl in ("A-x", "A", "C-x", "C"):
        p = pred[pred.model == mdl]
        cells = [ms(p[p.type == t].groupby("seed")["0.5 s NMSE theta"].median(), "{:.3g}") for t in ("free swing", "two-tone", "fading")]
        cells.append(ms(p.groupby("seed")["0.5 s NMSE theta"].median(), "{:.3g}"))
        out.append(f"| {LABEL[mdl]} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------- summary
MODELS = ("A-x", "A", "C-x", "C")
LABEL = {"A-x": "A without x (A-x)", "A": "A with x (default)", "C-x": "C without x (C-x)", "C": "C with x (default)"}


def ms(s: pd.Series, fmt: str) -> str:
    s = s.dropna()
    if s.empty:
        return "–"
    return f"{fmt.format(s.mean())} [{fmt.format(s.min())}–{fmt.format(s.max())}]"


def summarize(runs: pd.DataFrame, inv: pd.DataFrame) -> str:
    out = ["# Cart-position ablation: A and C with and without x in the network input", "",
           "Mean [min–max] over seeds. Simulation: against the true velocity, steady window. Rig: against",
           "the zero-phase Savitzky–Golay reference, second half; the reference is itself accurate to about",
           "2 mrad/s (4 on the fading run), so smaller differences are ties.", ""]
    cols = [("th RMSE [mrad/s]", "{:.1f}"), ("th gain", "{:.3f}"), ("th lag [ms]", "{:.0f}"),
            ("th <10Hz [mrad/s]", "{:.1f}"), ("th >10Hz [mrad/s]", "{:.2f}"),
            ("th estimate >10Hz from 0.1 s [mrad/s]", "{:.2f}"), ("x_dot RMSE [mm/s]", "{:.2f}")]
    for ds in DATASETS:
        sub = runs[runs.dataset == ds]
        if sub.empty:
            continue
        out += [f"## {ds}", "", "| | " + " | ".join(c for c, _ in cols) + " | diverged |",
                "|:---|" + "---:|" * (len(cols) + 1)]
        for mdl in MODELS:
            r = sub[sub.model == mdl]
            out.append(f"| {LABEL[mdl]} | " + " | ".join(ms(r[c], f) if c in r else "–" for c, f in cols)
                       + f" | {int(r['diverged'].sum())}/{len(r)} |")
        out.append("")
    cr = runs[runs.dataset.isin(list(CROSS))]
    if "cross 0.5 s NMSE theta" in cr:
        out += ["## Cross-run prediction on the rig (frozen model, NMSE; 1 = no better than the mean)", "",
                "| | free swing → two-tone, 0.5 s θ | one-step θ̈ | two-tone → free swing, 0.5 s θ | one-step θ̈ |",
                "|:---|---:|---:|---:|---:|"]
        for mdl in MODELS:
            cells = []
            for ds in ("free swing", "two-tone"):
                r = cr[(cr.model == mdl) & (cr.dataset == ds)]
                cells += [ms(r["cross 0.5 s NMSE theta"], "{:.3g}"), ms(r["cross one-step NMSE theta_ddot"], "{:.3g}")]
            out.append(f"| {LABEL[mdl]} | " + " | ".join(cells) + " |")
        out.append("")
    if len(inv):
        out += ["## Invariance: change of the θ̇ estimate when only the network's x input is altered [mrad/s]", "",
                "| | " + " | ".join(DATASETS) + " |", "|:---|" + "---:|" * len(DATASETS)]
        for mdl in ("A", "C"):
            for p in PERTURB:
                cells = []
                for ds in DATASETS:
                    r = inv[(inv.model == mdl) & (inv.perturbation == p) & (inv.dataset == ds)]
                    dv = int(r["diverged"].sum())
                    cells.append(ms(r["th estimate change RMS [mrad/s]"], "{:.2g}") + (f" ({dv} diverged)" if dv else ""))
                out.append(f"| {mdl}, {p} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--summary", action="store_true", help="only rebuild summary.md from the saved CSVs")
    ap.add_argument("--rig-eval", action="store_true",
                    help="the full rig evaluation (README 12.5.4) for A, A-x, C, C-x on all recordings")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.rig_eval:
        rd = OUT_DIR / "rig_eval"
        if args.summary:
            text = rig_eval_summary(pd.read_csv(rd / "velocity.csv"), pd.read_csv(rd / "prediction.csv"))
        else:
            text = rig_eval(args.seeds, args.workers)
        (rd / "summary.md").write_text(text + "\n", encoding="utf-8")
        print(text)
        return
    if not args.summary:
        jobs = [(ds, m, s) for m in ("C", "A", "C-x", "A-x") for ds in DATASETS for s in range(args.seeds)]
        runs, inv = [], []
        t0 = time.time()
        with ProcessPoolExecutor(args.workers) as ex:
            for res in ex.map(job, jobs):
                runs += res["runs"]
                inv += res["invariance"]
                r = res["runs"][0]
                print(f"{r['dataset']:10s} {r['model']:4s} seed {r['seed']}  {r['seconds']:.0f} s  ({time.time() - t0:.0f} s)", flush=True)
        pd.DataFrame(runs).to_csv(OUT_DIR / "runs.csv", index=False)
        pd.DataFrame(inv).to_csv(OUT_DIR / "invariance.csv", index=False)
    text = summarize(pd.read_csv(OUT_DIR / "runs.csv"), pd.read_csv(OUT_DIR / "invariance.csv")) + "\n" + position_force_check()
    (OUT_DIR / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
