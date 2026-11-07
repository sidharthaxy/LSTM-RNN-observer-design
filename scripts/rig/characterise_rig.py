#!/usr/bin/env python3
"""
Characterisation of the real-rig recordings in data/rig/ (session of 6 Nov 2026), README §12.5.

Checks, in the order of docs/rig_session_2026-11-06.md §11:
    A  integrity: header, row count, 1 ms steps, initial values, encoder quantisation
    B  excitation reconstructed from the logged total command and the centring loop, against the
       nominal input of each run (catches column, sign, amplitude and timing errors)
    C  sign of the angle: gain from cart acceleration to pendulum angle (rigid pendulum: 1/g)
    D  actuator and friction: break-away from rest, a cart model fitted while moving, and that model
       simulated in closed loop against the measured cart travel
    E  pendulum natural frequency and damping from the free swings, and the size of the mass step

Usage:  python scripts/rig/characterise_rig.py [--data data/rig] [--no-plots]
Writes  results/rig/characterisation.csv and figures/rig/characterisation.png

Automatic calibration (README §12.5.5): one free swing and one drive run give the observers'
starting values and limits, written as a block of cal_ fields into metadata.txt:
        python scripts/rig/characterise_rig.py --calibrate run2_freeswing_trial2 run4_twotone_trial2
        [--metadata data/rig/metadata.txt] [--dry-run]
The drive run may be the staircase (run3_deadzone_*), which also gives the break-away voltages, but
its short pulses misreport the acceleration per volt (§12.5.5); a two-tone run is the better choice.
newtons_per_volt is read from the metadata: acceleration per volt fixes only the ratio of amplifier
gain to cart mass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.benchmark.scenarios import h1_filtered_square, h2_multisine, h3_slow_chirp, h4_pulse_train  # noqa: E402

DT = 1e-3
G = 9.81
X_COUNT = 0.078 / 1024            # cart encoder step [m]
A_COUNT = 2.0 * np.pi / 2048      # pendulum encoder step [rad]
HEADER = "time_s,cart_pos,pend_angle,command_v"

# Amplitudes actually used on 6 Nov 2026 (docs/rig_session_2026-11-06.md §5)
A_TWOTONE = {"run4_twotone_trial1": 0.25}
A_TWOTONE_DEFAULT, A_FADING, A_HELDOUT = 0.35, 0.40, 0.70
RUNS = ["run1_rest_trial1", "run3_deadzone_trial1", "run2_freeswing_trial1", "run2_freeswing_trial2",
        "run2_freeswing_trial3", "run4_twotone_trial1", "run4_twotone_trial2", "run4_twotone_trial3",
        "run5_fading_trial1", "run5_fading_trial2", "run5_fading_trial3", "run6_heldout_square",
        "run6_heldout_multisine", "run6_heldout_chirp", "run6_heldout_pulse", "run7_mass_step_twotone"]


def load(folder: Path, name: str) -> Tuple[np.ndarray, ...]:
    d = np.loadtxt(folder / f"{name}.csv", delimiter=",", skiprows=1)
    return tuple(d.T)


def nominal(name: str, t: np.ndarray) -> np.ndarray:
    """The excitation of scripts/rig/rig_inputs.m, with the amplitudes of the session."""
    if "rest" in name or "freeswing" in name:
        return np.zeros_like(t)
    if "deadzone" in name:
        k = np.floor(t / 3); ph = t - 3 * k; a = 0.10 + 0.05 * k
        u = a * ((ph < 0.5).astype(float) - ((ph >= 1.5) & (ph < 2.0)))
        u[a > 0.80 + 1e-9] = 0.0
        return u
    if "twotone" in name:
        return A_TWOTONE.get(name, A_TWOTONE_DEFAULT) * (2 * np.sin(1.5 * t) + 1.2 * np.cos(3 * t))
    if "fading" in name:
        return A_FADING * 3 * np.exp(-t / 8) * np.sin(1.6 * t)
    tt = np.arange(0.0, 12.0 + 1e-9, DT)
    shape = {"square": h1_filtered_square, "multisine": h2_multisine, "chirp": h3_slow_chirp, "pulse": h4_pulse_train}
    f = next(v for k, v in shape.items() if k in name)
    peak = np.abs([f(s) for s in tt]).max()
    return A_HELDOUT * np.array([f(s) for s in t]) / peak


def loop_output(x: np.ndarray) -> np.ndarray:
    """Simulink ode1 realisation of the centring loop G(s) = (2.03 s + 1.5)/(0.02 s + 1) on e = -x."""
    y = np.empty_like(x); z = 0.0
    for k, xk in enumerate(x):
        y[k] = -101.5 * xk - 100.0 * z
        z = 0.95 * z - 0.05 * xk
    return y


def sg(s: np.ndarray, deriv: int, window: int = 101) -> np.ndarray:
    return savgol_filter(s, window, 3, deriv=deriv, delta=DT)


# ---------------------------------------------------------------------- A, B
def integrity(folder: Path) -> pd.DataFrame:
    rows = []
    for n in RUNS:
        header = (folder / f"{n}.csv").open().readline().strip()
        t, x, a, c = load(folder, n)
        res = c - loop_output(x) - nominal(n, t)
        rows.append({"run": n, "header ok": header == HEADER, "rows": len(t), "t0 [s]": t[0],
                     "steps = 1 ms": bool(np.all(np.abs(np.diff(t) - DT) < 1e-9)),
                     "x0 = 0": x[0] == 0.0, "angle0 = pi": abs(a[0] - np.pi) < 1e-4,
                     "x in counts (max dev)": float(np.abs(x / X_COUNT - np.round(x / X_COUNT)).max()),
                     "angle in counts (max dev)": float(np.abs((a - np.pi) / A_COUNT - np.round((a - np.pi) / A_COUNT)).max()),
                     "excitation residual [V]": float(np.abs(res).max()),
                     "x min [m]": x.min(), "x max [m]": x.max(), "angle min": a.min(), "angle max": a.max()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- C
def angle_gain(folder: Path, name: str, window: int = 301) -> Tuple[float, float]:
    """Fit phi'' + 2 z w phi' + w^2 phi = b x'' (phi = logged angle - pi); returns (DC gain b/w^2, w)."""
    t, x, a, c = load(folder, name)
    phi = a - np.pi
    m = slice(1000, -1000)
    p, pd_, pdd = (sg(phi, k, window)[m] for k in (0, 1, 2))
    xdd = sg(x, 2, window)[m]
    (c1, c0, b), *_ = np.linalg.lstsq(np.column_stack([-pd_, -p, xdd]), pdd, rcond=None)
    return float(b / c0), float(np.sqrt(c0))


# ---------------------------------------------------------------------- E
def free_swing(folder: Path, name: str) -> Dict[str, float]:
    t, x, a, c = load(folder, name)
    ps = savgol_filter(a - np.pi, 51, 3)
    absps = np.abs(ps)
    pk = np.where((absps[1:-1] >= absps[:-2]) & (absps[1:-1] > absps[2:]) & (absps[1:-1] > 0.01))[0] + 1
    i0 = pk[np.argmax(absps[pk] > 0.9 * absps.max())]             # first extremum after release
    s = np.sign(ps[i0:]); idx = np.where((s[:-1] < 0) & (s[1:] >= 0))[0] + i0
    tc = np.array([t[i] - ps[i] * (t[i + 1] - t[i]) / (ps[i + 1] - ps[i]) for i in idx])
    T = np.diff(tc)
    peaks = pk[(pk >= i0) & (ps[pk] > 0)]
    amp0 = float(absps[i0])
    wd_early = 2 * np.pi / T[:3].mean()
    slope = np.polyfit(t[peaks], np.log(ps[peaks]), 1)[0]
    return {"release amplitude [deg]": np.degrees(amp0), "periods": len(T),
            "w, first 3 periods [rad/s]": wd_early,
            "w, small amplitude [rad/s]": wd_early / (1 - amp0 ** 2 / 16),      # large-angle period correction
            "zeta (exponential fit)": -slope / (2 * np.pi / T.mean()),
            "final amplitude [deg]": np.degrees(ps[peaks[-1]])}


# ---------------------------------------------------------------------- D
def breakaway(folder: Path, name: str = "run3_deadzone_trial1", still: int = 100, moved: int = 3) -> pd.DataFrame:
    """Staircase pulses that start with the cart at rest: did the cart move (>= `moved` counts) during the pulse?

    Each pulse is a step, so the command jumps straight to the pulse level: the break-away voltage lies
    between the largest level that left the cart still and the smallest level that moved it.
    """
    t, x, a, c = load(folder, name)
    cnt = np.round(x / X_COUNT).astype(int)
    rows = []
    for k in range(15):
        level = 0.10 + 0.05 * k
        for d, t0 in ((+1, 3.0 * k), (-1, 3.0 * k + 1.5)):
            i0 = int(round(t0 / DT))
            if np.any(cnt[max(0, i0 - still):i0 + 1] != cnt[i0]):
                continue                                      # the cart was already moving
            travel = d * (cnt[i0:i0 + 500] - cnt[i0])
            rows.append({"level [V]": round(level, 2), "direction": "+x" if d > 0 else "-x",
                         "command at start [V]": c[i0 + 1], "moved": bool(travel.max() >= moved),
                         "travel in pulse [counts]": int(travel.max())})
    return pd.DataFrame(rows)


def bracket(br: pd.DataFrame, d: str) -> Tuple[float, float]:
    b = br[br.direction == d]
    still = b[~b.moved]["level [V]"]; moving = b[b.moved]["level [V]"]
    return (float(still.max()) if len(still) else 0.0), (float(moving.min()) if len(moving) else float("nan"))


CART_TERMS = ["alpha [m/s2 per V]", "viscous [1/s]", "Coulomb +x [m/s2]", "Coulomb -x [m/s2]", "coupling [m]", "offset [m/s2]"]


def cart_fit(folder: Path, names: List[str], vmin: float = 0.02) -> Tuple[np.ndarray, float]:
    """x'' = alpha u - b x' - C+ [x'>0] + C- [x'<0] + delta phi_r'' + offset, on samples where the cart moves."""
    Xs, ys = [], []
    for n in names:
        t, x, a, c = load(folder, n)
        phr = -(a - np.pi)                              # repo convention (angle_sign = -1)
        xd, xdd, pdd = sg(x, 1), sg(x, 2), sg(phr, 2)
        m = np.abs(xd) > vmin; m[:300] = m[-300:] = False
        Xs.append(np.column_stack([c, -xd, -(xd > 0).astype(float), (xd < 0).astype(float), pdd, np.ones_like(c)])[m])
        ys.append(xdd[m])
    X, y = np.vstack(Xs), np.concatenate(ys)
    th, *_ = np.linalg.lstsq(X, y, rcond=None)
    return th, float(1 - np.var(y - X @ th) / np.var(y))


# ---------------------------------------------------------------------- automatic calibration
MARGIN_FRACTION = 0.5     # eps_M = this fraction of the smallest eigenvalue of the calibrated M(theta)
CAL_BLOCK = "# --- automatic calibration (scripts/rig/characterise_rig.py --calibrate); edit by re-running it ---"


def cart_fit_with_drive(folder: Path, names: List[str], angle_sign: float, vmin: float = 0.02,
                        u_eps: float = 0.01) -> Tuple[Dict[str, float], float]:
    """cart_fit plus a term along tanh(u / u_eps) (the drive's own friction compensation), in logged volts."""
    Xs, ys = [], []
    for n in names:
        t, x, a, c = load(folder, n)
        phr = angle_sign * (a - np.pi)                  # repo convention
        xd, xdd, pdd = sg(x, 1), sg(x, 2), sg(phr, 2)
        m = np.abs(xd) > vmin; m[:300] = m[-300:] = False
        Xs.append(np.column_stack([c, -xd, -(xd > 0).astype(float), (xd < 0).astype(float), pdd, np.ones_like(c),
                                   np.tanh(c / u_eps)])[m])
        ys.append(xdd[m])
    X, y = np.vstack(Xs), np.concatenate(ys)
    th, *_ = np.linalg.lstsq(X, y, rcond=None)
    names_ = ["alpha", "viscous", "coulomb_pos", "coulomb_neg", "coupling", "offset", "drive"]
    return dict(zip(names_, map(float, th))), float(1 - np.var(y - X @ th) / np.var(y))


def calibrate(folder: Path, swing: str, drive: str, newtons_per_volt: float, angle_sign: float) -> Dict[str, Any]:
    """
    Starting values and limits of B, D and E from one free swing and one drive run (metadata cal_ fields).

    Pendulum: natural frequency (gravity scale = its square) and damping ratio from the free swing.
    Cart: x'' = alpha u - b x' - C sgn(x') + delta phi'' + c + kappa tanh(u / 0.01 V), fitted while
    the cart moves. With M + m = newtons_per_volt / alpha (only the ratio is identifiable), the model's
    terms in newtons: d_v = b (M+m), d_c = C (M+m) (mean of both directions), f_0 = -c (M+m),
    k_u = kappa (M+m); m l = delta (M+m); I + m l^2 = m l g / omega^2 (any rigid pendulum);
    eps_M = MARGIN_FRACTION x lambda_min of [[M+m, m l cos th], [m l cos th, I + m l^2]] over th.
    """
    fs = free_swing(folder, swing)
    wn, zeta = float(fs["w, small amplitude [rad/s]"]), max(float(fs["zeta (exponential fit)"]), 0.0)
    f, r2 = cart_fit_with_drive(folder, [drive], angle_sign)
    if f["alpha"] <= 0.0:
        raise ValueError(f"{drive}: the fitted acceleration per volt is {f['alpha']:.3g}; this run cannot calibrate the cart.")
    mc = newtons_per_volt / f["alpha"]
    ml = max(f["coupling"], 0.0) * mc
    J = ml * G / wn ** 2
    lam = min(float(np.linalg.eigvalsh(np.array([[mc, ml * s], [ml * s, J]]))[0]) for s in (-1.0, 1.0))
    out: Dict[str, Any] = {
        "cal_runs": f"{swing}, {drive}",
        "cal_natural_frequency_rad_s": wn, "cal_damping_ratio": zeta,
        "cal_accel_per_volt": f["alpha"], "cal_cart_inertia_kg": mc,
        "cal_pendulum_ml_kg_m": ml, "cal_pendulum_inertia_kg_m2": J,
        "cal_inertia_margin": MARGIN_FRACTION * lam if lam > 0.0 else 0.0,
        "cal_cart_viscous_n_s_per_m": max(f["viscous"], 0.0) * mc,
        "cal_pendulum_viscous_n_m_s": 2.0 * zeta * J * wn,
        "cal_cart_coulomb_n": max(0.5 * (f["coulomb_pos"] + f["coulomb_neg"]), 0.0) * mc,
        "cal_constant_force_n": -f["offset"] * mc,
        "cal_command_force_n": max(f["drive"], 0.0) * mc,
        "cal_breakaway_pos_v": 0.0, "cal_breakaway_neg_v": 0.0,
    }
    if "deadzone" in drive:
        br = breakaway(folder, drive)
        out["cal_breakaway_pos_v"] = bracket(br, "+x")[1]
        out["cal_breakaway_neg_v"] = bracket(br, "-x")[1]
    out["_fit_r2"] = r2
    return out


def write_calibration(path: Path, cal: Dict[str, Any]) -> None:
    """Replaces the cal_ block of metadata.txt (every other line is kept as it is)."""
    keep = [ln for ln in path.read_text(encoding="utf-8").splitlines()
            if not ln.strip().startswith("cal_") and ln.strip() != CAL_BLOCK]
    while keep and not keep[-1].strip():
        keep.pop()
    block = [CAL_BLOCK] + [f"{k} = {v}" if isinstance(v, str) else f"{k} = {v:.6g}"
                           for k, v in cal.items() if k.startswith("cal_")]
    path.write_text("\n".join(keep + [""] + block) + "\n", encoding="utf-8")


def closed_loop(u_exc: np.ndarray, th: np.ndarray, wn: float, zeta: float, K: float) -> np.ndarray:
    """Cart model `th` and pendulum (wn, zeta, DC gain K) in the rig's closed loop; returns x(t)."""
    al, be, cp, cm, dl, off = th
    x = xd = ph = phd = z = 0.0
    out = np.empty_like(u_exc)
    den = 1.0 - dl * wn * wn * K
    for k, ue in enumerate(u_exc):
        u = ue - 101.5 * x - 100.0 * z; z = 0.95 * z - 0.05 * x
        free = -wn * wn * ph - 2 * zeta * wn * phd
        drive = al * u + off + dl * free
        if abs(xd) < 1e-3:
            static = cp if drive > 0 else cm
            xdd = 0.0 if abs(drive) <= static else (drive - np.sign(drive) * static) / den
            if xdd == 0.0:
                xd = 0.0
        else:
            xdd = (drive - be * xd - (cp if xd > 0 else -cm)) / den
        phdd = free + wn * wn * K * xdd
        out[k] = x
        x += DT * xd; xd += DT * xdd; ph += DT * phd; phd += DT * phdd
    return out


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "rig")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--calibrate", nargs=2, metavar=("FREE_SWING", "DRIVE_RUN"),
                    help="only the automatic calibration: write the cal_ block of metadata.txt and exit")
    ap.add_argument("--metadata", type=Path, help="metadata file to write the calibration to (default: <data>/metadata.txt)")
    ap.add_argument("--dry-run", action="store_true", help="with --calibrate: print, do not write")
    args = ap.parse_args()
    folder = args.data.resolve()
    pd.set_option("display.width", 220)
    if args.calibrate:
        from src.rig.rig_data import load_metadata  # noqa: E402
        meta_path = (args.metadata or folder / "metadata.txt").resolve()
        meta = load_metadata(folder)                 # units and signs of the recordings, amplifier gain
        swing, drive = args.calibrate
        cal = calibrate(folder, swing, drive, float(meta["newtons_per_volt"]), float(meta["angle_sign"]))
        print(f"Automatic calibration from {cal['cal_runs']} (cart fit R2 {cal.pop('_fit_r2'):.3f}; "
              f"newtons_per_volt {meta['newtons_per_volt']} from the metadata):")
        for k, v in cal.items():
            print(f"   {k:30s} {v}")
        print("Hand-tuned values it replaces: gravity scale 30.0 (D, measured; E learns it from 5), eps_M 0.02 (E) and "
              "0.05 (B, D), cart prior 2.0 kg, friction and force terms starting at 0.")
        if not args.dry_run:
            write_calibration(meta_path, cal)
            print(f"Written to {meta_path}")
        return
    summary: Dict[str, float] = {}

    print("A, B. Integrity, quantisation, and the excitation reconstructed from the total command")
    tab = integrity(folder)
    print(tab.drop(columns=["x min [m]", "x max [m]", "angle min", "angle max"]).to_string(index=False, float_format="{:.3g}".format))
    summary["max excitation residual [V]"] = float(tab["excitation residual [V]"].max())

    print("\nC. Gain from cart acceleration to the logged angle (rigid pendulum: magnitude 1/g = %.4f)" % (1 / G))
    gains = {n: angle_gain(folder, n) for n in RUNS if "twotone" in n or "fading" in n}
    for n, (K, w) in gains.items():
        print(f"   {n:26s} K = {K:+.4f} rad per m/s2")
    K = float(np.median([k for k, _ in gains.values()]))
    summary["angle gain K [rad per m/s2]"] = K
    print(f"   -> the logged angle grows when the bob swings towards +x: angle_sign = {-1 if K < 0 else 1}")

    print("\nE. Free swings (cart held by friction and the loop)")
    fs = pd.DataFrame({n: free_swing(folder, n) for n in RUNS if "freeswing" in n}).T
    print(fs.to_string(float_format="{:.4f}".format))
    wn = float(fs["w, small amplitude [rad/s]"].mean()); zeta = float(fs["zeta (exponential fit)"].min())
    summary.update({"wn [rad/s]": wn, "equivalent length [m]": G / wn ** 2, "zeta (smallest swing)": zeta})
    shift = []
    for W in (101, 151, 201):
        w = [angle_gain(folder, n, W)[1] for n in ("run4_twotone_trial2", "run4_twotone_trial3", "run7_mass_step_twotone")]
        shift.append(100 * (w[2] / np.mean(w[:2]) - 1))
        print(f"   mass step, fit window {W} ms: wn run7 / run4 trials 2-3 = {shift[-1]:+.2f} %   (trial 2 vs 3: {100 * (w[1] / w[0] - 1):+.2f} %)")
    summary["mass step: wn shift [%]"] = float(np.mean(shift))

    print("\nD. Actuator and friction")
    br = breakaway(folder)
    print("   staircase pulses that start with the cart at rest:")
    print(br.to_string(index=False, float_format="{:+.3f}".format))
    for d in ("+x", "-x"):
        lo, hi = bracket(br, d)
        print(f"   break-away towards {d}: " + (f"between {lo:.2f} V (still) and {hi:.2f} V (moved)" if lo > 0 else f"below {hi:.2f} V (moved on the smallest pulse)"))
        summary[f"break-away {d}: largest level still [V]"] = lo
        summary[f"break-away {d}: smallest level moving [V]"] = hi
    ident = [n for n in RUNS if "twotone_trial" in n or "fading" in n]
    th, r2 = cart_fit(folder, ident)
    print(f"   cart model fitted while moving on run4 and run5 (R2 = {r2:.3f}):")
    for name, v in zip(CART_TERMS, th):
        print(f"      {name:22s} {v:+.4f}")
        summary[f"cart {name}"] = float(v)
    print(f"      in volts: Coulomb +x {th[2] / th[0]:.3f}, -x {th[3] / th[0]:.3f}, viscous {th[1] / th[0]:.3f} V per m/s, offset {th[5] / th[0]:+.3f} V")
    print(f"   simulation's actuator: alpha 2 N/V / 2.63 kg = {2 / 2.63:.3f} m/s2 per V; stiction band to start "
          f"(0.15 V x 2 N/V + 0.35 N)/2 N/V = 0.325 V")
    print("   mass step judged by the cart, same model fitted run by run (a heavier pendulum: lower alpha, higher coupling):")
    for n in ("run4_twotone_trial2", "run4_twotone_trial3", "run7_mass_step_twotone"):
        th_n, r2_n = cart_fit(folder, [n])
        print(f"      {n:24s} alpha {th_n[0]:.3f} m/s2 per V, coupling {th_n[4]:.4f} m (R2 {r2_n:.3f})")
        summary[f"coupling [m], {n}"] = float(th_n[4])
    print("   the fitted model simulated in the rig's closed loop, with each run's excitation:")
    travel = []
    for n in RUNS[1:]:
        if "freeswing" in n:
            continue
        t, x, a, c = load(folder, n)
        xs = closed_loop(nominal(n, t), th, wn, zeta, -K)
        travel.append({"run": n, "measured min": x.min(), "measured max": x.max(), "model min": xs.min(), "model max": xs.max(),
                       "rms difference [m]": float(np.sqrt(np.mean((xs - x) ** 2)))})
    travel = pd.DataFrame(travel)
    print(travel.to_string(index=False, float_format="{:+.3f}".format))
    summary["closed-loop model: worst rms [m]"] = float(travel["rms difference [m]"].max())

    out = ROOT / "results" / "rig"; out.mkdir(parents=True, exist_ok=True)
    pd.Series(summary).to_csv(out / "characterisation.csv", header=["value"])
    print(f"\nSummary: {out / 'characterisation.csv'}")
    if not args.no_plots:
        plot(folder, th, wn, zeta, -K, fs, br, ROOT / "figures" / "rig" / "characterisation.png")


def plot(folder: Path, th: np.ndarray, wn: float, zeta: float, K: float, fs: pd.DataFrame, br: pd.DataFrame, path: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange, green = "#2a78d6", "#eb6834", "#1baf7a"
    mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "axes.edgecolor": ink2, "axes.titlesize": 10, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": grid,
                         "grid.linewidth": 0.6, "xtick.color": ink2, "ytick.color": ink2, "legend.frameon": False,
                         "legend.fontsize": 8, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.2))
    # (a) free swing
    t, x, a, c = load(folder, "run2_freeswing_trial3")
    ax[0, 0].plot(t, np.degrees(a - np.pi), color=blue, linewidth=0.6)
    ax[0, 0].set(title="Free swing, trial 3 (released at about 30°)", xlabel="time [s]", ylabel="pendulum angle from hanging [deg]")
    # (b) frequency against amplitude
    amp = fs["release amplitude [deg]"].to_numpy(); w0 = fs["w, first 3 periods [rad/s]"].to_numpy()
    aa = np.linspace(0, 32, 50)
    ax[0, 1].plot(amp, w0, "o", color=blue, label="measured, first 3 periods")
    ax[0, 1].plot(aa, wn * (1 - np.radians(aa) ** 2 / 16), color=ink2, linewidth=1, label=f"ω = {wn:.3f} (1 − θ₀²/16)")
    ax[0, 1].axhline(2.51, color=orange, linestyle="--", linewidth=1, label="repo's simulated pendulum, 2.51 rad/s")
    ax[0, 1].set(title="Natural frequency against swing amplitude", xlabel="release amplitude [deg]", ylabel="ω [rad/s]", ylim=(2.2, 5.7))
    ax[0, 1].legend(loc="center right")
    # (c) measured against modelled cart travel
    n = "run6_heldout_square"
    t, x, a, c = load(folder, n)
    ax[1, 0].plot(t, x, color=ink, linewidth=1.4, label="measured")
    ax[1, 0].plot(t, closed_loop(nominal(n, t), th, wn, zeta, K), color=green, linewidth=1.2, label="fitted cart model, closed loop")
    ax[1, 0].set(title="Held-out square input: cart position", xlabel="time [s]", ylabel="x [m]")
    ax[1, 0].legend(loc="lower left")
    # (d) break-away: staircase pulses that start from rest
    for d, col, sgn in (("+x", blue, 1), ("-x", orange, -1)):
        b = br[br.direction == d]
        ax[1, 1].scatter(b["level [V]"][b.moved], sgn * b["travel in pulse [counts]"][b.moved] * X_COUNT * 1e3, color=col, s=22,
                         label=f"towards {d}: moved")
        ax[1, 1].scatter(b["level [V]"][~b.moved], sgn * b["travel in pulse [counts]"][~b.moved] * X_COUNT * 1e3, facecolors="none",
                         edgecolors=col, s=40, label=f"towards {d}: stayed (< 3 counts)")
    ax[1, 1].axvline(0.325, color=ink2, linestyle="--", linewidth=1, label="simulation's start threshold, 0.325 V")
    ax[1, 1].axhline(0, color=ink2, linewidth=0.6)
    ax[1, 1].set(title="Staircase pulses from rest: cart travel during each 0.5 s pulse", xlabel="pulse level [V]",
                 ylabel="travel [mm]")
    ax[1, 1].legend(loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"Figure: {path}")


if __name__ == "__main__":
    main()
