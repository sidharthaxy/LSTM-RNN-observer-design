"""
Loading of recorded runs from the Feedback 33-936S rig.

A recording folder holds one CSV per run and one `metadata.txt`:

    data/rig/
        metadata.txt
        run1_rest_trial1.csv                                   noise at rest
        run2_freeswing_trial1.csv ... trial3.csv               identification runs
        run3_deadzone_trial1.csv                               slow command ramp
        run4_twotone_trial1.csv ... trial3.csv                 identification runs
        run5_fading_trial1.csv ... trial3.csv                  identification runs
        run6_heldout_square.csv, _multisine, _chirp, _pulse    held-out runs (never used for learning)
        run7_mass_step_twotone.csv                             identification run with a changed pendulum

The run type is read from the file name (the keywords rest, freeswing, deadzone, twotone, fading,
heldout, mass_step); the run number and the trial suffix are free.

CSV columns (header row required): time_s, cart_pos, pend_angle, command_v.
Units and sign conventions are declared in metadata.txt as `key = value` lines (see
METADATA_TEMPLATE), so raw encoder counts and amplifier volts can be logged as they are.

Internal convention (the one of the simulation): cart position in metres, pendulum angle in
radians with 0 = upright and pi = hanging, input force in newtons, uniform 1 kHz sampling.

The rig has no velocity sensor. `reference_derivatives` builds a reference from the recorded
positions with a zero-phase (centred) Savitzky-Golay fit over 0.3 s. It uses future samples, so it
is far more accurate than any causal estimate, but it is itself an estimate, not a measurement. Its
own noise is measured on the rest run and reported as the floor below which differences mean nothing.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.signal import savgol_filter

from src.benchmark.scenarios import DT, Dataset
from src.plant.pendulum_plant import PendulumParameters

COLUMNS = ("time_s", "cart_pos", "pend_angle", "command_v")

METADATA_TEMPLATE: Dict[str, Any] = {
    "cart_units": "counts",            # "counts" or "m"
    "cart_counts_per_m": 4096.0,
    "cart_sign": 1,                    # +1 if the logged position grows in the direction a positive command pushes
    "angle_units": "counts",           # "counts", "rad" or "deg"
    "angle_counts_per_rev": 4096.0,
    "angle_zero": "hanging",           # where the logged angle reads zero: "hanging" or "upright"
    "angle_sign": 1,                   # +1 or -1: rotation direction of the logged angle (the evaluation checks it and says if it is flipped)
    "command_units": "V",              # "V" or "N"
    "newtons_per_volt": 2.0,           # amplifier gain (20 N at 10 V)
    "track_half_length_m": 0.5,
    "pendulum": "standard",            # free text: which rod / mass was fitted
    "mass_step_pendulum": "",          # free text: what differs in the mass_step runs
    "notes": "",
}

# Keyword in the file name -> role in the evaluation (first match wins)
ROLES = (
    ("heldout", "test"),
    ("mass_step", "train"),
    ("deadzone", "deadzone"),
    ("rest", "rest"),
    ("freeswing", "train"),
    ("twotone", "train"),
    ("fading", "train"),
)
METADATA_COMMENTS = {
    "cart_units": "counts or m",
    "cart_sign": "+1 or -1: direction of the logged cart position",
    "angle_units": "counts, rad or deg",
    "angle_zero": "where the logged angle reads zero: hanging or upright",
    "angle_sign": "+1 or -1: rotation direction of the logged angle",
    "command_units": "V or N",
    "newtons_per_volt": "amplifier gain (used when command_units = V)",
    "pendulum": "which rod / mass was fitted",
    "mass_step_pendulum": "what differs in the mass_step runs",
}


@dataclass
class RigRun:
    """One recorded run, converted to the internal convention and resampled to 1 kHz."""
    name: str
    role: str
    t: np.ndarray           # (N,) uniform, starting at 0
    meas: np.ndarray        # (N, 2) [x (m), theta (rad, 0 = upright)]
    u: np.ndarray           # (N,) commanded force [N]
    near_stop: bool         # the cart came within 2 cm of an end stop
    max_dt_error: float     # largest deviation of the logged sample time from its median [s]


def role_of(name: str) -> str:
    low = name.lower()
    for keyword, role in ROLES:
        if keyword in low:
            return role
    raise ValueError(f"{name}: unknown run type. The file name must contain one of {[k for k, _ in ROLES]}.")


def is_mass_step(name: str) -> bool:
    """Runs recorded with the changed pendulum (their models are not scored on the standard held-out runs)."""
    return "mass_step" in name.lower()


def metadata_text(meta: Dict[str, Any] | None = None) -> str:
    """metadata.txt content: one `key = value` line per field, with a comment where the choice is not obvious."""
    meta = {**METADATA_TEMPLATE, **(meta or {})}
    lines = ["# Recording conditions of the rig runs in this folder. Lines starting with # are ignored."]
    for key, value in meta.items():
        if key in METADATA_COMMENTS:
            lines.append(f"# {METADATA_COMMENTS[key]}")
        lines.append(f"{key} = {value}")
    return "\n".join(lines) + "\n"


def parse_metadata(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"metadata.txt: cannot read the line {line!r} (expected key = value).")
        key, value = (part.strip() for part in line.split("=", 1))
        if key not in METADATA_TEMPLATE:
            raise ValueError(f"metadata.txt: unknown field {key!r}. Known fields: {sorted(METADATA_TEMPLATE)}.")
        default = METADATA_TEMPLATE[key]
        out[key] = value if isinstance(default, str) else type(default)(float(value))
    return out


def load_metadata(folder: Path) -> Dict[str, Any]:
    path = folder / "metadata.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Create it with: python experiments/run_rig_evaluation.py --template > {path}")
    meta = {**METADATA_TEMPLATE, **parse_metadata(path.read_text(encoding="utf-8"))}
    if meta["cart_units"] not in ("counts", "m") or meta["angle_units"] not in ("counts", "rad", "deg"):
        raise ValueError("metadata.txt: cart_units must be counts|m and angle_units counts|rad|deg.")
    if meta["angle_zero"] not in ("hanging", "upright") or meta["command_units"] not in ("V", "N"):
        raise ValueError("metadata.txt: angle_zero must be hanging|upright and command_units V|N.")
    return meta


def convert(raw: np.ndarray, meta: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """raw (N, 3) = [cart_pos, pend_angle, command] in logged units -> (meas (N, 2), u (N,)) in SI."""
    x = raw[:, 0] / meta["cart_counts_per_m"] if meta["cart_units"] == "counts" else raw[:, 0]
    scale = {"counts": 2.0 * np.pi / meta["angle_counts_per_rev"], "rad": 1.0, "deg": np.pi / 180.0}[meta["angle_units"]]
    theta = float(meta["angle_sign"]) * scale * raw[:, 1] + (np.pi if meta["angle_zero"] == "hanging" else 0.0)
    u = raw[:, 2] * (meta["newtons_per_volt"] if meta["command_units"] == "V" else 1.0)
    return np.column_stack((float(meta["cart_sign"]) * x, theta)), u


def load_run(path: Path, meta: Dict[str, Any]) -> RigRun:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = [h.strip() for h in next(reader)]
        missing = [c for c in COLUMNS if c not in header]
        if missing:
            raise ValueError(f"{path.name}: missing column(s) {missing}; found {header}.")
        idx = [header.index(c) for c in COLUMNS]
        rows = np.array([[float(r[i]) for i in idx] for r in reader if r and r[0].strip()])
    if len(rows) < 2000:
        raise ValueError(f"{path.name}: only {len(rows)} samples; at least 2 s of data are needed.")
    t_raw = rows[:, 0] - rows[0, 0]
    steps = np.diff(t_raw)
    if np.any(steps <= 0.0):
        raise ValueError(f"{path.name}: time_s must increase strictly.")
    meas_raw, u_raw = convert(rows[:, 1:], meta)
    t = np.arange(0.0, t_raw[-1], DT)                       # uniform 1 kHz grid
    meas = np.column_stack([np.interp(t, t_raw, meas_raw[:, j]) for j in range(2)])
    k = np.clip(np.searchsorted(t_raw, t, side="right") - 1, 0, len(u_raw) - 1)
    u = u_raw[k]                                            # zero-order hold, as the DAQ applies it
    near_stop = bool(np.abs(meas[:, 0]).max() > meta["track_half_length_m"] - 0.02)
    return RigRun(path.stem, role_of(path.stem), t, meas, u, near_stop, float(np.abs(steps - np.median(steps)).max()))


def load_folder(folder: Path) -> Tuple[Dict[str, Any], List[RigRun]]:
    meta = load_metadata(folder)
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {folder}.")
    return meta, [load_run(f, meta) for f in files]


def reference_derivatives(meas: np.ndarray, window: float = 0.301, order: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(position, velocity, acceleration), each (N, 2): zero-phase Savitzky-Golay fit of the recording."""
    w = int(round(window / DT)) | 1
    out = [savgol_filter(meas, w, order, deriv=d, delta=DT, axis=0, mode="interp") for d in (0, 1, 2)]
    return out[0], out[1], out[2]


def as_dataset(run: RigRun) -> Dataset:
    """A benchmark Dataset whose "true" state is the zero-phase reference (the rig has no velocity sensor)."""
    pos, vel, acc = reference_derivatives(run.meas)
    state = np.column_stack((pos[:, 0], vel[:, 0], pos[:, 1], vel[:, 1]))
    return Dataset(run.name, run.t, state, acc, run.meas, run.u, PendulumParameters())


def write_run(path: Path, t: np.ndarray, x: np.ndarray, theta: np.ndarray, force: np.ndarray, meta: Dict[str, Any]) -> None:
    """Inverse of `load_run` (used to export simulated runs in the rig's file format)."""
    cart = float(meta["cart_sign"]) * x * (meta["cart_counts_per_m"] if meta["cart_units"] == "counts" else 1.0)
    scale = {"counts": meta["angle_counts_per_rev"] / (2.0 * np.pi), "rad": 1.0, "deg": 180.0 / np.pi}[meta["angle_units"]]
    angle = float(meta["angle_sign"]) * (theta - (np.pi if meta["angle_zero"] == "hanging" else 0.0)) * scale
    if meta["cart_units"] == "counts":
        cart = np.round(cart)
    if meta["angle_units"] == "counts":
        angle = np.round(angle)
    cmd = force / (meta["newtons_per_volt"] if meta["command_units"] == "V" else 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(COLUMNS)
        wr.writerows(zip(np.round(t, 6), cart, angle, np.round(cmd, 6)))
