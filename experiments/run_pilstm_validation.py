#!/usr/bin/env python3
"""
Approach B validation: physics-informed PI-LSTM vs. black-box Lb-LSTM (Approach A) on the
Feedback 33-936S rig under extreme conditions.

Scenario (50 s at 1 kHz, open loop):
    * Large initial deflection: theta(0) = 0.5 rad from upright (plant convention). The
      pendulum falls and swings through almost the whole circle, theta in [0.5, 5.8] rad.
    * Encoder noise U(-1.0 deg, +1.0 deg) on theta (plus the testbed's +/-0.2 mm cart noise
      and 4096-count quantization).
    * Sudden parameter shift at t = 25 s: pendulum mass +50 % (m -> 1.5 m, and I -> 1.5 I:
      same geometry, denser rod; --mass-only keeps I).
    * Excitation u(t) = 2.0 sin(3 t) + 1.2 cos(6 t); the cart starts at the drift-cancelling
      velocity v0 = -2 / (3 (M + m)). (The 1.5 rad/s excitation of Approach A drives the cart
      into the +/-0.5 m bumpers once the mass step breaks the momentum balance.)

Observers receive the encoder data and the commanded force only. Data normalization for both
LSTMs comes from the first 5 s of encoder data.

Reported:
    1. Velocity reconstruction RMSE / peaks per window (initial transient, pre-shift, post-shift).
    2. Physical plausibility along the trajectory: lambda_min(M_hat) >= eps_M, kinetic energy
       T_hat = 1/2 q_dot^T M_hat q_dot >= 0, skew-symmetry residual of M_hat_dot - 2 V_m_hat,
       and, for BOTH observers, the implied input gain b_hat = d Phi_hat / d u, which for any
       positive-definite inertia equals M^{-1} B and so must have b_hat_x = (M^{-1})_11 > 0.
    3. Frozen models (weights at t = 50 s) released from rest with u = 0: free swing vs. the
       true post-shift rigid body, and conservation of the learned energy H_hat = T_hat + P_hat.
    4. --seeds N: robustness over noise / initialization seeds.
    5. --ablations: structural and prior ablations of the PI-LSTM.

Usage:
    python experiments/run_pilstm_validation.py [--seeds 5] [--ablations] [--no-plots]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
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

from src.adaptation.jacobian_engine import lstm_forward
from src.baselines.classical_estimators import DirtyDerivativeFilter
from src.observers.blackbox_lstm import LbLSTMObserver, LbLSTMObserverConfig
from src.observers.pilstm_observer import PILSTMObserver, PILSTMObserverConfig
from src.plant.pendulum_plant import PendulumParameters, PendulumPlant
from src.plant.sensor_noise import SensorNoiseConfig, SensorNoiseModel
from src.utils.metrics import compute_chattering_index

DT = 0.001
NORM_WINDOW = 5.0
WINDOWS: Dict[str, Tuple[float, float]] = {
    "transient 0-5 s": (0.0, 5.0),
    "pre-shift 15-25 s": (15.0, 25.0),
    "post-shift 25-30 s": (25.0, 30.0),
    "steady 35-50 s": (35.0, 50.0),
}
LOG_EVERY = 10  # diagnostics that need extra model evaluations are logged every 10 samples


# ---------------------------------------------------------------------- scenario
@dataclass(frozen=True)
class Scenario:
    t_final: float = 50.0
    theta0: float = 0.5
    theta_noise_deg: float = 1.0
    t_shift: float = 25.0
    mass_factor: float = 1.5
    scale_inertia: bool = True
    actuator_effects: bool = False
    noise_seed: int = 42


@dataclass
class Trajectory:
    t: np.ndarray            # (N,)
    state: np.ndarray        # (N, 4) true [x, x_dot, theta, theta_dot]
    accel: np.ndarray        # (N, 2) true [x_ddot, theta_ddot]
    meas: np.ndarray         # (N, 2) encoder [x, theta]
    u: np.ndarray            # (N,) commanded force
    M_true: np.ndarray       # (N, 2, 2) true inertia at theta(t)
    b_true: np.ndarray       # (N, 2) true input gain M^{-1} B
    T_true: np.ndarray       # (N,) true kinetic energy
    params_after: PendulumParameters

    @property
    def velocities(self) -> np.ndarray:
        return self.state[:, [1, 3]]


def excitation(t: float) -> float:
    return 2.0 * np.sin(3.0 * t) + 1.2 * np.cos(6.0 * t)


def shifted_params(p: PendulumParameters, sc: Scenario) -> PendulumParameters:
    return replace(p, m=sc.mass_factor * p.m, I=sc.mass_factor * p.I if sc.scale_inertia else p.I)


def simulate_scenario(sc: Scenario) -> Trajectory:
    p = PendulumParameters()
    plant = PendulumPlant(p)
    plant.reset(np.array([0.0, -2.0 / (3.0 * (p.M + p.m)), sc.theta0, 0.0]))
    noise = SensorNoiseModel(SensorNoiseConfig(theta_noise_deg=sc.theta_noise_deg))
    noise.seed(sc.noise_seed)
    p_after = shifted_params(p, sc)

    t = np.arange(0.0, sc.t_final, DT)
    N = t.size
    state = np.zeros((N, 4))
    accel = np.zeros((N, 2))
    meas = np.zeros((N, 2))
    u = np.zeros(N)
    M_true = np.zeros((N, 2, 2))
    shifted = False
    for k, tk in enumerate(t):
        if not shifted and tk >= sc.t_shift - 1e-9:
            plant.params = p_after
            shifted = True
        s = plant.state.to_array()
        state[k] = s
        meas[k] = noise.measure(s[0], s[2])
        u[k] = excitation(float(tk))
        f = noise.apply_actuator_deadzone_and_stiction(u[k], s[1]) if sc.actuator_effects else u[k]
        accel[k] = plant.forward_dynamics(s, f)
        M_true[k] = plant.mass_matrix(s[2])
        plant.step(f, DT)
    if np.any(np.abs(state[:, 0]) >= p.x_limit - 1e-9):
        print("[!] cart reached a track bumper: non-smooth dynamics in the scenario")
    b_true = np.linalg.solve(M_true, np.broadcast_to(np.array([1.0, 0.0]), (N, 2))[..., None])[..., 0]
    vel = state[:, [1, 3]]
    T_true = 0.5 * np.einsum("ni,nij,nj->n", vel, M_true, vel)
    return Trajectory(t, state, accel, meas, u, M_true, b_true, T_true, p_after)


@dataclass(frozen=True)
class Normalization:
    """Data-driven signal normalization from the first NORM_WINDOW seconds of encoder data."""
    pos_offset: Tuple[float, float]
    pos_scale: Tuple[float, float]
    vel_scale: Tuple[float, float]
    u_scale: float


def normalization(traj: Trajectory) -> Normalization:
    n0 = min(len(traj.t), int(round(NORM_WINDOW / DT)))
    y = traj.meas[:n0]
    mu = y.mean(axis=0)
    half = np.maximum(np.abs(y - mu).max(axis=0), 1e-3)
    vel = np.zeros((n0, 2))
    for j in range(2):
        f = DirtyDerivativeFilter(0.05)
        f.reset(y[0, j])
        vel[:, j] = [f.update(v, DT) for v in y[:, j]]
    vmax = np.maximum(np.abs(vel).max(axis=0), 1e-3)
    return Normalization(
        (float(mu[0]), float(mu[1])), (float(1 / half[0]), float(1 / half[1])),
        (float(1 / vmax[0]), float(1 / vmax[1])), float(1 / max(np.abs(traj.u[:n0]).max(), 1e-3)),
    )


def pilstm_config(norm: Normalization, seed: int = 0, **overrides: Any) -> PILSTMObserverConfig:
    return replace(PILSTMObserverConfig(velocity_scale=norm.vel_scale, seed=seed), **overrides)


def lblstm_config(norm: Normalization, seed: int = 0) -> LbLSTMObserverConfig:
    return LbLSTMObserverConfig(
        input_scale=(*norm.pos_scale, *norm.vel_scale, norm.u_scale),
        input_offset=(*norm.pos_offset, 0.0, 0.0, 0.0),
        seed=seed,
    )


# ---------------------------------------------------------------------- observer runs
@dataclass
class ObserverRun:
    velocity: np.ndarray         # (N, 2)
    phi: np.ndarray              # (N, 2) acceleration model output
    input_gain: np.ndarray       # (N // LOG_EVERY, 2) implied d Phi_hat / d u
    diverged_at: float | None
    # PI-LSTM only (empty for the black box)
    lam_min: np.ndarray
    T_hat: np.ndarray
    M_hat: np.ndarray            # (N // LOG_EVERY, 2, 2) at q_hat(t)
    skew_max: float
    friction_power: np.ndarray
    block_norms: np.ndarray      # (N // LOG_EVERY, 4)
    block_drift: np.ndarray      # (N // LOG_EVERY, 4): ||theta_b(t) - theta_b(0)||
    theta: np.ndarray            # final parameters


def _x0(traj: Trajectory) -> np.ndarray:
    return np.array([traj.meas[0, 0], 0.0, traj.meas[0, 1], 0.0])


def run_pilstm(traj: Trajectory, cfg: PILSTMObserverConfig) -> ObserverRun:
    obs = PILSTMObserver(cfg)
    obs.reset(_x0(traj))
    N = len(traj.t)
    nl = (N + LOG_EVERY - 1) // LOG_EVERY
    vel, phi = np.full((N, 2), np.nan), np.full((N, 2), np.nan)
    lam_min, T_hat, fpow = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)
    gain, M_hat, norms = np.full((nl, 2), np.nan), np.full((nl, 2, 2), np.nan), np.full((nl, 4), np.nan)
    drift = np.full((nl, 4), np.nan)
    blocks = [obs.layout.block_slice(b) for b in obs.layout.BLOCKS]
    B = obs.net.B[:, 0]
    rng = np.random.default_rng(0)
    skew_max = 0.0
    diverged = None
    for k in range(N):
        est = obs.update(traj.meas[k], traj.u[k], DT)
        if not np.all(np.isfinite(est)) or np.abs(est).max() > 1e3:
            diverged = float(traj.t[k])
            break
        c = obs.last_cache
        assert c is not None
        vel[k] = est[[1, 3]]
        phi[k] = c.phi
        M = c.inertia.M
        lam_min[k] = np.linalg.eigvalsh(M)[0]
        T_hat[k] = 0.5 * c.q_dot @ M @ c.q_dot
        fpow[k] = c.q_dot @ c.friction_force
        if k % LOG_EVERY == 0:
            j = k // LOG_EVERY
            gain[j] = np.linalg.solve(M, B)
            M_hat[j] = M
            norms[j] = list(obs.block_norms().values())
            drift[j] = [np.linalg.norm(obs.theta[sl] - obs.theta0[sl]) for sl in blocks]
            N_skew = obs.net.skew_residual(obs.theta, c.q, c.q_dot)
            z = rng.normal(size=2)
            skew_max = max(skew_max, abs(float(z @ N_skew @ z)) / float(z @ z))
    return ObserverRun(vel, phi, gain, diverged, lam_min, T_hat, M_hat, skew_max, fpow, norms, drift, np.copy(obs.theta))


def run_lblstm(traj: Trajectory, cfg: LbLSTMObserverConfig) -> ObserverRun:
    obs = LbLSTMObserver(cfg)
    obs.reset(_x0(traj))
    N = len(traj.t)
    nl = (N + LOG_EVERY - 1) // LOG_EVERY
    vel, phi = np.full((N, 2), np.nan), np.full((N, 2), np.nan)
    gain = np.full((nl, 2), np.nan)
    du = 1e-3
    diverged = None
    for k in range(N):
        est = obs.update(traj.meas[k], traj.u[k], DT)
        if not np.all(np.isfinite(est)) or np.abs(est).max() > 1e3:
            diverged = float(traj.t[k])
            break
        vel[k] = est[[1, 3]]
        phi[k] = obs.phi_hat
        if k % LOG_EVERY == 0:
            u = np.array([traj.u[k]])
            gain[k // LOG_EVERY] = (obs.lstm(u + du).phi - obs.lstm(u - du).phi) / (2.0 * du)
    empty = np.zeros(0)
    return ObserverRun(vel, phi, gain, diverged, empty, empty, np.zeros((0, 2, 2)), float("nan"),
                       empty, np.zeros((0, 4)), np.zeros((0, 4)), np.copy(obs.theta))


def run_dirty_derivative(traj: Trajectory, tau_d: float = 0.02) -> np.ndarray:
    vel = np.zeros((len(traj.t), 2))
    for j in range(2):
        f = DirtyDerivativeFilter(tau_d)
        f.reset(traj.meas[0, j])
        vel[:, j] = [f.update(y, DT) for y in traj.meas[:, j]]
    return vel


# ---------------------------------------------------------------------- metrics
def velocity_metrics(traj: Trajectory, v_hat: np.ndarray) -> Dict[str, float]:
    t = traj.t
    err = traj.velocities - v_hat
    out: Dict[str, float] = {}
    for name, (a, b) in WINDOWS.items():
        m = (t >= a) & (t < b)
        rmse = np.sqrt(np.nanmean(err[m] ** 2, axis=0))
        out[f"RMSE x_dot {name}"] = rmse[0]
        out[f"RMSE th_dot {name}"] = rmse[1]
    for name, (a, b) in {"0-2 s": (0.0, 2.0), "25-27 s": (25.0, 27.0)}.items():
        m = (t >= a) & (t < b)
        out[f"peak |th_dot err| {name}"] = float(np.nanmax(np.abs(err[m, 1])))
    ss = t >= WINDOWS["steady 35-50 s"][0]
    if np.all(np.isfinite(v_hat[ss])):
        out["chatter th_dot (steady)"] = float(
            compute_chattering_index(v_hat[ss, 1], traj.velocities[ss, 1], DT)["chattering_ratio"] or np.nan)
    return out


def plausibility_metrics(traj: Trajectory, run: ObserverRun, eps_M: float | None) -> Dict[str, float | str]:
    t_log = traj.t[::LOG_EVERY]
    b_true = traj.b_true[::LOG_EVERY]
    ok = np.all(np.isfinite(run.input_gain), axis=1)
    g = run.input_gain[ok]
    ss = t_log[ok] >= WINDOWS["steady 35-50 s"][0]
    rel = np.linalg.norm(g - b_true[ok], axis=1) / np.linalg.norm(b_true[ok], axis=1)
    out: Dict[str, float | str] = {
        "implied (M^-1)_11 <= 0 [% of samples]": 100.0 * float(np.mean(g[:, 0] <= 0.0)),
        "implied M^-1 B rel. error (steady)": float(np.median(rel[ss])) if ss.any() else np.nan,
    }
    fin = np.all(np.isfinite(run.phi), axis=1) & (traj.t >= WINDOWS["steady 35-50 s"][0])
    if fin.any():
        nmse = np.mean((run.phi[fin] - traj.accel[fin]) ** 2, axis=0) / np.var(traj.accel[fin], axis=0)
        out["online accel NMSE x_ddot (steady)"] = float(nmse[0])
        out["online accel NMSE th_ddot (steady)"] = float(nmse[1])
    if eps_M is None:
        for key in ("min lambda_min(M_hat)", "min T_hat", "max |z^T (M_dot - 2V) z| / |z|^2",
                    "M_hat rel. error @ 25 s", "M_hat rel. error @ 50 s", "friction injects power [% of samples]"):
            out[key] = "n/a (no M_hat)"
        return out
    out["min lambda_min(M_hat)"] = float(np.nanmin(run.lam_min))
    out["min T_hat"] = float(np.nanmin(run.T_hat))
    out["max |z^T (M_dot - 2V) z| / |z|^2"] = run.skew_max
    M_true_log = traj.M_true[::LOG_EVERY]
    for label, tq in (("25 s", 25.0 - DT), ("50 s", float(traj.t[-1]))):
        j = min(int(round(tq / DT)) // LOG_EVERY, len(run.M_hat) - 1)
        out[f"M_hat rel. error @ {label}"] = float(
            np.linalg.norm(run.M_hat[j] - M_true_log[j]) / np.linalg.norm(M_true_log[j]))
    out["friction injects power [% of samples]"] = 100.0 * float(np.nanmean(run.friction_power < 0.0))
    return out


# ---------------------------------------------------------------------- frozen-model free swing
@dataclass
class FreeSwing:
    t: np.ndarray
    truth: np.ndarray        # (N, 4)
    pi_twin: np.ndarray      # (N, 4)
    a_twin: np.ndarray       # (N, 4)
    H_pi: np.ndarray         # learned energy along the PI twin
    H_true_truth: np.ndarray  # true energy along the truth
    H_true_pi: np.ndarray    # true energy along the PI twin
    H_true_a: np.ndarray     # true energy along the black-box twin
    T_peak: float            # peak true kinetic energy of the swing (energy scale)


def _rk4(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, h: float) -> np.ndarray:
    k1 = f(x)
    k2 = f(x + 0.5 * h * k1)
    k3 = f(x + 0.5 * h * k2)
    k4 = f(x + h * k3)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def free_swing(pi_cfg: PILSTMObserverConfig, pi_theta: np.ndarray, a_cfg: LbLSTMObserverConfig,
               a_theta: np.ndarray, params: PendulumParameters, horizon: float = 10.0,
               theta0: float = np.pi - 1.0) -> FreeSwing:
    """Frozen models released from rest (u = 0) vs. the true frictionless rigid body."""
    t = np.arange(0.0, horizon, DT)
    q0 = np.array([0.0, theta0])
    u0 = np.zeros(1)

    true_p = replace(params, b=0.0, d=0.0, x_limit=np.inf)
    plant = PendulumPlant(true_p)
    truth = np.zeros((t.size, 4))
    s = np.array([q0[0], 0.0, q0[1], 0.0])
    for k in range(t.size):
        truth[k] = s
        s = _rk4(lambda z: plant.state_derivative(z, 0.0), s, DT)
    H_true = np.array([plant.total_energy(z) for z in truth])
    T_peak = max(0.5 * z[[1, 3]] @ plant.mass_matrix(z[2]) @ z[[1, 3]] for z in truth)

    pi = PILSTMObserver(pi_cfg)
    net, th = pi.net, pi_theta

    def f_pi(z: np.ndarray) -> np.ndarray:
        return np.concatenate((z[2:], net.conservative_acceleration(th, z[:2], z[2:], u0)))

    zp = np.concatenate((q0, np.zeros(2)))
    pi_tr = np.zeros((t.size, 4))
    H_pi = np.zeros(t.size)
    for k in range(t.size):
        pi_tr[k] = zp[[0, 2, 1, 3]]
        H_pi[k] = net.kinetic_energy(th, zp[:2], zp[2:]) + net.potential_energy(th, zp[:2])
        zp = _rk4(f_pi, zp, DT)

    a = LbLSTMObserver(a_cfg)
    a.theta = np.copy(a_theta)
    L = a_cfg.hidden_dim

    def f_a(z: np.ndarray) -> np.ndarray:
        x1, x2, c_hat, h_hat = z[:2], z[2:4], z[4:4 + L], z[4 + L:]
        cache = lstm_forward(a.theta, a.build_zeta(x1, x2, u0, h_hat), c_hat, a.layout)
        return np.concatenate((x2, cache.phi, a_cfg.b_c * (cache.c - c_hat), a_cfg.b_h * (cache.h - h_hat)))

    za = np.concatenate((q0, np.zeros(2), np.zeros(2 * L)))
    a_tr = np.full((t.size, 4), np.nan)
    for k in range(t.size):
        a_tr[k] = za[[0, 2, 1, 3]]
        za = _rk4(f_a, za, DT)
        if not np.all(np.isfinite(za[:4])) or np.abs(za[:4]).max() > 1e3:
            break
    H_true_pi = np.array([plant.total_energy(z) for z in pi_tr])
    H_true_a = np.array([plant.total_energy(z) if np.all(np.isfinite(z)) else np.nan for z in a_tr])
    return FreeSwing(t, truth, pi_tr, a_tr, H_pi, H_true, H_true_pi, H_true_a, float(T_peak))


# ---------------------------------------------------------------------- seeds / ablations
def _seed_trial(args: Tuple[int, Scenario]) -> Dict[str, float]:
    seed, sc = args
    traj = simulate_scenario(replace(sc, noise_seed=sc.noise_seed + seed))
    norm = normalization(traj)
    row: Dict[str, float] = {"seed": seed}
    for name, run in (("PI-LSTM", run_pilstm(traj, pilstm_config(norm, seed))),
                      ("Lb-LSTM", run_lblstm(traj, lblstm_config(norm, seed)))):
        m = velocity_metrics(traj, run.velocity)
        row[f"{name} diverged"] = float(run.diverged_at is not None)
        for key in ("RMSE x_dot steady 35-50 s", "RMSE th_dot steady 35-50 s",
                    "RMSE th_dot post-shift 25-30 s", "peak |th_dot err| 0-2 s"):
            row[f"{name} {key}"] = m[key]
        row[f"{name} (M^-1)_11<=0 %"] = float(plausibility_metrics(traj, run, None)["implied (M^-1)_11 <= 0 [% of samples]"])
    return row


ABLATIONS: Dict[str, Dict[str, Any]] = {
    "default (prior diag(2, 0.2))": {},
    "prior diag(1, 0.1)": {"inertia_init": (1.0, 0.1)},
    "prior diag(5, 0.5)": {"inertia_init": (5.0, 0.5)},
    "prior = true M(theta=pi/2), pre-shift": {"inertia_init": (2.63, 0.129)},
    "inertia frozen at prior (gamma_M = 0)": {"gamma_M": 0.0},
    "Euclidean-metric gradient": {"metric": "euclidean"},
    "x not cyclic (M, P depend on x)": {"cyclic": (False, False)},
    "random tanh features, no 2nd harmonic": {"harmonics": 1, "n_features": 24},
    "no friction LSTM adaptation": {"gamma_F_gates": 0.0, "gamma_F_out": 0.0},
    "unconstrained friction output F = W_h^T h": {"dissipative_friction": False},
}


def _ablation_trial(args: Tuple[str, Scenario]) -> Dict[str, object]:
    name, sc = args
    traj = simulate_scenario(sc)
    norm = normalization(traj)
    cfg = pilstm_config(norm, **ABLATIONS[name])
    if not cfg.cyclic[0]:
        cfg = replace(cfg, position_offset=(norm.pos_offset[0], 0.0), position_scale=(norm.pos_scale[0], 1.0))
    run = run_pilstm(traj, cfg)
    row: Dict[str, object] = {"variant": name, "diverged at [s]": run.diverged_at}
    if run.diverged_at is None:
        m = velocity_metrics(traj, run.velocity)
        p = plausibility_metrics(traj, run, cfg.eps_M)
        row.update({
            "RMSE x_dot steady": m["RMSE x_dot steady 35-50 s"],
            "RMSE th_dot steady": m["RMSE th_dot steady 35-50 s"],
            "RMSE th_dot post-shift": m["RMSE th_dot post-shift 25-30 s"],
            "M_hat rel. err @ 50 s": p["M_hat rel. error @ 50 s"],
        })
    return row


# ---------------------------------------------------------------------- plotting
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
COLORS = {"PI-LSTM": "#2a78d6", "Lb-LSTM (A)": "#eb6834", "Dirty derivative": "#1baf7a"}


def _style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 1.1, "font.family": "DejaVu Sans",
    })


def _moving_rms(x: np.ndarray, window: int) -> np.ndarray:
    return np.sqrt(np.convolve(np.nan_to_num(x) ** 2, np.ones(window) / window, mode="same"))


def make_figures(out_dir: Path, traj: Trajectory, vel: Dict[str, np.ndarray], pi: ObserverRun,
                 bb: ObserverRun, pi_cfg: PILSTMObserverConfig, swing: FreeSwing, t_shift: float) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    t = traj.t
    saved: List[Path] = []
    order = ("Dirty derivative", "Lb-LSTM (A)", "PI-LSTM")

    # 1. Velocity reconstruction around the three stress events.
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex="col")
    windows = [(0.0, 3.0, "Large deflection, 0-3 s"), (24.0, 28.0, "Mass +50 % at 25 s"),
               (46.0, 50.0, "Steady state, 46-50 s")]
    labels = [r"$\dot{x}$ [m/s]", r"$\dot{\theta}$ [rad/s]"]
    for col, (a, b, title) in enumerate(windows):
        m = (t >= a) & (t <= b)
        for row in range(2):
            ax = axes[row, col]
            for name in order:
                dd = name == "Dirty derivative"
                ax.plot(t[m], vel[name][m, row], color=COLORS[name], label=name,
                        lw=0.6 if dd else 1.2, alpha=0.55 if dd else 1.0)
            ax.plot(t[m], traj.velocities[m, row], color=INK, lw=1.4, ls="--", label="True")
            if a <= t_shift <= b:
                ax.axvline(t_shift, color=INK_2, lw=0.8, ls=":")
            if col == 0:
                ax.set_ylabel(labels[row])
            if row == 0:
                ax.set_title(title, loc="left")
            else:
                ax.set_xlabel("Time [s]")
    # the dirty derivative's noise would dominate the y-range: clip to the truth's range
    for row in range(2):
        for col, (a, b, _) in enumerate(windows):
            m = (t >= a) & (t <= b)
            lo, hi = traj.velocities[m, row].min(), traj.velocities[m, row].max()
            pad = 0.25 * (hi - lo) + 0.05
            axes[row, col].set_ylim(lo - pad, hi + pad)
    h, n = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, n, loc="upper right", ncol=4, bbox_to_anchor=(0.99, 0.995))
    fig.suptitle("Velocity reconstruction, U(±1°) noise", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    saved.append(out_dir / "velocity_tracking.png")
    fig.savefig(saved[-1])
    plt.close(fig)

    # 2. Error norm.
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for name in order:
        err = np.linalg.norm(traj.velocities - vel[name], axis=1)
        ax.semilogy(t, _moving_rms(err, 200), color=COLORS[name], label=name, lw=1.2)
    ax.axvline(t_shift, color=INK_2, lw=0.8, ls=":")
    ax.text(t_shift + 0.3, 0.03, "mass +50 %", transform=ax.get_xaxis_transform(), color=INK_2, fontsize=8, va="bottom")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\|x_2 - \hat{x}_2\|$ (200 ms RMS)")
    ax.set_title("Velocity estimation error norm", loc="left")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3)
    fig.tight_layout()
    saved.append(out_dir / "error_norm.png")
    fig.savefig(saved[-1])
    plt.close(fig)

    # 3. Inertia learning and implied input gain.
    tl = t[::LOG_EVERY]
    Mt = traj.M_true[::LOG_EVERY]
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.8), sharex=True)
    for ax, (i, j, lab) in zip(axes.ravel()[:3], ((0, 0, r"$M_{11}$ [kg]"), (0, 1, r"$M_{12}$ [kg m]"),
                                                 (1, 1, r"$M_{22}$ [kg m²]"))):
        ax.plot(tl, Mt[:, i, j], color=INK, lw=1.3, ls="--", label=r"true $M(\theta(t))$")
        ax.plot(tl, pi.M_hat[:, i, j], color=COLORS["PI-LSTM"], lw=1.1, label=r"$\hat M(\hat q(t))$")
        ax.axvline(t_shift, color=INK_2, lw=0.8, ls=":")
        ax.set_ylabel(lab)
        ax.set_title(f"Learned inertia entry {lab.split(' ')[0]}", loc="left")
    axes[0, 0].legend(loc="lower right")
    ax = axes[1, 1]
    ax.plot(tl, traj.b_true[::LOG_EVERY, 0], color=INK, lw=1.3, ls="--", label=r"true $(M^{-1})_{11}$")
    ax.plot(tl, bb.input_gain[:, 0], color=COLORS["Lb-LSTM (A)"], lw=0.9, label=r"Lb-LSTM $\partial\hat\Phi_x/\partial u$")
    ax.plot(tl, pi.input_gain[:, 0], color=COLORS["PI-LSTM"], lw=1.1, label=r"PI-LSTM $(\hat M^{-1})_{11}$")
    ax.axhline(0.0, color=INK_2, lw=0.8)
    ax.axvline(t_shift, color=INK_2, lw=0.8, ls=":")
    ax.set_ylabel(r"input gain [1/kg]")
    ax.set_title("Implied cart input gain (must be > 0)", loc="left")
    ax.set_ylim(-0.1, 0.75)
    ax.legend(loc="upper left", ncol=3, fontsize=7)
    for bottom_ax in axes[1]:
        bottom_ax.set_xlabel("Time [s]")
    fig.tight_layout()
    saved.append(out_dir / "inertia_learning.png")
    fig.savefig(saved[-1])
    plt.close(fig)

    # 4. Physical invariants along the trajectory.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    ax.plot(t, pi.lam_min, color=COLORS["PI-LSTM"], lw=1.1, label=r"$\lambda_{\min}(\hat M(\hat q))$")
    ax.plot(t, np.linalg.eigvalsh(traj.M_true)[:, 0], color=INK, lw=1.2, ls="--", label=r"true $\lambda_{\min}(M(q))$")
    ax.axhline(pi_cfg.eps_M, color=INK_2, lw=0.9, ls=":")
    ax.text(0.5, pi_cfg.eps_M, r"structural floor $\epsilon_M$", color=INK_2, fontsize=8, va="bottom")
    ax.set_ylim(0.0, None)
    ax.set_xlabel("Time [s]")
    ax.set_title("Positive-definiteness margin", loc="left")
    ax.legend(loc="upper right")
    ax = axes[1]
    ax.plot(t, traj.T_true, color=INK, lw=1.2, ls="--", label="true T")
    ax.plot(t, pi.T_hat, color=COLORS["PI-LSTM"], lw=0.9, label=r"$\hat T = \frac{1}{2}\hat{\dot q}^T\hat M\hat{\dot q}$")
    ax.axvline(t_shift, color=INK_2, lw=0.8, ls=":")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Kinetic energy [J]")
    ax.set_title("Kinetic energy metric (never negative)", loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    saved.append(out_dir / "physical_invariants.png")
    fig.savefig(saved[-1])
    plt.close(fig)

    # 5. Frozen models: free swing and learned-energy conservation.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    ax.plot(swing.t, swing.truth[:, 2] - np.pi, color=INK, lw=1.4, ls="--", label="True rigid body (post-shift)")
    ax.plot(swing.t, swing.a_twin[:, 2] - np.pi, color=COLORS["Lb-LSTM (A)"], lw=1.1, label="Frozen Lb-LSTM")
    ax.plot(swing.t, swing.pi_twin[:, 2] - np.pi, color=COLORS["PI-LSTM"], lw=1.2, label="Frozen PI-LSTM")
    lim = 1.6 * np.abs(swing.truth[:, 2] - np.pi).max()
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\theta - \pi$ [rad]")
    ax.set_title("Free swing from rest, u = 0 (unseen)", loc="left")
    ax.legend(loc="lower left", fontsize=7)
    ax = axes[1]
    H0 = swing.H_true_truth[0]
    ax.plot(swing.t, swing.H_true_truth - H0, color=INK, lw=1.4, ls="--", label="True rigid body")
    ax.plot(swing.t, swing.H_true_a - H0, color=COLORS["Lb-LSTM (A)"], lw=1.1, label="Frozen Lb-LSTM")
    ax.plot(swing.t, swing.H_true_pi - H0, color=COLORS["PI-LSTM"], lw=1.2, label="Frozen PI-LSTM")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$H(t) - H(0)$ [J]")
    ax.set_title("True mechanical energy along each free response", loc="left")
    drift = np.abs(swing.H_pi - swing.H_pi[0]).max()
    ax.text(0.02, 0.04, f"PI-LSTM's own learned energy $\\hat H$: max drift {drift:.1e} J",
            transform=ax.transAxes, color=INK_2, fontsize=8)
    ax.legend(loc="center right", fontsize=7)
    fig.tight_layout()
    saved.append(out_dir / "free_swing_energy.png")
    fig.savefig(saved[-1])
    plt.close(fig)

    # 6. Adaptation per block: drift from initialization.
    fig, ax = plt.subplots(figsize=(11, 3.4))
    names = ("M", "V", "G", "F")
    bars = np.array([pi_cfg.w_bar_M, pi_cfg.w_bar_V, pi_cfg.w_bar_G, pi_cfg.w_bar_F])
    styles = {"M": ("-", 1.4), "V": (":", 1.3), "G": ("--", 1.3), "F": ("-.", 1.2)}
    for j, blk in enumerate(names):
        ls, lw = styles[blk]
        ax.semilogy(tl[1:], np.maximum(pi.block_drift[1:, j], 1e-6), color=COLORS["PI-LSTM"], ls=ls, lw=lw,
                    label=rf"$\|\hat\theta_{blk}(t) - \hat\theta_{blk}(0)\|$")
    ax.axvline(t_shift, color=INK_2, lw=0.8, ls=":")
    frac = np.nanmax(pi.block_norms / bars[None, :])
    ax.text(0.01, 0.04, rf"every block stays below {100 * frac:.0f} % of its projection radius $\bar W$",
            transform=ax.transAxes, color=INK_2, fontsize=8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Drift from initialization")
    ax.set_ylim(1e-4, 5.0)
    ax.set_title("Adaptation per parameter block", loc="left")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4)
    fig.tight_layout()
    saved.append(out_dir / "block_adaptation.png")
    fig.savefig(saved[-1])
    plt.close(fig)
    return saved


# ---------------------------------------------------------------------- main
def _fmt(x: object) -> str:
    if isinstance(x, str) or x is None:
        return str(x)
    x = float(x)  # type: ignore[arg-type]
    if not np.isfinite(x):
        return "nan"
    return f"{x:.2e}" if (x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e4)) else f"{x:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--t-final", type=float, default=50.0)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=0, help="extra noise/initialization seeds (0 = skip)")
    parser.add_argument("--ablations", action="store_true", help="PI-LSTM structural / prior ablations")
    parser.add_argument("--mass-only", action="store_true", help="shift m only (keep I)")
    parser.add_argument("--actuator-effects", action="store_true", help="enable PCI-1711 dead-zone / stiction")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--fig-dir", type=Path, default=ROOT_DIR / "figures" / "approach_b")
    parser.add_argument("--results-dir", type=Path, default=ROOT_DIR / "results" / "approach_b")
    args = parser.parse_args()

    sc = Scenario(t_final=args.t_final, noise_seed=args.noise_seed, scale_inertia=not args.mass_only,
                  actuator_effects=args.actuator_effects)
    pd.set_option("display.width", 220)
    print("=" * 100)
    print("APPROACH B: PHYSICS-INFORMED PI-LSTM vs BLACK-BOX Lb-LSTM, FEEDBACK 33-936S, EXTREME CONDITIONS")
    print(f"theta(0) = {sc.theta0} rad from upright; noise U(±{sc.theta_noise_deg}°); "
          f"pendulum mass ×{sc.mass_factor} at t = {sc.t_shift} s; u = 2 sin 3t + 1.2 cos 6t")
    print("=" * 100)

    t0 = time.perf_counter()
    traj = simulate_scenario(sc)
    norm = normalization(traj)
    pi_cfg = pilstm_config(norm)
    a_cfg = lblstm_config(norm)
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
    with ProcessPoolExecutor(max_workers=3, mp_context=ctx) as pool:
        f_pi = pool.submit(run_pilstm, traj, pi_cfg)
        f_a = pool.submit(run_lblstm, traj, a_cfg)
        f_dd = pool.submit(run_dirty_derivative, traj)
        pi, bb, dd = f_pi.result(), f_a.result(), f_dd.result()
    print(f"[+] simulated {len(traj.t)} samples in {time.perf_counter() - t0:.1f} s "
          f"(PI-LSTM {pi.theta.size} parameters, Lb-LSTM {bb.theta.size} weights)")
    for name, run in (("PI-LSTM", pi), ("Lb-LSTM (A)", bb)):
        if run.diverged_at is not None:
            print(f"[!] {name} diverged at t = {run.diverged_at:.3f} s")

    vel = {"PI-LSTM": pi.velocity, "Lb-LSTM (A)": bb.velocity, "Dirty derivative": dd}
    table = pd.DataFrame({k: velocity_metrics(traj, v) for k, v in vel.items()})
    print("\n1. Velocity reconstruction (RMSE in m/s and rad/s; chatter = TV(estimate) / TV(truth)):")
    print(table.map(_fmt).to_string())

    phys = pd.DataFrame({"PI-LSTM": plausibility_metrics(traj, pi, pi_cfg.eps_M),
                         "Lb-LSTM (A)": plausibility_metrics(traj, bb, None)})
    print(f"\n2. Physical plausibility along the trajectory (eps_M = {pi_cfg.eps_M}):")
    print(phys.map(_fmt).to_string())

    swing = free_swing(pi_cfg, pi.theta, a_cfg, bb.theta, traj.params_after)
    drift = np.abs(swing.H_pi - swing.H_pi[0]).max()
    th_rmse = {n: float(np.sqrt(np.nanmean((tw[:, 2] - swing.truth[:, 2]) ** 2)))
               for n, tw in (("PI-LSTM", swing.pi_twin), ("Lb-LSTM (A)", swing.a_twin))}
    a_valid = float(np.mean(np.isfinite(swing.a_twin[:, 2])))
    print("\n3. Frozen models (weights at t_final), free swing from rest at theta = pi - 1 rad, u = 0, 10 s:")
    print(f"   PI-LSTM learned energy drift max |H_hat(t) - H_hat(0)| = {drift:.2e} J "
          f"(true peak kinetic energy of the swing {swing.T_peak:.3f} J)")
    print(f"   theta RMSE vs true rigid body: PI-LSTM {th_rmse['PI-LSTM']:.3f} rad, "
          f"Lb-LSTM {th_rmse['Lb-LSTM (A)']:.3f} rad (finite for {100 * a_valid:.0f} % of the horizon)")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.results_dir / "velocity_metrics.csv")
    phys.to_csv(args.results_dir / "plausibility_metrics.csv")
    np.savez(args.results_dir / "pilstm_final_theta.npz", theta=pi.theta)

    if args.seeds > 0:
        seeds = list(range(1, args.seeds + 1))
        print(f"\n4. Robustness over noise / initialization seeds {seeds}:")
        with ProcessPoolExecutor(mp_context=ctx) as pool:
            rows = list(pool.map(_seed_trial, [(s, sc) for s in seeds]))
        rob = pd.DataFrame(rows).set_index("seed")
        summary = rob.agg(["median", "min", "max"]).T
        print(summary.map(_fmt).to_string())
        summary.to_csv(args.results_dir / "seed_robustness.csv")
    if args.ablations:
        print("\n5. PI-LSTM ablations (same trajectory, noise seed as above):")
        with ProcessPoolExecutor(mp_context=ctx) as pool:
            abl_rows = list(pool.map(_ablation_trial, [(n, sc) for n in ABLATIONS]))
        abl = pd.DataFrame(abl_rows).set_index("variant")
        print(abl.map(_fmt).to_string())
        abl.to_csv(args.results_dir / "ablations.csv")

    if not args.no_plots:
        paths = make_figures(args.fig_dir, traj, vel, pi, bb, pi_cfg, swing, sc.t_shift)
        print("\n[+] Figures:")
        for p in paths:
            print(f"    {p.relative_to(ROOT_DIR)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
