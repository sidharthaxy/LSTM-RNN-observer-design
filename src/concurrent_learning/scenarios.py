"""
Training and test inputs for the concurrent-learning study.

Weak-excitation (non-PE) training commands for the open-loop cart-pendulum:
    decaying sine        u(t) = A exp(-t / tau) sin(omega t)
    single-frequency     u(t) = A env(t) sin(omega t)          (raised-cosine switch-on)
The decaying command excites the plant only transiently: for t >> tau the regressor
collapses onto the free, lightly damped pendulum swing and the sliding-window excitation
Gramian loses rank. The single-frequency command drives the plant onto one periodic orbit,
whose regressor spans only a low-dimensional subspace of the feature space.

Every run starts near the hanging equilibrium (the upright one is open-loop unstable) with
the cart velocity chosen to cancel the secular drift the command would otherwise inject
into the free cart (it would run into the +/-0.5 m bumpers).
"""

from __future__ import annotations

from typing import Callable, List, Tuple
import numpy as np

from src.plant.pendulum_plant import PendulumParameters
from src.simulation.open_loop import OpenLoopTrajectory, simulate_open_loop

DT = 0.001
HANGING_STATE = np.array([0.0, 0.0, np.pi, 0.0])

# Operating-range normalization for zeta, as in Approach A: 1 / (expected half-range) of
# [x, theta, x_dot, theta_dot, u] = [0.33 m, 0.33 rad, 0.5 m/s, 1 rad/s, 2 N].
ZETA_INPUT_SCALE: Tuple[float, ...] = (3.0, 3.0, 2.0, 1.0, 0.5)

InputFn = Callable[[float], float]


def decaying_sine(amplitude: float = 3.0, omega: float = 1.6, tau: float = 8.0) -> InputFn:
    def u(t: float) -> float:
        return float(amplitude * np.exp(-t / tau) * np.sin(omega * t))
    return u


def single_sine(amplitude: float = 1.5, omega: float = 2.0, t_ramp: float = 2.0) -> InputFn:
    def u(t: float) -> float:
        env = 0.5 * (1.0 - np.cos(np.pi * min(t / t_ramp, 1.0))) if t_ramp > 0 else 1.0
        return float(amplitude * env * np.sin(omega * t))
    return u


def multisine(amplitudes: Tuple[float, ...] = (0.6, 0.4, 0.3), omegas: Tuple[float, ...] = (1.1, 2.9, 4.7),
              t_ramp: float = 2.0) -> InputFn:
    def u(t: float) -> float:
        env = 0.5 * (1.0 - np.cos(np.pi * min(t / t_ramp, 1.0))) if t_ramp > 0 else 1.0
        return float(env * sum(a * np.sin(w * t) for a, w in zip(amplitudes, omegas)))
    return u


def drift_cancelling_state(u_fn: InputFn, t_final: float, dt: float = DT,
                           params: PendulumParameters | None = None) -> Tuple[float, float]:
    """
    Initial cart (x0, v0) for the rigid-cart approximation x_ddot = u / (M + m):
    v0 gives zero mean velocity on [0, t_final] (no secular drift; for u = A sin(omega t),
    v0 = -A / ((M+m) omega)), and x0 centres the resulting excursion on the track.
    """
    p = params or PendulumParameters()
    t = np.arange(0.0, t_final, dt)
    u = np.array([u_fn(float(tk)) for tk in t])
    vel = np.cumsum(u) * dt / p.total_cart_mass
    v0 = -float(np.mean(vel))
    pos = np.cumsum(vel + v0) * dt
    return -0.5 * float(pos.max() + pos.min()), v0


def weak_excitation_command(kind: str) -> InputFn:
    if kind == "decaying":
        return decaying_sine()
    if kind == "single_freq":
        return single_sine()
    raise ValueError(f"Unknown weak-excitation scenario '{kind}' (use 'decaying' or 'single_freq').")


def simulate_weak_excitation(
    kind: str = "decaying",
    t_final: float = 60.0,
    noise_seed: int = 42,
    theta_offset: float = 0.3,
    dt: float = DT,
) -> Tuple[OpenLoopTrajectory, InputFn]:
    """Training trajectory under a non-PE command (encoder noise and quantization on)."""
    u_fn = weak_excitation_command(kind)
    x_c, v0 = drift_cancelling_state(u_fn, t_final, dt)
    x0 = np.array([x_c, v0, np.pi - theta_offset, 0.0])
    traj = simulate_open_loop(u_fn, t_final, dt, x0, apply_actuator_effects=False, seed=noise_seed)
    return traj, u_fn


def measurement_offset(traj: OpenLoopTrajectory, dt: float = DT) -> Tuple[float, ...]:
    """Data-driven centring of zeta from the first second of encoder data (velocities, u: 0)."""
    n0 = min(len(traj.t), int(round(1.0 / dt)))
    mean_meas = traj.measurements[:n0].mean(axis=0)
    return (float(mean_meas[0]), float(mean_meas[1]), 0.0, 0.0, 0.0)


def generalization_test_suite() -> List[Tuple[str, InputFn, float, np.ndarray]]:
    """
    Unseen open-loop inputs (name, u(t), horizon [s], initial state) near the hanging
    equilibrium: Approach A's suite (step doublet, chirp; from rest) plus a multisine and an
    off-frequency sine. The last two start from the drift-cancelling cart state so that the
    true cart stays off the bumpers, which no twin models.
    """
    from src.identification.extract_model import default_test_suite
    suite = [(name, u, horizon, HANGING_STATE.copy()) for name, u, horizon in default_test_suite()]
    for name, u, horizon in (
        ("multisine", multisine(), 12.0),
        ("sine_1.2rad/s", single_sine(amplitude=1.0, omega=1.2), 12.0),
    ):
        x0 = HANGING_STATE.copy()
        x0[0], x0[1] = drift_cancelling_state(u, horizon)
        suite.append((name, u, horizon, x0))
    return suite
