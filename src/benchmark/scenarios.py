"""
Shared benchmark scenarios: every approach's home training scenario, plus a held-out test set.

Training scenarios (each is the validation scenario of one branch, reproduced exactly):
    S1  "hanging, two-sine"        Approach A: u = 2 sin 1.5t + 1.2 cos 3t, theta(0) = pi - 0.4,
                                   drift-cancelling cart velocity, U(+-0.5 deg), 50 s.
    S2  "swing-through + mass step" Approach B: theta(0) = 0.5 rad from upright, u = 2 sin 3t + 1.2 cos 6t,
                                   U(+-1 deg), pendulum mass and inertia +50 % at 25 s, 50 s.
    S3  "weak excitation"          Approach C: u = 3 exp(-t/8) sin 1.6t (non-PE), centred cart,
                                   theta(0) = pi - 0.3, U(+-0.5 deg), 60 s.

Held-out test inputs (defined for this benchmark, never used to tune any branch):
    H1 filtered square wave, H2 random-phase multisine on frequencies absent from all training
    inputs, H3 slow chirp 0.1 -> 0.6 Hz, H4 pulse train. Each starts at rest at the hanging
    equilibrium with the drift-cancelling cart state (the true cart stays off the bumpers), on the
    plant the model ended its training on (post-shift parameters for S2).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Tuple
import numpy as np

from src.plant.pendulum_plant import PendulumParameters, PendulumPlant
from src.plant.sensor_noise import SensorNoiseConfig, SensorNoiseModel

DT = 0.001
InputFn = Callable[[float], float]


@dataclass
class Dataset:
    """One noisy training (or test) run of the rig."""
    name: str
    t: np.ndarray               # (N,)
    state: np.ndarray           # (N, 4) true [x, x_dot, theta, theta_dot]
    accel: np.ndarray           # (N, 2) true accelerations
    meas: np.ndarray            # (N, 2) encoder [x, theta]
    u: np.ndarray               # (N,) commanded force (= applied: actuator effects off, as in all branches)
    params_final: PendulumParameters
    shift_time: float | None = None

    @property
    def velocities(self) -> np.ndarray:
        return self.state[:, [1, 3]]

    @property
    def positions(self) -> np.ndarray:
        return self.state[:, [0, 2]]


def simulate(
    name: str, u_fn: InputFn, t_final: float, x0: np.ndarray, noise_deg: float, noise_seed: int,
    params: PendulumParameters | None = None, shift: Tuple[float, PendulumParameters] | None = None,
    noise: bool = True,
) -> Dataset:
    p = params or PendulumParameters()
    plant = PendulumPlant(p)
    plant.reset(np.asarray(x0, dtype=np.float64))
    cfg = SensorNoiseConfig(theta_noise_deg=noise_deg) if noise else SensorNoiseConfig(theta_noise_deg=0.0, x_noise_mm=0.0)
    sensor = SensorNoiseModel(cfg)
    sensor.seed(noise_seed)
    t = np.arange(0.0, t_final, DT)
    N = t.size
    state, accel, meas, u = np.zeros((N, 4)), np.zeros((N, 2)), np.zeros((N, 2)), np.zeros(N)
    shifted = False
    for k, tk in enumerate(t):
        if shift is not None and not shifted and tk >= shift[0] - 1e-9:
            plant.params = shift[1]
            shifted = True
        s = plant.state.to_array()
        state[k] = s
        meas[k] = sensor.measure(s[0], s[2])
        u[k] = u_fn(float(tk))
        accel[k] = plant.forward_dynamics(s, u[k])
        plant.step(u[k], DT)
    return Dataset(name, t, state, accel, meas, u, plant.params, None if shift is None else shift[0])


# ---------------------------------------------------------------------- training scenarios
def drift_cancelling_state(u_fn: InputFn, t_final: float, p: PendulumParameters | None = None) -> Tuple[float, float]:
    """(x0, v0): zero-mean rigid-cart velocity and a centred excursion (Approach C's helper)."""
    p = p or PendulumParameters()
    t = np.arange(0.0, t_final, DT)
    vel = np.cumsum([u_fn(float(tk)) for tk in t]) * DT / p.total_cart_mass
    v0 = -float(np.mean(vel))
    pos = np.cumsum(vel + v0) * DT
    return -0.5 * float(pos.max() + pos.min()), v0


def s1_hanging_two_sine(seed: int) -> Dataset:
    p = PendulumParameters()
    u = lambda t: 2.0 * np.sin(1.5 * t) + 1.2 * np.cos(3.0 * t)
    x0 = np.array([0.0, -2.0 / (1.5 * p.total_cart_mass), np.pi - 0.4, 0.0])
    return simulate("S1 hanging, two-sine", u, 50.0, x0, 0.5, 42 + seed)


def s2_swing_mass_step(seed: int) -> Dataset:
    p = PendulumParameters()
    u = lambda t: 2.0 * np.sin(3.0 * t) + 1.2 * np.cos(6.0 * t)
    x0 = np.array([0.0, -2.0 / (3.0 * p.total_cart_mass), 0.5, 0.0])
    p_after = replace(p, m=1.5 * p.m, I=1.5 * p.I)
    return simulate("S2 swing-through + mass step", u, 50.0, x0, 1.0, 42 + seed, shift=(25.0, p_after))


def s3_weak_excitation(seed: int) -> Dataset:
    u = lambda t: 3.0 * np.exp(-t / 8.0) * np.sin(1.6 * t)
    x_c, v0 = drift_cancelling_state(u, 60.0)
    return simulate("S3 weak excitation", u, 60.0, np.array([x_c, v0, np.pi - 0.3, 0.0]), 0.5, 42 + seed)


TRAINING: Dict[str, Callable[[int], Dataset]] = {
    "S1": s1_hanging_two_sine,
    "S2": s2_swing_mass_step,
    "S3": s3_weak_excitation,
}
STEADY_WINDOW: Dict[str, Tuple[float, float]] = {"S1": (35.0, 50.0), "S2": (35.0, 50.0), "S3": (45.0, 60.0)}


# ---------------------------------------------------------------------- held-out tests
def _ramp(t: float, t_ramp: float = 1.0) -> float:
    return 0.5 * (1.0 - np.cos(np.pi * min(t / t_ramp, 1.0)))


def h1_filtered_square(t: float) -> float:
    """+-0.8 N square wave, period 4 s, through a first-order lag (tau = 0.3 s), in closed form."""
    period, tau, A = 4.0, 0.3, 0.8
    k = int(t // (period / 2))
    level = A if k % 2 == 0 else -A
    prev = -level if k > 0 else 0.0
    return float(level + (prev - level) * np.exp(-(t - k * period / 2) / tau))


_H2_RNG = np.random.default_rng(2026)
_H2_W = np.array([0.7, 2.2, 3.7, 5.3])
_H2_PH = _H2_RNG.uniform(0.0, 2.0 * np.pi, 4)


def h2_multisine(t: float) -> float:
    return float(_ramp(t) * 0.35 * np.sum(np.sin(_H2_W * t + _H2_PH)))


def h3_slow_chirp(t: float) -> float:
    f0, f1, T = 0.1, 0.6, 12.0
    return float(_ramp(t) * 1.0 * np.sin(2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t * t / T)))


def h4_pulse_train(t: float) -> float:
    ph = t % 2.0
    return 1.0 if 0.2 <= ph < 0.5 else (-1.0 if 1.2 <= ph < 1.5 else 0.0)


HELD_OUT: Dict[str, InputFn] = {
    "H1 filtered square": h1_filtered_square,
    "H2 multisine": h2_multisine,
    "H3 slow chirp": h3_slow_chirp,
    "H4 pulse train": h4_pulse_train,
}
HELD_OUT_HORIZON = 12.0


def held_out_tests(params: PendulumParameters) -> List[Dataset]:
    """Noise-free truth for the held-out inputs on the given plant (scoring is against the truth)."""
    out = []
    for name, u in HELD_OUT.items():
        x_c, v0 = drift_cancelling_state(u, HELD_OUT_HORIZON, params)
        out.append(simulate(name, u, HELD_OUT_HORIZON, np.array([x_c, v0, np.pi, 0.0]), 0.0, 0,
                            params=params, noise=False))
    return out
