"""
Model registry and frozen-model ("twin") adapters for the shared benchmark.

Every approach runs with its own branch defaults. The black-box models (A, C) get the same
data-driven input normalization Approach B's benchmark gave Approach A: offset and scale from
the first 5 s of encoder data, velocity scale from a dirty derivative, input scale from max |u|.

    A      black-box Lb-LSTM (Approach A)
    A-x    A with the cart position removed from its input (scale 0)       <- improvement 2
    B      physics-informed PI-LSTM (Approach B)
    C      concurrent-learning Lb-LSTM (Approach C)
    C-x    C with the cart position removed from its input                  <- improvement 2
    D      physics-structured integral-CL observer (Approach D)
    E      hybrid of B and D (Approach E): D's model and integral CL, blended by information
           with B's instantaneous law

Improvement 2 rationale: the rig's dynamics do not depend on the cart position x (x is a cyclic
coordinate). B and D build that in; A and C see x as an input and must extrapolate whenever a
test visits cart positions the training did not.

Twins. After training, each model is frozen and decoupled from its observer. A common interface
evaluates it on any trajectory:
    evaluate(q, q_dot, u, mem) -> (acceleration, d mem / dt)
with `mem` the model's internal memory (LSTM cell / hidden states; empty for D).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Protocol, Tuple
import numpy as np

from src.baselines.classical_estimators import DirtyDerivativeFilter
from src.benchmark.scenarios import DT, Dataset
from src.identification.extract_model import FrozenLbLSTM, LbLSTMDigitalTwin
from src.observers.blackbox_lstm import LbLSTMObserver, LbLSTMObserverConfig
from src.observers.cl_lstm_observer import CLLbLSTMObserver, CLLbLSTMObserverConfig
from src.observers.hybrid_observer import HybridObserver, HybridObserverConfig
from src.observers.pi_icl_observer import PIICLObserver, PIICLObserverConfig
from src.observers.pilstm_observer import PILSTMObserver, PILSTMObserverConfig

MODELS = ("A", "A-x", "B", "C", "C-x", "D", "E")
NORM_WINDOW = 5.0


@dataclass(frozen=True)
class Normalization:
    pos_offset: Tuple[float, float]
    pos_scale: Tuple[float, float]
    vel_scale: Tuple[float, float]
    u_scale: float


def normalization(data: Dataset) -> Normalization:
    """Approach B's data-driven normalization (first NORM_WINDOW seconds of encoder data)."""
    n0 = min(len(data.t), round(NORM_WINDOW / DT))
    y = data.meas[:n0]
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
        (float(1 / vmax[0]), float(1 / vmax[1])), float(1 / max(np.abs(data.u[:n0]).max(), 1e-3)),
    )


class Observer(Protocol):
    def reset(self, initial_state: np.ndarray | None = None, reset_weights: bool = True) -> None: ...
    def update(self, y: np.ndarray, u: float | np.ndarray, dt: float) -> np.ndarray: ...


def _lstm_scales(norm: Normalization, drop_x: bool) -> Dict[str, Any]:
    scale = [*norm.pos_scale, *norm.vel_scale, norm.u_scale]
    if drop_x:
        scale[0] = 0.0
    return {"input_scale": tuple(scale), "input_offset": (*norm.pos_offset, 0.0, 0.0, 0.0)}


def make_observer(name: str, data: Dataset, seed: int) -> Observer:
    norm = normalization(data)
    if name in ("A", "A-x"):
        return LbLSTMObserver(LbLSTMObserverConfig(seed=seed, **_lstm_scales(norm, name == "A-x")))
    if name in ("C", "C-x"):
        return CLLbLSTMObserver(CLLbLSTMObserverConfig(seed=seed, **_lstm_scales(norm, name == "C-x")))
    if name == "B":
        return PILSTMObserver(PILSTMObserverConfig(velocity_scale=norm.vel_scale, seed=seed))
    if name == "D":
        return PIICLObserver(PIICLObserverConfig(seed=seed))
    if name == "E":
        return HybridObserver(HybridObserverConfig(seed=seed))
    raise KeyError(name)


# ---------------------------------------------------------------------- twins
class Twin(Protocol):
    n_mem: int
    def evaluate(self, q: np.ndarray, q_dot: np.ndarray, u: np.ndarray, mem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...


class LSTMTwin:
    """Frozen A / C model (Approach A's digital twin)."""

    def __init__(self, obs: LbLSTMObserver) -> None:
        cfg = obs.config
        base = LbLSTMObserverConfig(**{k: getattr(cfg, k) for k in LbLSTMObserverConfig.__dataclass_fields__})
        self.twin = LbLSTMDigitalTwin(FrozenLbLSTM(np.copy(obs.theta), base, 0.0))
        self.L = cfg.hidden_dim
        self.n_mem = 2 * self.L
        self.b_c, self.b_h = cfg.b_c, cfg.b_h

    def evaluate(self, q: np.ndarray, q_dot: np.ndarray, u: np.ndarray, mem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c_hat, h_hat = mem[: self.L], mem[self.L:]
        cache = self.twin._lstm(q, q_dot, u, c_hat, h_hat)
        return cache.phi, np.concatenate((self.b_c * (cache.c - c_hat), self.b_h * (cache.h - h_hat)))


class PILSTMTwin:
    """Frozen B model: structured network with its friction-LSTM memory."""

    def __init__(self, obs: PILSTMObserver) -> None:
        self.net, self.theta = obs.net, np.copy(obs.theta)
        self.L = obs.config.friction_hidden
        self.n_mem = 2 * self.L
        self.b_c, self.b_h = obs.config.b_c, obs.config.b_h

    def evaluate(self, q: np.ndarray, q_dot: np.ndarray, u: np.ndarray, mem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c_hat, h_hat = mem[: self.L], mem[self.L:]
        cache = self.net.forward(self.theta, q, q_dot, u, c_hat, h_hat)
        fr = cache.friction
        return cache.phi, np.concatenate((self.b_c * (fr.c - c_hat), self.b_h * (fr.h - h_hat)))


class ELTwin:
    """Frozen D model (memoryless)."""

    def __init__(self, obs: PIICLObserver) -> None:
        self.model, self.theta = obs.model, np.copy(obs.theta)
        self.n_mem = 0

    def evaluate(self, q: np.ndarray, q_dot: np.ndarray, u: np.ndarray, mem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.model.acceleration(self.theta, q, q_dot, u)[0], mem


def make_twin(obs: Observer) -> Twin:
    if isinstance(obs, HybridObserver):          # before PIICLObserver: the hybrid subclasses it
        return obs.frozen_model()
    if isinstance(obs, PIICLObserver):
        return ELTwin(obs)
    if isinstance(obs, PILSTMObserver):
        return PILSTMTwin(obs)
    if isinstance(obs, LbLSTMObserver):
        return LSTMTwin(obs)
    raise TypeError(type(obs))


# ---------------------------------------------------------------------- twin evaluation
def settle_memory(twin: Twin, q: np.ndarray, q_dot: np.ndarray, u: np.ndarray, t_settle: float = 2.0) -> np.ndarray:
    mem = np.zeros(twin.n_mem)
    for _ in range(round(t_settle / DT) if twin.n_mem else 0):
        mem = mem + DT * twin.evaluate(q, q_dot, u, mem)[1]
    return mem


def teacher_forced(twin: Twin, data: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """(accelerations (N, 2), memory (N, n_mem)) of the twin evaluated on the true states."""
    N = len(data.t)
    q, qd, u = data.positions, data.velocities, data.u[:, None]
    mem = settle_memory(twin, q[0], qd[0], u[0])
    acc, mems = np.zeros((N, 2)), np.zeros((N, twin.n_mem))
    for k in range(N):
        mems[k] = mem
        a, dm = twin.evaluate(q[k], qd[k], u[k], mem)
        acc[k] = a
        mem = mem + DT * dm
    return acc, mems


def short_horizon(twin: Twin, data: Dataset, mems: np.ndarray, horizon: float = 0.5) -> np.ndarray:
    """Open-loop RK4 predictions restarted from the true state (and teacher-forced memory) every `horizon` s."""
    N, H = len(data.t), max(1, round(horizon / DT))
    u = data.u[:, None]
    pred = np.zeros((N, 4))

    def f(z: np.ndarray, uk: np.ndarray) -> np.ndarray:
        a, dm = twin.evaluate(z[:2], z[2:4], uk, z[4:])
        return np.concatenate((z[2:4], a, dm))

    for k0 in range(0, N, H):
        z = np.concatenate((data.positions[k0], data.velocities[k0], mems[k0]))
        for k in range(k0, min(k0 + H, N)):
            pred[k] = z[[0, 2, 1, 3]]
            k1 = f(z, u[k])
            k2 = f(z + 0.5 * DT * k1, u[k])
            k3 = f(z + 0.5 * DT * k2, u[k])
            k4 = f(z + DT * k3, u[k])
            z = z + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return pred


def nmse(truth: np.ndarray, pred: np.ndarray) -> float:
    var = float(np.var(truth))
    return float(np.mean((truth - pred) ** 2)) / var if var > 1e-12 else float("nan")


def twin_metrics(twin: Twin, data: Dataset) -> Dict[str, float]:
    acc, mems = teacher_forced(twin, data)
    pred = short_horizon(twin, data, mems)
    return {
        "one-step NMSE x_ddot": nmse(data.accel[:, 0], acc[:, 0]),
        "one-step NMSE theta_ddot": nmse(data.accel[:, 1], acc[:, 1]),
        "0.5 s NMSE x": nmse(data.state[:, 0], pred[:, 0]),
        "0.5 s NMSE theta": nmse(data.state[:, 2], pred[:, 2]),
    }
