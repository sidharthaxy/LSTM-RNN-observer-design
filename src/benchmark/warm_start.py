"""
Improvement 3: model-free velocity injection for any observer ("warm start").

Every observer starts from q_hat(0) = y(0), q_dot_hat(0) = 0. When a coordinate is already moving
at t = 0 (the drift-cancelling cart velocity of scenarios S1-S3), the observer must first absorb
that velocity error through its filter.

The wrapper runs the observer normally from the first sample and, in parallel, buffers the first
`window` seconds of encoder data. At the end of the window it fits a degree-`order` polynomial by
least squares and differentiates it at the last sample (causal Savitzky-Golay end-point estimate).
For each coordinate the fitted position and velocity are injected into the observer's estimate only
if the velocity is significant, |q_dot_fit| > z sigma_v, where sigma_v = sigma_q ||c_1|| and sigma_q is estimated
from the fit residual. For an injected coordinate the auxiliary filter (p, nu), which by then
encodes the velocity error the observer has been absorbing, is re-initialized exactly as at
start-up (p = (alpha + k_r) q_tilde, nu = 0, i.e. eta = 0), so it does not keep "correcting" an
error that the injection removed. Coordinates that fail the test are left exactly as the observer
has them; LSTM memories and weights are never touched.

Design history (shared-benchmark README, Section 5): resetting the whole observer at the end of the
window, with or without the significance test, helped the moving coordinate but made the resting
one worse: the reset discarded the observer's first `window` seconds of adaptation and restarted
its filter transient. Injecting only the significant velocities leaves every other observer state,
filter state and weight untouched.
"""

from __future__ import annotations

from typing import List
import numpy as np

from src.benchmark.models import Observer
from src.concurrent_learning.savitzky_golay import SavitzkyGolay


def _state_attributes(observer: Observer) -> tuple[str, str, str]:
    """(position estimate, velocity estimate, filter state p) attribute names."""
    if hasattr(observer, "x2_hat"):                 # A / C (LbLSTMObserver)
        return "x1_hat", "x2_hat", "p"
    if hasattr(observer, "p_f"):                    # D (PIICLObserver)
        return "q_hat", "q_dot_hat", "p_f"
    if hasattr(observer, "q_dot_hat"):              # B (PILSTMObserver)
        return "q_hat", "q_dot_hat", "p"
    raise TypeError(f"{type(observer).__name__} exposes no velocity estimate.")


class WarmStartObserver:
    def __init__(self, observer: Observer, dt: float, window: float = 0.1, order: int = 2, z_gate: float = 3.0) -> None:
        self.inner = observer
        self.z_gate = z_gate
        self.sg = SavitzkyGolay.from_duration(window, order, dt, evaluate_at="end")
        self._pos_attr, self._vel_attr, self._p_attr = _state_attributes(observer)
        self._buf: List[np.ndarray] = []
        self.started = False
        self.initial_estimate: np.ndarray | None = None    # fitted [q, q_dot] (interleaved)
        self.injected: np.ndarray | None = None            # per coordinate: velocity injected?
        self.sigma_v: np.ndarray | None = None

    def reset(self, initial_state: np.ndarray | None = None, reset_weights: bool = True) -> None:
        self.inner.reset(initial_state, reset_weights)
        self._buf = []
        self.started = False

    def update(self, y: np.ndarray, u: float | np.ndarray, dt: float) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        est = self.inner.update(y, u, dt)
        if self.started:
            return est
        self._buf.append(y.copy())
        if len(self._buf) < self.sg.window:
            return est
        Y = np.vstack(self._buf)
        q, qd = self.sg.apply(Y, 0), self.sg.apply(Y, 1)
        tau = (np.arange(self.sg.window) - (self.sg.window - 1)) * self.sg.dt
        V = np.vander(tau, self.sg.order + 1, increasing=True)
        coef = np.linalg.lstsq(V, Y, rcond=None)[0]
        dof = max(self.sg.window - self.sg.order - 1, 1)
        sigma_q = np.sqrt(np.sum((Y - V @ coef) ** 2, axis=0) / dof)
        sigma_v = sigma_q * self.sg.noise_gain(1)
        inj = np.abs(qd) > self.z_gate * sigma_v
        self.sigma_v, self.injected = sigma_v, inj
        self.initial_estimate = np.column_stack((q, qd)).ravel()
        obs = self.inner
        getattr(obs, self._pos_attr)[inj] = q[inj]      # the lagging position estimate as well
        getattr(obs, self._vel_attr)[inj] = qd[inj]
        cfg = obs.config  # type: ignore[attr-defined]
        q_tilde = y - getattr(obs, self._pos_attr)
        getattr(obs, self._p_attr)[inj] = (cfg.alpha + cfg.k_r) * q_tilde[inj]
        obs.nu[inj] = 0.0  # type: ignore[attr-defined]
        self.started = True
        return self.inner.state_estimate()  # type: ignore[attr-defined]
