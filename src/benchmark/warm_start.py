"""
Improvement 3: model-free warm start of any observer.

Every observer starts from q_hat(0) = y(0), q_dot_hat(0) = 0. When the plant is already moving
at t = 0 (the drift-cancelling cart velocity of scenarios S1-S3, for example), the observer must
first absorb that velocity error through its filter. The wrapper buffers the first `window`
seconds of encoder data, fits a degree-`order` polynomial by least squares and evaluates it and
its derivative at the LAST sample (causal Savitzky-Golay end-point estimate); the wrapped
observer is then reset to [q_fit, q_dot_fit] and runs normally from the next sample on.

Significance gate. A coordinate that starts at rest gains nothing from the fit and would only
receive its noise (sigma_v = sigma_q ||c_1||, with sigma_q estimated from the fit residual). Each
fitted velocity is therefore kept only if |q_dot_fit| > z sigma_v (default z = 3) and set to 0
otherwise, the value an unwrapped observer starts from. During
the buffering window the wrapper reports [y, 0], exactly what an unwrapped observer starts from.

The wrapper touches neither the observer's dynamics nor its adaptation, so every stability
argument of the wrapped observer applies from the reset on.
"""

from __future__ import annotations

from typing import List
import numpy as np

from src.benchmark.models import Observer
from src.concurrent_learning.savitzky_golay import SavitzkyGolay


class WarmStartObserver:
    def __init__(self, observer: Observer, dt: float, window: float = 0.1, order: int = 2, z_gate: float = 3.0) -> None:
        self.inner = observer
        self.z_gate = z_gate
        self.sg = SavitzkyGolay.from_duration(window, order, dt, evaluate_at="end")
        self._buf: List[np.ndarray] = []
        self.started = False
        self.initial_estimate: np.ndarray | None = None

    def reset(self, initial_state: np.ndarray | None = None, reset_weights: bool = True) -> None:
        self._buf = []
        self.started = False

    def update(self, y: np.ndarray, u: float | np.ndarray, dt: float) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if self.started:
            return self.inner.update(y, u, dt)
        self._buf.append(y.copy())
        if len(self._buf) < self.sg.window:
            return np.column_stack((y, np.zeros_like(y))).ravel()
        Y = np.vstack(self._buf)
        q, qd = self.sg.apply(Y, 0), self.sg.apply(Y, 1)
        # residual of the fitted polynomial -> measurement-noise estimate per coordinate
        tau = (np.arange(self.sg.window) - (self.sg.window - 1)) * self.sg.dt
        V = np.vander(tau, self.sg.order + 1, increasing=True)
        coef = np.linalg.lstsq(V, Y, rcond=None)[0]
        dof = max(self.sg.window - self.sg.order - 1, 1)
        sigma_q = np.sqrt(np.sum((Y - V @ coef) ** 2, axis=0) / dof)
        self.sigma_v = sigma_q * self.sg.noise_gain(1)
        qd = np.where(np.abs(qd) > self.z_gate * self.sigma_v, qd, 0.0)
        self.initial_estimate = np.column_stack((q, qd)).ravel()
        self.inner.reset(self.initial_estimate)
        self.started = True
        return self.initial_estimate.copy()
