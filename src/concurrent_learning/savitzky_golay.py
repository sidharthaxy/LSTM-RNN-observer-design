"""
Causal Savitzky-Golay smoothing and differentiation (Savitzky & Golay, 1964).

A degree-P polynomial is least-squares fitted to the last W (odd) samples,
tau_k = (k - c) T_s, k = 0..W-1, and evaluated with its derivatives at
    c = (W-1)/2   ("centre": delayed by (W-1)/2 samples, zero phase lag, exact for degree <= P), or
    c = W-1       ("end":    no delay, larger noise gain).
With the rows of pinv(V), V_ki = tau_k^i, the i-th derivative estimate is i! [pinv(V)]_i . y.
For i.i.d. noise of standard deviation sigma the i-th estimate has standard deviation
sigma * ||c_i|| (`noise_gain(i)`).
"""

from __future__ import annotations

from typing import Literal
import numpy as np


class SavitzkyGolay:
    def __init__(self, window: int, order: int, dt: float, evaluate_at: Literal["centre", "end"] = "centre") -> None:
        if window % 2 == 0:
            window += 1
        if order < 1 or window <= order + 1:
            raise ValueError("Need order >= 1 and window > order + 1.")
        self.window = window
        self.order = order
        self.dt = dt
        self.delay = (window - 1) // 2 if evaluate_at == "centre" else 0
        c = (window - 1) // 2 if evaluate_at == "centre" else window - 1
        tau = (np.arange(window) - c) * dt
        pinv = np.linalg.pinv(np.vander(tau, order + 1, increasing=True))
        fact = np.cumprod(np.concatenate(([1.0], np.arange(1, order + 1))))
        self.coeffs = pinv * fact[:, None]            # row i: i-th derivative filter

    @classmethod
    def from_duration(cls, seconds: float, order: int, dt: float, evaluate_at: Literal["centre", "end"] = "centre") -> SavitzkyGolay:
        return cls(round(seconds / dt) | 1, order, dt, evaluate_at)

    def noise_gain(self, deriv: int) -> float:
        return float(np.linalg.norm(self.coeffs[deriv]))

    def apply(self, samples: np.ndarray, deriv: int) -> np.ndarray:
        """samples: (W, n) oldest first -> (n,) estimate of the deriv-th derivative."""
        Y = np.asarray(samples, dtype=np.float64)
        if Y.shape[0] != self.window:
            raise ValueError(f"Expected {self.window} samples, got {Y.shape[0]}.")
        return self.coeffs[deriv] @ Y
