"""
Approach A: Pure black-box continuous-time Lb-LSTM adaptive state observer
(Griffis, Patil, Hart & Dixon, IEEE Control Systems Letters, 2024).

Plant class (n generalized coordinates, m inputs, only x1 measured):
    x1_dot = x2,   x2_dot = g(x, u)        g unknown: no mass / inertia / damping knowledge.

Continuous-time LSTM (Eqs. 7-8), zeta = [x_hat1^T, x_hat2^T, u^T, h_hat^T, 1]^T:
    f = sigma_g(W_f^T zeta),  i = sigma_g(W_i^T zeta),  c* = sigma_c(W_c^T zeta),  o = sigma_g(W_o^T zeta)
    c_hat_dot = -b_c c_hat + b_c (f * c_hat + i * c*)
    h_hat_dot = -b_h h_hat + b_h (o * sigma_c(c)),      c = f * c_hat + i * c*
    Phi_hat   = W_h^T (o * sigma_c(c))
c_hat / h_hat are the continuous memory states (low-pass filtered cell and hidden state);
c / h are the instantaneous cell and hidden outputs of the gate network.

Dynamic auxiliary filter (Eqs. 3-6), x_tilde1 = y - x_hat1 (measurable):
    eta    = p - (alpha + k_r) x_tilde1
    p_dot  = -(k_r + 2 alpha) p - nu + ((alpha + k_r)^2 + 1) x_tilde1
    nu_dot = p - alpha nu - (alpha + k_r) x_tilde1
    e      = x_tilde1 + nu

Observer (Eq. 11):
    chi         = -(3 alpha + k_r) eta + (2 - alpha^2) x_tilde1 - nu
    x_hat1_dot  = x_hat2
    x_hat2_dot  = Phi_hat + k_s sgn(e) + chi          (or k_s tanh(e / eps_tanh))

The x_tilde1 coefficient (2 - alpha^2) is the unique value for which the cross terms of
V = 1/2 (x_tilde1^T x_tilde1 + eta^T eta + nu^T nu + r^T r), r = x_tilde2 + alpha x_tilde1 + eta,
cancel exactly (see README, Section 3). `chi_x1_coeff` overrides it.

Adaptation: theta_hat_dot = proj(Gamma Phi'^T e), ||theta_hat|| <= W_bar (src/adaptation).
Integration: explicit (forward) Euler at the sampling period, optionally sub-stepped with
the measurement and input held constant (zero-order hold) over the sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np

from src.adaptation.jacobian_engine import (
    LSTMCache,
    LSTMWeightLayout,
    LyapunovAdaptationLaw,
    lstm_forward,
)


@dataclass
class LbLSTMObserverConfig:
    """Gains and architecture of the black-box Lb-LSTM observer."""
    n_coords: int = 2              # n: dimension of x1 (measured generalized coordinates)
    n_inputs: int = 1              # m: dimension of u
    hidden_dim: int = 16           # L: LSTM cell width

    # Auxiliary filter and robust feedback
    alpha: float = 4.0             # Filter pole / e-dynamics rate (e_dot = -alpha e + r)
    k_r: float = 8.0               # Filtered-error damping gain
    k_s: float = 0.2               # Robust sliding gain (must dominate the NN residual)
    sign_mode: Literal["sgn", "tanh"] = "sgn"
    tanh_eps: float = 0.01         # Boundary-layer width for sign_mode="tanh"
    chi_x1_coeff: float | None = None  # None -> 2 - alpha^2 (exact cancellation)

    # LSTM memory time constants [1/s]
    b_c: float = 5.0
    b_h: float = 5.0

    # Lyapunov adaptation
    gamma_gates: float = 30.0      # Gamma on vec(W_c), vec(W_i), vec(W_f), vec(W_o)
    gamma_out: float = 300.0       # Gamma on vec(W_h)
    w_bar: float = 40.0            # Compact ball radius ||theta_hat|| <= W_bar
    proj_eps: float = 0.1          # Projection boundary-layer width (relative)
    adapt: bool = True             # False -> weights frozen

    # Input preconditioning zeta = [s * ([x_hat1, x_hat2, u] - mu), h_hat, 1] with per-channel
    # scale s and offset mu (length 2n + m each). This is data normalization (e.g. z-scoring
    # from a short window of measurements), not model knowledge: it keeps the gate
    # pre-activations out of saturation for signals of different units and operating points.
    input_scale: Tuple[float, ...] | None = None
    input_offset: Tuple[float, ...] | None = None

    init_scale: float = 0.5        # Std of the random gate-weight initialization
    n_substeps: int = 1            # Euler sub-steps per sample
    seed: int = 0


@dataclass
class ObserverDerivatives:
    """Time derivatives of all observer states plus the signals they were built from."""
    x1_hat: np.ndarray
    x2_hat: np.ndarray
    p: np.ndarray
    nu: np.ndarray
    c_hat: np.ndarray
    h_hat: np.ndarray
    theta: np.ndarray | None
    e: np.ndarray
    eta: np.ndarray
    cache: LSTMCache


def _interleave(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """[x1_0, x2_0, x1_1, x2_1, ...] -> plant ordering [x, x_dot, theta, theta_dot] for n = 2."""
    return np.column_stack((x1, x2)).ravel()


class LbLSTMObserver:
    """
    Black-box continuous-time Lb-LSTM observer. Compatible with
    `run_benchmark_comparison(custom_observers=...)`: `reset(x0)` and `update(y, u, dt)`
    use the interleaved state ordering [x1_0, x2_0, x1_1, x2_1, ...].
    """

    def __init__(self, config: LbLSTMObserverConfig | None = None) -> None:
        self.config = config or LbLSTMObserverConfig()
        cfg = self.config
        n, m, L = cfg.n_coords, cfg.n_inputs, cfg.hidden_dim

        self.zeta_dim = 2 * n + m + L + 1
        self.layout = LSTMWeightLayout(self.zeta_dim, L, n)
        self.adaptation = LyapunovAdaptationLaw(
            self.layout, cfg.gamma_gates, cfg.gamma_out, cfg.w_bar, cfg.proj_eps
        )

        if cfg.input_scale is None:
            self.input_scale = np.ones(2 * n + m, dtype=np.float64)
        else:
            self.input_scale = np.asarray(cfg.input_scale, dtype=np.float64)
            if self.input_scale.shape != (2 * n + m,):
                raise ValueError(f"input_scale must have length 2n + m = {2 * n + m}.")

        if cfg.input_offset is None:
            self.input_offset = np.zeros(2 * n + m, dtype=np.float64)
        else:
            self.input_offset = np.asarray(cfg.input_offset, dtype=np.float64)
            if self.input_offset.shape != (2 * n + m,):
                raise ValueError(f"input_offset must have length 2n + m = {2 * n + m}.")

        self.chi_x1_coeff = (
            2.0 - cfg.alpha ** 2 if cfg.chi_x1_coeff is None else float(cfg.chi_x1_coeff)
        )

        # Initial weights: random gates (so c* and h are not identically zero and the
        # gradient Phi'^T e is non-degenerate), zero readout (Phi_hat(0) = 0, no prior model).
        rng = np.random.default_rng(cfg.seed)
        self.theta0 = np.zeros(self.layout.n_params, dtype=np.float64)
        self.theta0[: self.layout.n_gate_params] = rng.normal(
            0.0, cfg.init_scale, size=self.layout.n_gate_params
        )
        self.adaptation.enforce_bound(self.theta0)

        self.theta = np.copy(self.theta0)
        self.x1_hat = np.zeros(n, dtype=np.float64)
        self.x2_hat = np.zeros(n, dtype=np.float64)
        self.p = np.zeros(n, dtype=np.float64)
        self.nu = np.zeros(n, dtype=np.float64)
        self.c_hat = np.zeros(L, dtype=np.float64)
        self.h_hat = np.zeros(L, dtype=np.float64)

        self.phi_hat = np.zeros(n, dtype=np.float64)
        self.e = np.zeros(n, dtype=np.float64)
        self._filter_initialized = False

    # ------------------------------------------------------------------ state handling
    def reset(self, initial_state: np.ndarray | None = None, reset_weights: bool = True) -> None:
        """Resets observer states; `initial_state` uses the interleaved ordering."""
        n = self.config.n_coords
        if initial_state is None:
            self.x1_hat = np.zeros(n, dtype=np.float64)
            self.x2_hat = np.zeros(n, dtype=np.float64)
        else:
            x0 = np.asarray(initial_state, dtype=np.float64).reshape(n, 2)
            self.x1_hat = np.copy(x0[:, 0])
            self.x2_hat = np.copy(x0[:, 1])
        self.p.fill(0.0)
        self.nu.fill(0.0)
        self.c_hat.fill(0.0)
        self.h_hat.fill(0.0)
        self.phi_hat.fill(0.0)
        self.e.fill(0.0)
        self._filter_initialized = False
        if reset_weights:
            self.theta = np.copy(self.theta0)

    def state_estimate(self) -> np.ndarray:
        """Interleaved estimate [x_hat1_0, x_hat2_0, x_hat1_1, x_hat2_1, ...]."""
        return _interleave(self.x1_hat, self.x2_hat)

    @property
    def theta_norm(self) -> float:
        return float(np.linalg.norm(self.theta))

    # ------------------------------------------------------------------ dynamics
    def build_zeta(self, x1_hat: np.ndarray, x2_hat: np.ndarray, u: np.ndarray, h_hat: np.ndarray) -> np.ndarray:
        """zeta = [s * ([x_hat1, x_hat2, u] - mu), h_hat, 1]."""
        signals = (np.concatenate((x1_hat, x2_hat, u)) - self.input_offset) * self.input_scale
        return np.concatenate((signals, h_hat, (1.0,)))

    def robust_term(self, e: np.ndarray) -> np.ndarray:
        cfg = self.config
        if cfg.sign_mode == "tanh":
            return cfg.k_s * np.tanh(e / cfg.tanh_eps)
        return cfg.k_s * np.sign(e)

    def lstm(self, u: np.ndarray) -> LSTMCache:
        """Evaluates the LSTM at the current observer state."""
        zeta = self.build_zeta(self.x1_hat, self.x2_hat, u, self.h_hat)
        return lstm_forward(self.theta, zeta, self.c_hat, self.layout)

    def derivatives(self, y: np.ndarray, u: np.ndarray) -> ObserverDerivatives:
        """Continuous-time vector field of every observer state at the current state."""
        cfg = self.config
        a, k_r = cfg.alpha, cfg.k_r

        # Measurable errors and auxiliary filter outputs (Eqs. 3-6)
        x1_tilde = y - self.x1_hat
        eta = self.p - (a + k_r) * x1_tilde
        e = x1_tilde + self.nu

        # LSTM (Eqs. 7-8)
        cache = self.lstm(u)

        # Observer (Eq. 11)
        chi = -(3.0 * a + k_r) * eta + self.chi_x1_coeff * x1_tilde - self.nu

        return ObserverDerivatives(
            x1_hat=self.x2_hat,
            x2_hat=cache.phi + self.robust_term(e) + chi,
            p=-(k_r + 2.0 * a) * self.p - self.nu + ((a + k_r) ** 2 + 1.0) * x1_tilde,
            nu=self.p - a * self.nu - (a + k_r) * x1_tilde,
            c_hat=cfg.b_c * (cache.c - self.c_hat),
            h_hat=cfg.b_h * (cache.h - self.h_hat),
            theta=self.adaptation.theta_dot(self.theta, cache, e) if cfg.adapt else None,
            e=e,
            eta=eta,
            cache=cache,
        )

    def _euler_step(self, y: np.ndarray, u: np.ndarray, dt: float) -> None:
        d = self.derivatives(y, u)
        self.x1_hat = self.x1_hat + dt * d.x1_hat
        self.x2_hat = self.x2_hat + dt * d.x2_hat
        self.p = self.p + dt * d.p
        self.nu = self.nu + dt * d.nu
        self.c_hat = self.c_hat + dt * d.c_hat
        self.h_hat = self.h_hat + dt * d.h_hat
        if d.theta is not None:
            self.theta += dt * d.theta
            self.adaptation.enforce_bound(self.theta)
        self.phi_hat = d.cache.phi
        self.e = d.e

    def update(self, y: np.ndarray, u: float | np.ndarray, dt: float) -> np.ndarray:
        """
        Advances the observer by one sampling period.
        y: measured x1 (length n), u: applied input (scalar or length m).
        Returns the interleaved state estimate.
        """
        y_arr = np.asarray(y, dtype=np.float64)
        u_arr = np.atleast_1d(np.asarray(u, dtype=np.float64))

        if not self._filter_initialized:
            # p(0) = (alpha + k_r) x_tilde1(0), nu(0) = 0  =>  eta(0) = 0
            cfg = self.config
            self.p = (cfg.alpha + cfg.k_r) * (y_arr - self.x1_hat)
            self.nu = np.zeros_like(self.p)
            self._filter_initialized = True

        h = dt / self.config.n_substeps
        for _ in range(self.config.n_substeps):
            self._euler_step(y_arr, u_arr, h)
        return self.state_estimate()
