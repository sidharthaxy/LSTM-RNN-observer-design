"""
Approach B: Lyapunov-based physics-informed LSTM (PI-LSTM) adaptive state observer
(Hart, Griffis, Patil & Dixon, 2024) for Euler-Lagrange systems
    M(q) q_ddot + V_m(q, q_dot) q_dot + G(q) + F(q_dot) = B u + d(t),   y = q + v.

Observer (same auxiliary filter and chi as the black-box Lb-LSTM of Approach A; only the
acceleration model changes):
    q_hat_dot     = q_dot_hat
    q_dot_hat_dot = Phi_hat + k_s sgn(e) + chi
    Phi_hat       = M_hat^{-1}(q_hat) [ B u - V_m_hat(q_hat, q_dot_hat) q_dot_hat - G_hat(q_hat) - F_hat ]

Dynamic auxiliary filter, q_tilde = y - q_hat (measurable):
    eta    = p - (alpha + k_r) q_tilde
    p_dot  = -(k_r + 2 alpha) p - nu + ((alpha + k_r)^2 + 1) q_tilde
    nu_dot = p - alpha nu - (alpha + k_r) q_tilde
    e      = q_tilde + nu
    chi    = -(3 alpha + k_r) eta + (2 - alpha^2) q_tilde - nu
With r = q_dot_tilde + alpha q_tilde + eta, V_0 = 1/2 (|q_tilde|^2 + |eta|^2 + |nu|^2 + |r|^2) satisfies
    V_0_dot = -alpha (|q_tilde|^2 + |eta|^2 + |nu|^2) - k_r |r|^2 + r^T (q_ddot - Phi_hat - k_s sgn(e))
(derivation: Approach A README, Section 3). None of these signals needs the velocity.

Adaptation (src/adaptation/pilstm_jacobian_engine.py):
    theta_beta_hat_dot = proj( Gamma_beta Phi'_beta^T e ),   beta in {M, V, G, F}.

Physical invariants hold for *every* theta_hat (src/observers/pilstm_network.py), so they hold
along the whole adaptation transient, not only at convergence:
    M_hat = M_hat^T >= eps_M I  =>  ||M_hat^{-1}|| <= 1 / eps_M, T_hat = 1/2 q_dot^T M_hat q_dot >= 0,
    M_hat_dot - 2 V_m_hat skew-symmetric, G_hat = grad P_hat, q_dot^T F_hat >= 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple
import numpy as np

from src.adaptation.pilstm_jacobian_engine import PILSTMAdaptationLaw
from src.observers.pilstm_network import PILSTMCache, PILSTMNetwork


@dataclass
class PILSTMObserverConfig:
    """Structure, gains and adaptation rates of the PI-LSTM observer."""
    n_coords: int = 2
    n_inputs: int = 1
    input_matrix: Tuple[Tuple[float, ...], ...] = ((1.0,), (0.0,))   # B: force acts on the cart
    revolute: Tuple[bool, ...] = (False, True)                      # joint types of q = [x, theta]

    # Structured sub-networks
    harmonics: int = 2             # Fourier features [cos k q_i, sin k q_i], k <= K, per revolute joint
    n_features: int = 0            # extra random tanh features of q (shared by M, V, G)
    feature_scale: float = 1.0     # std of the fixed feature weights V_0
    position_scale: Tuple[float, ...] | None = None    # prismatic-coordinate normalization s
    position_offset: Tuple[float, ...] | None = None   # prismatic-coordinate centering mu
    linear_skip: bool = True       # rho = [1, xi(q), tanh(.)] instead of [1, tanh(.)]
    cyclic: Tuple[bool, ...] = (True, False)   # x absent from the learned Lagrangian (level track)
    eps_M: float = 0.05            # lambda_min(M_hat) >= eps_M
    inertia_init: Tuple[float, ...] = (2.0, 0.2)       # M_hat(q; theta(0)) = diag(inertia_init)
    inertia_scale: Tuple[float, ...] | None = None     # Cholesky row units d_i^2; None -> inertia_init
    gyroscopic: bool = True        # adaptive skew-symmetric term S(q, q_dot; theta_V)
    friction_hidden: int = 8       # L_F: friction LSTM width
    velocity_scale: Tuple[float, ...] | None = None    # friction LSTM input normalization
    dissipative_friction: bool = True   # F_hat = diag(d(h) >= 0) q_dot (passive) vs. unconstrained
    damping_offset: float = -3.0        # d_i(0) = d0_i softplus(-3) = 0.049 d0_i, d0 = inertia scale
    b_c: float = 5.0               # friction LSTM memory bandwidths [1/s]
    b_h: float = 5.0
    gate_scale: float = 0.5        # std of the friction gate initialization

    # Auxiliary filter and robust feedback (identical to Approach A)
    alpha: float = 4.0
    k_r: float = 8.0
    k_s: float = 0.2
    sign_mode: Literal["sgn", "tanh"] = "sgn"
    tanh_eps: float = 0.01
    chi_x1_coeff: float | None = None   # None -> 2 - alpha^2

    # Blockwise Lyapunov adaptation
    gamma_M: float = 5.0
    gamma_V: float = 0.5
    gamma_G: float = 10.0
    gamma_F_gates: float = 5.0
    gamma_F_out: float = 20.0
    w_bar_M: float = 5.0
    w_bar_V: float = 5.0
    w_bar_G: float = 20.0
    w_bar_F: float = 20.0
    proj_eps: float = 0.1
    metric: Literal["euclidean", "kinetic"] = "kinetic"   # error metric of the adaptation gradient
    adapt: bool = True

    n_substeps: int = 1
    seed: int = 0


class PILSTMObserver:
    """
    Physics-informed Lb-LSTM observer. Same interface as `LbLSTMObserver`:
    `reset(x0)` / `update(y, u, dt)` with interleaved ordering [q_0, q_dot_0, q_1, q_dot_1, ...].
    """

    def __init__(self, config: PILSTMObserverConfig | None = None) -> None:
        self.config = config or PILSTMObserverConfig()
        cfg = self.config
        n = cfg.n_coords
        self.net = PILSTMNetwork(
            n_coords=n,
            input_matrix=np.asarray(cfg.input_matrix, dtype=np.float64),
            revolute=cfg.revolute,
            n_features=cfg.n_features,
            harmonics=cfg.harmonics,
            feature_scale=cfg.feature_scale,
            position_scale=cfg.position_scale,
            position_offset=cfg.position_offset,
            linear_skip=cfg.linear_skip,
            cyclic=cfg.cyclic,
            inertia_scale=cfg.inertia_init if cfg.inertia_scale is None else cfg.inertia_scale,
            eps_M=cfg.eps_M,
            gyroscopic=cfg.gyroscopic,
            friction_hidden=cfg.friction_hidden,
            velocity_scale=cfg.velocity_scale,
            dissipative_friction=cfg.dissipative_friction,
            damping_offset=cfg.damping_offset,
            seed=cfg.seed,
        )
        if self.net.B.shape[1] != cfg.n_inputs:
            raise ValueError("input_matrix must have n_inputs columns.")
        self.layout = self.net.layout
        self.adaptation = PILSTMAdaptationLaw(
            self.net,
            gammas={"M": cfg.gamma_M, "V": cfg.gamma_V, "G": cfg.gamma_G,
                    "F_gates": cfg.gamma_F_gates, "F_out": cfg.gamma_F_out},
            w_bars={"M": cfg.w_bar_M, "V": cfg.w_bar_V, "G": cfg.w_bar_G, "F": cfg.w_bar_F},
            proj_eps=cfg.proj_eps,
            metric=cfg.metric,
        )
        self.chi_x1_coeff = 2.0 - cfg.alpha ** 2 if cfg.chi_x1_coeff is None else cfg.chi_x1_coeff

        self.theta0 = self.net.initial_parameters(cfg.inertia_init, cfg.gate_scale, seed=cfg.seed)
        for b, norm in self.adaptation.block_norms(self.theta0).items():
            if norm > self.adaptation.w_bars[b] / np.sqrt(1.0 + cfg.proj_eps):
                raise ValueError(f"theta_{b}(0) lies outside its projection ball; raise w_bar_{b}.")
        self.theta = np.copy(self.theta0)

        L_F = cfg.friction_hidden
        self.q_hat = np.zeros(n)
        self.q_dot_hat = np.zeros(n)
        self.p = np.zeros(n)
        self.nu = np.zeros(n)
        self.c_hat = np.zeros(L_F)
        self.h_hat = np.zeros(L_F)
        self.phi_hat = np.zeros(n)
        self.e = np.zeros(n)
        self.last_cache: PILSTMCache | None = None
        self._filter_initialized = False

    # ------------------------------------------------------------------ state handling
    def reset(self, initial_state: np.ndarray | None = None, reset_weights: bool = True) -> None:
        n = self.config.n_coords
        if initial_state is None:
            self.q_hat = np.zeros(n)
            self.q_dot_hat = np.zeros(n)
        else:
            x0 = np.asarray(initial_state, dtype=np.float64).reshape(n, 2)
            self.q_hat = np.copy(x0[:, 0])
            self.q_dot_hat = np.copy(x0[:, 1])
        for arr in (self.p, self.nu, self.c_hat, self.h_hat, self.phi_hat, self.e):
            arr.fill(0.0)
        self.last_cache = None
        self._filter_initialized = False
        if reset_weights:
            self.theta = np.copy(self.theta0)

    def state_estimate(self) -> np.ndarray:
        return np.column_stack((self.q_hat, self.q_dot_hat)).ravel()

    @property
    def theta_norm(self) -> float:
        return float(np.linalg.norm(self.theta))

    def block_norms(self) -> Dict[str, float]:
        return self.adaptation.block_norms(self.theta)

    # ------------------------------------------------------------------ physics diagnostics
    def inertia_estimate(self, q: np.ndarray | None = None) -> np.ndarray:
        return self.net.inertia(self.theta, self.q_hat if q is None else q).M

    def kinetic_energy_estimate(self) -> float:
        """T_hat = 1/2 q_dot_hat^T M_hat(q_hat) q_dot_hat."""
        return self.net.kinetic_energy(self.theta, self.q_hat, self.q_dot_hat)

    def friction_power(self) -> float:
        """q_dot_hat^T F_hat: >= 0 when the learned friction dissipates."""
        return 0.0 if self.last_cache is None else float(self.last_cache.q_dot @ self.last_cache.friction_force)

    # ------------------------------------------------------------------ dynamics
    def robust_term(self, e: np.ndarray) -> np.ndarray:
        cfg = self.config
        if cfg.sign_mode == "tanh":
            return cfg.k_s * np.tanh(e / cfg.tanh_eps)
        return cfg.k_s * np.sign(e)

    def model(self, u: np.ndarray) -> PILSTMCache:
        return self.net.forward(self.theta, self.q_hat, self.q_dot_hat, u, self.c_hat, self.h_hat)

    def _euler_step(self, y: np.ndarray, u: np.ndarray, dt: float) -> None:
        cfg = self.config
        a, k_r = cfg.alpha, cfg.k_r

        q_tilde = y - self.q_hat
        eta = self.p - (a + k_r) * q_tilde
        e = q_tilde + self.nu
        cache = self.model(u)
        chi = -(3.0 * a + k_r) * eta + self.chi_x1_coeff * q_tilde - self.nu

        d_q = self.q_dot_hat
        d_qd = cache.phi + self.robust_term(e) + chi
        d_p = -(k_r + 2.0 * a) * self.p - self.nu + ((a + k_r) ** 2 + 1.0) * q_tilde
        d_nu = self.p - a * self.nu - (a + k_r) * q_tilde
        d_c = cfg.b_c * (cache.friction.c - self.c_hat)
        d_h = cfg.b_h * (cache.friction.h - self.h_hat)
        d_theta = self.adaptation.theta_dot(self.theta, cache, e) if cfg.adapt else None

        self.q_hat = self.q_hat + dt * d_q
        self.q_dot_hat = self.q_dot_hat + dt * d_qd
        self.p = self.p + dt * d_p
        self.nu = self.nu + dt * d_nu
        self.c_hat = self.c_hat + dt * d_c
        self.h_hat = self.h_hat + dt * d_h
        if d_theta is not None:
            self.theta += dt * d_theta
            self.adaptation.enforce_bound(self.theta)
        self.phi_hat = cache.phi
        self.e = e
        self.last_cache = cache

    def update(self, y: np.ndarray, u: float | np.ndarray, dt: float) -> np.ndarray:
        """Advances the observer by one sampling period (ZOH on y, u; Euler sub-steps)."""
        y_arr = np.asarray(y, dtype=np.float64)
        u_arr = np.atleast_1d(np.asarray(u, dtype=np.float64))
        if not self._filter_initialized:
            cfg = self.config
            self.p = (cfg.alpha + cfg.k_r) * (y_arr - self.q_hat)
            self.nu = np.zeros_like(self.p)
            self._filter_initialized = True
        h = dt / self.config.n_substeps
        for _ in range(self.config.n_substeps):
            self._euler_step(y_arr, u_arr, h)
        return self.state_estimate()
