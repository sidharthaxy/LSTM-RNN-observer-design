"""
Approach C: Concurrent-Learning Lyapunov-based LSTM observer (CL-Lb-LSTM).

The observer is the continuous-time Lb-LSTM of Approach A (Griffis, Patil, Hart & Dixon,
IEEE L-CSS 2024): LSTM memory ODEs, dynamic auxiliary filter (p, nu), robust term and
chi feedback are inherited unchanged from `LbLSTMObserver`. Only the weight adaptation
changes, from the instantaneous law to the dual-loop concurrent-learning law

    theta_hat_dot = proj( Gamma Phi'(t)^T e(t)
                          + Gamma_CL sum_{j=1}^N Phi'(t_j)^T (a(t_j) - Phi_hat(t_j)) ).

Parameterization. theta = [theta_g, theta_h] with theta_g the gate weights
vec(W_c, W_i, W_f, W_o) and theta_h = vec(W_h) the readout.
    * theta_h (readout) is learned by both loops. Phi_hat = W_h^T h is linear in theta_h, so
      Phi'_j = I_n kron h_j^T, Omega = I_n kron sum_j h_j h_j^T, and the rank condition
      lambda_min(Omega) > 0 is attainable with N >= L stored points.
    * theta_g (gates) receive only the CL term by default (gamma_gates = 0,
      gamma_cl_gates = 5): the full Jacobian Phi'_g(t_j) of the stored points drives the gates
      down the stack loss 1/2 sum_j ||a_j - Phi_hat_j||^2 (first-order Taylor form of the CL
      term in the nonlinear parameters). The stack features h_j are re-evaluated at the
      current gates every step. gamma_gates > 0 adds Approach A's instantaneous gate law;
      gamma_cl_gates = 0 freezes the gates (fixed recurrent feature map, for which the
      readout convergence proof in the README is exact).
    The rank condition cannot be imposed on the full theta: rank(Omega_full) <= n N while
    dim(theta) = 4dL + Ln (1440 for the default architecture). It is imposed, and monitored,
    on the readout block; the gate block has no excitation guarantee.

History stack data. The acceleration x_ddot(t_j) is never measured. Its proxy comes from
`SavitzkyGolayAccelerationProxy`, a causal, delayed-centre Savitzky-Golay differentiator
over the last W measured positions: at time t_k it returns the second derivative at the
window centre t_k - (W-1)/2 * dt. The observer keeps a ring buffer of its own (zeta, c_hat,
u, y) snapshots so the stack entry pairs the proxy with the network input at exactly that
centre time. Only past data is used; the (W-1)/2-sample delay is irrelevant for CL, which
learns from recorded data anyway.

Integration. The instantaneous law, the gate CL term and all observer states use the
forward Euler scheme of Approach A. With the gates fixed within a step, the CL term Gamma_CL (B - Omega W_h),
B = sum_j h_j a_j^T, is affine in W_h and stiff (Gamma_CL lambda_max(Omega) T_s can exceed
2), so it is integrated implicitly (IMEX Euler):
    W_h^+ = (I + T_s Gamma_CL Omega)^{-1} (W_h + T_s Gamma Phi'^T e + T_s Gamma_CL B),
which is unconditionally stable and has the exact fixed point W_h = Omega^{-1} B of the CL
loop. The norm bound ||theta|| <= W_bar is enforced after each step (Approach A safeguard).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Deque, Tuple
import numpy as np

from scipy.special import expit

from src.adaptation.jacobian_engine import LSTMWeightLayout
from src.concurrent_learning.history_stack import HistoryStack, HistoryStackEntry, StackDecision
from src.observers.blackbox_lstm import LbLSTMObserver, LbLSTMObserverConfig

if TYPE_CHECKING:
    from src.identification.extract_model import FrozenLbLSTM


# ---------------------------------------------------------------------- acceleration proxy
class SavitzkyGolayAccelerationProxy:
    """
    Causal delayed-centre Savitzky-Golay differentiator for the unmeasured acceleration.

    A degree-P polynomial is least-squares fitted to the last W (odd) position samples,
    tau_k = (k - (W-1)/2) dt, k = 0..W-1, and differentiated at the window centre:
        q(t_c) ~ c0^T Y,   q_dot(t_c) ~ c1^T Y,   q_ddot(t_c) ~ c2^T Y,
    with c_i the rows of the pseudo-inverse of the Vandermonde matrix (times i!).
    For i.i.d. measurement noise of standard deviation sigma the estimate has standard
    deviation sigma ||c2||; for P = 2 (or 3, identical for a centred window)
    ||c2||^2 ~ 720 dt / T_w^5, T_w = W dt. The centred window is exact for polynomials of
    degree <= P and has zero phase lag relative to t_c.
    """

    def __init__(self, window_samples: int, poly_order: int, dt: float) -> None:
        if window_samples % 2 == 0:
            window_samples += 1
        if window_samples <= poly_order + 1 or poly_order < 2:
            raise ValueError("Need poly_order >= 2 and window_samples > poly_order + 1.")
        self.window = window_samples
        self.poly_order = poly_order
        self.dt = dt
        self.delay = (window_samples - 1) // 2
        tau = (np.arange(window_samples) - self.delay) * dt
        V = np.vander(tau, poly_order + 1, increasing=True)   # (W, P+1)
        pinv = np.linalg.pinv(V)                              # (P+1, W)
        self.c0 = pinv[0]
        self.c1 = pinv[1]
        self.c2 = 2.0 * pinv[2]

    @classmethod
    def from_duration(cls, window_seconds: float, poly_order: int, dt: float) -> SavitzkyGolayAccelerationProxy:
        return cls(round(window_seconds / dt) | 1, poly_order, dt)

    @property
    def noise_gain(self) -> float:
        """||c2||: acceleration-estimate std per unit position-noise std."""
        return float(np.linalg.norm(self.c2))

    def estimate(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """positions: (W, n) oldest first. Returns (q, q_dot, q_ddot) at the window centre."""
        Y = np.asarray(positions, dtype=np.float64)
        if Y.shape[0] != self.window:
            raise ValueError(f"Expected {self.window} samples, got {Y.shape[0]}.")
        return self.c0 @ Y, self.c1 @ Y, self.c2 @ Y


def batch_lstm(
    theta: np.ndarray, Z: np.ndarray, C: np.ndarray, layout: LSTMWeightLayout, sensitivities: bool = False,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Batched LSTM cell for the stored operating points: Z (N, d) zetas, C (N, L) cell memories.
    Returns H (N, L) hidden outputs and, if requested, the stacked gate sensitivities
    [delta_c, delta_i, delta_f, delta_o] (N, 4L) of `jacobian_engine.gate_sensitivities`.
    """
    L = layout.hidden_dim
    W_gates, _ = layout.unpack(theta)
    pre = Z @ W_gates
    c_star = np.tanh(pre[:, 0:L])
    i_gate = expit(pre[:, L:2 * L])
    f_gate = expit(pre[:, 2 * L:3 * L])
    o_gate = expit(pre[:, 3 * L:4 * L])
    tanh_c = np.tanh(f_gate * C + i_gate * c_star)
    H = o_gate * tanh_c
    if not sensitivities:
        return H, None
    dh_dc = o_gate * (1.0 - tanh_c ** 2)
    delta = np.hstack((
        dh_dc * i_gate * (1.0 - c_star ** 2),
        dh_dc * c_star * i_gate * (1.0 - i_gate),
        dh_dc * C * f_gate * (1.0 - f_gate),
        tanh_c * o_gate * (1.0 - o_gate),
    ))
    return H, delta


def batch_hidden_features(theta: np.ndarray, Z: np.ndarray, C: np.ndarray, layout: LSTMWeightLayout) -> np.ndarray:
    """Instantaneous LSTM hidden outputs h for a batch of (zeta, c_hat): Z (N, d), C (N, L) -> (N, L)."""
    return batch_lstm(theta, Z, C, layout)[0]


# ---------------------------------------------------------------------- observer
@dataclass
class CLLbLSTMObserverConfig(LbLSTMObserverConfig):
    """Approach A gains plus the concurrent-learning loop."""
    gamma_gates: float = 0.0          # Gates frozen: fixed recurrent feature map (see module doc)
    w_bar: float = 80.0               # ||theta|| bound; ~19 of it is the frozen random gate init

    cl_enabled: bool = True           # False -> standard instantaneous Lb-LSTM law (stack still recorded)
    gamma_cl: float = 50.0            # Gamma_CL on vec(W_h) (implicit step)
    gamma_cl_gates: float = 5.0       # Gamma_CL on the gate weights (explicit step; no rank guarantee)
    cl_normalize_outputs: bool = False  # True: weight CL residuals by Lambda = diag(1 / Var(a_i)) of the stack
    stack_capacity: int = 48          # N (must be >= hidden_dim)
    record_start: float = 2.0         # [s] ignore data before the state estimate has converged
    record_interval: float = 0.02     # [s] minimum spacing between candidate points
    novelty_tol: float = 0.05         # relative regressor-change gate
    min_rel_improvement: float = 1e-3 # relative lambda_min gain required for a swap
    sg_window: float = 0.25           # [s] Savitzky-Golay window T_w
    sg_order: int = 3                 # Savitzky-Golay polynomial degree


@dataclass
class _Snapshot:
    t: float
    zeta: np.ndarray
    c_hat: np.ndarray
    h: np.ndarray
    u: np.ndarray
    y: np.ndarray


class CLLbLSTMObserver(LbLSTMObserver):
    """
    Concurrent-learning Lb-LSTM observer. Drop-in replacement for `LbLSTMObserver`
    (same `reset` / `update` interface and state ordering).
    """

    def __init__(self, config: CLLbLSTMObserverConfig | None = None) -> None:
        cfg = config or CLLbLSTMObserverConfig()
        super().__init__(cfg)
        self.cl_config = cfg
        L = cfg.hidden_dim
        self.h_slice = self.layout.block_slice("h")
        self.stack = HistoryStack(
            capacity=cfg.stack_capacity,
            regressor_dim=L,
            novelty_tol=cfg.novelty_tol,
            min_rel_improvement=cfg.min_rel_improvement,
        )
        self.proxy: SavitzkyGolayAccelerationProxy | None = None
        self._buffer: Deque[_Snapshot] = deque()
        self._reset_cl_state()

    # ------------------------------------------------------------------ state handling
    def _reset_cl_state(self) -> None:
        self.stack.clear()
        self._buffer.clear()
        self.t = 0.0
        self._last_record_t = -np.inf
        self._pending: _Snapshot | None = None
        self._capture = False
        self._cl_cache_key: tuple[int, float] | None = None
        self._omega = np.zeros((self.cl_config.hidden_dim, self.cl_config.hidden_dim))
        self._B = np.zeros((self.cl_config.hidden_dim, self.cl_config.n_coords))
        self._implicit_inv: np.ndarray | None = None
        self._lambda = np.ones(self.cl_config.n_coords)
        self._gate_grad: np.ndarray | None = None
        self.last_decision: StackDecision | None = None
        self.h_features = np.zeros(self.cl_config.hidden_dim)   # h(t_k) of the last sample
        self._arrays_version = -1
        self._arrays: Tuple[np.ndarray, np.ndarray, np.ndarray] = (np.zeros(0), np.zeros(0), np.zeros(0))

    def reset(self, initial_state: np.ndarray | None = None, reset_weights: bool = True) -> None:
        super().reset(initial_state, reset_weights)
        self._reset_cl_state()

    # ------------------------------------------------------------------ parameter views
    @property
    def W_h(self) -> np.ndarray:
        """Readout W_h (L x n), a copy."""
        return self.layout.weight_matrix(self.theta, "h").copy()

    @property
    def theta_h(self) -> np.ndarray:
        return self.theta[self.h_slice].copy()

    @property
    def feature_map_adapts(self) -> bool:
        """True when the gate weights (hence the features h_j of stored points) change online."""
        cfg = self.cl_config
        return cfg.adapt and (cfg.gamma_gates > 0.0 or (cfg.cl_enabled and cfg.gamma_cl_gates > 0.0))

    def _stack_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(Z, C, A): stored zetas, cell memories and targets, cached per stack version."""
        if self._arrays_version != self.stack.version:
            self._arrays = (
                np.vstack([e.zeta for e in self.stack.entries]),
                np.vstack([e.c_hat for e in self.stack.entries]),
                self.stack.targets(),
            )
            self._arrays_version = self.stack.version
        return self._arrays

    def _stack_inputs(self) -> Tuple[np.ndarray, np.ndarray]:
        Z, C, _ = self._stack_arrays()
        return Z, C

    def lambda_min(self) -> float:
        """lambda_min(Omega) of the history stack (Omega = I_n kron sum_j h_j h_j^T)."""
        return self.stack.lambda_min()

    def stack_features(self) -> np.ndarray:
        """H = [h_1, ..., h_N]^T at the current gate weights, shape (N, L)."""
        if len(self.stack) == 0:
            return np.zeros((0, self.cl_config.hidden_dim))
        if not self.feature_map_adapts:
            return self.stack.data_matrix()
        Z, C = self._stack_inputs()
        return batch_hidden_features(self.theta, Z, C, self.layout)

    def cl_residuals(self) -> np.ndarray:
        """epsilon_j = a(t_j) - Phi_hat(t_j), shape (N, n)."""
        if len(self.stack) == 0:
            return np.zeros((0, self.cl_config.n_coords))
        return self.stack.targets() - self.stack_features() @ self.layout.weight_matrix(self.theta, "h")

    def stack_least_squares(self, ridge: float = 0.0) -> np.ndarray | None:
        """Fixed point of the CL loop, W_H = (Omega + ridge I)^{-1} B, or None while rank deficient."""
        if len(self.stack) < self.cl_config.hidden_dim:
            return None
        H = self.stack_features()
        A = self.stack.targets()
        omega = H.T @ H + ridge * np.eye(H.shape[1])
        if np.linalg.eigvalsh(omega)[0] <= 0.0:
            return None
        return np.linalg.solve(omega, H.T @ A)

    # ------------------------------------------------------------------ CL loop
    def output_weights(self) -> np.ndarray:
        """
        Diagonal of the residual weighting Lambda (n,): 1 / Var(a_i) over the stack targets
        when `cl_normalize_outputs` (every coordinate contributes in normalized units, the
        output-side analogue of the zeta normalization), else ones.
        """
        n = self.cl_config.n_coords
        if not self.cl_config.cl_normalize_outputs or len(self.stack) < 2:
            return np.ones(n)
        return 1.0 / (np.var(self._stack_arrays()[2], axis=0) + 1e-6)

    def _cl_terms(self, dt: float) -> None:
        """
        Omega = H^T H, B = H^T A, Lambda and the per-coordinate implicit-step inverses
        (I + T_s Gamma_CL Lambda_i Omega)^{-1}; plus, with gate CL on, the gate gradient
        sum_j Phi'_g(t_j)^T Lambda (a_j - Phi_hat_j). One batched LSTM pass per step when the
        feature map adapts; cached per stack version otherwise.
        """
        cfg = self.cl_config
        key = (self.stack.version, dt)
        adapts = self.feature_map_adapts
        gate_cl = cfg.gamma_cl_gates > 0.0
        if not adapts and key == self._cl_cache_key:
            return
        Z, C, A = self._stack_arrays()
        if adapts:
            H, delta = batch_lstm(self.theta, Z, C, self.layout, sensitivities=gate_cl)
        else:
            H, delta = self.stack.data_matrix(), None
        lam = self.output_weights()
        L = cfg.hidden_dim
        self._omega = H.T @ H
        self._B = H.T @ A
        self._lambda = lam
        self._implicit_inv = np.linalg.inv(
            np.eye(L)[None, :, :] + (dt * cfg.gamma_cl * lam)[:, None, None] * self._omega[None, :, :]
        )                                                                  # (n, L, L)
        self._gate_grad = None
        if gate_cl:
            assert delta is not None
            self._gate_grad = self._gate_gradient(H, delta, Z, A, lam)
        self._cl_cache_key = key

    def _gate_gradient(self, H: np.ndarray, delta: np.ndarray, Z: np.ndarray, A: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """
        sum_j Phi'_g(t_j)^T Lambda (a_j - Phi_hat_j) without forming Phi'_g. Per point the
        Approach A identity gives Phi'_g^T w = vec(zeta_j (delta_j * tile(W_h w, 4))^T), so the
        stack sum is vec(Z^T BACK) with BACK_j = delta_j * tile(W_h w_j, 4).
        """
        W_h = self.layout.weight_matrix(self.theta, "h")
        E = (A - H @ W_h) * lam                                            # Lambda-weighted residuals
        back = np.tile(E @ W_h.T, (1, 4)) * delta                          # (N, 4L)
        return (back.T @ Z).ravel()                                        # theta[:n_gate_params] layout

    def stack_gate_gradient(self) -> np.ndarray:
        """Gate block of sum_j Phi'(t_j)^T Lambda (a_j - Phi_hat(t_j)) at the current weights."""
        Z, C, A = self._stack_arrays()
        H, delta = batch_lstm(self.theta, Z, C, self.layout, sensitivities=True)
        assert delta is not None
        return self._gate_gradient(H, delta, Z, A, self.output_weights())

    def _euler_step(self, y: np.ndarray, u: np.ndarray, dt: float) -> None:
        d = self.derivatives(y, u)
        cfg = self.cl_config
        use_cl = cfg.cl_enabled and cfg.adapt and len(self.stack) > 0
        if use_cl:
            self._cl_terms(dt)                  # at the weights of the start of the step
        if self._capture:
            self._pending = _Snapshot(
                t=self.t, zeta=d.cache.zeta.copy(), c_hat=d.cache.c_hat.copy(),
                h=d.cache.h.copy(), u=u.copy(), y=y.copy(),
            )
            self.h_features = self._pending.h
            self._capture = False

        self.x1_hat = self.x1_hat + dt * d.x1_hat
        self.x2_hat = self.x2_hat + dt * d.x2_hat
        self.p = self.p + dt * d.p
        self.nu = self.nu + dt * d.nu
        self.c_hat = self.c_hat + dt * d.c_hat
        self.h_hat = self.h_hat + dt * d.h_hat
        if d.theta is not None:
            self.theta += dt * d.theta          # instantaneous (projected) law, explicit

        if use_cl:
            if self._gate_grad is not None:     # CL on the gates, explicit
                self.theta[: self.layout.n_gate_params] += dt * cfg.gamma_cl_gates * self._gate_grad
            assert self._implicit_inv is not None
            W = self.layout.weight_matrix(self.theta, "h")          # after the explicit part
            rhs = W + dt * cfg.gamma_cl * self._lambda[None, :] * self._B
            W_new = np.einsum("ikl,li->ki", self._implicit_inv, rhs)  # column i: inv_i @ rhs[:, i]
            self.theta[self.h_slice] = W_new.ravel(order="F")

        if cfg.adapt:
            self.adaptation.enforce_bound(self.theta)
        self.phi_hat = d.cache.phi
        self.e = d.e

    def _record(self, dt: float) -> None:
        """Pushes the pending snapshot and offers the window-centre point to the stack."""
        cfg = self.cl_config
        assert self._pending is not None
        if self.proxy is None:
            self.proxy = SavitzkyGolayAccelerationProxy.from_duration(cfg.sg_window, cfg.sg_order, dt)
            self._buffer = deque(maxlen=self.proxy.window)
        self._buffer.append(self._pending)
        self._pending = None
        if len(self._buffer) < self.proxy.window:
            return

        centre = self._buffer[self.proxy.delay]
        if centre.t < cfg.record_start or centre.t - self._last_record_t < cfg.record_interval - 1e-12:
            return
        positions = np.vstack([s.y for s in self._buffer])
        _, _, accel = self.proxy.estimate(positions)

        if self.feature_map_adapts:
            # Adapted feature map: rank decisions must use features at the current gates.
            self.stack.refresh_regressors(lambda Z, C: batch_hidden_features(self.theta, Z, C, self.layout))
            h = batch_hidden_features(self.theta, centre.zeta[None, :], centre.c_hat[None, :], self.layout)[0]
        else:
            h = centre.h
        entry = HistoryStackEntry(
            zeta=centre.zeta, c_hat=centre.c_hat, u=centre.u, x_meas=centre.y,
            accel_target=accel, t=centre.t, regressor=h,
        )
        self.last_decision = self.stack.consider(entry)
        self._last_record_t = centre.t

    def update(self, y: np.ndarray, u: float | np.ndarray, dt: float) -> np.ndarray:
        """Advances the observer by one sample and updates the history stack."""
        self._capture = True
        est = super().update(y, u, dt)
        self._record(dt)
        self.t += dt
        return est

    # ------------------------------------------------------------------ model extraction
    def base_config(self) -> LbLSTMObserverConfig:
        """The architecture fields only (what the digital twin needs)."""
        return LbLSTMObserverConfig(**{f.name: getattr(self.cl_config, f.name) for f in fields(LbLSTMObserverConfig)})

    def freeze(self) -> FrozenLbLSTM:
        """Snapshot of the current weights as an Approach-A-compatible frozen model."""
        from src.identification.extract_model import FrozenLbLSTM  # deferred: identification imports observers
        return FrozenLbLSTM(theta=np.copy(self.theta), config=self.base_config(), t_freeze=float(self.t))
