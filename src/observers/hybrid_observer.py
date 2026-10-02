"""
Approach E: hybrid of Approach B (physics-informed LSTM) and Approach D (physics-structured
integral concurrent learning).

Model. Approach D's linear-in-parameters Euler-Lagrange model. Approach B's dissipative friction
LSTM can be added as a residual (`residual=True`; off by default, see the README):

    Phi_hat = M_hat^{-1} (B u - Y_rest(q_hat, q_dot_hat) theta_hat - F_res),
    F_res   = diag(d) q_dot_hat,   d_i = d0_i softplus(phi_F,i + offset) >= 0,

where phi_F is the readout of a continuous-time LSTM on the estimated velocities. The rigid body,
viscous and Coulomb terms stay linear in theta (so integral CL can identify them); the residual can
only dissipate, so the learned model stays passive. The ICL windows do not include the residual.

Three changes relative to Approach D:

1. Own-coordinate inertia freeze. The diagonal inertia M_ii of a joint does not depend on that
   joint's own coordinate. The feature coefficients of M_ii that depend on q_i are fixed at zero.
   Without this, the normalized regression of an unactuated row has the degenerate near-solution
   M_ii proportional to (1 + cos q_i) when the data stay near q_i = pi: the row's inertia vanishes
   there, and its scale becomes arbitrary.

2. Information-weighted blend of the two adaptation laws. For a stack with information matrix
   Omega and cross term c (Omega phi* = c), Approach D pulls phi to the least-squares solution at
   the same rate in every direction (Newton form), which amplifies noise in weakly excited
   directions. Here the pull is damped (Levenberg-Marquardt):

       CL increment:            (Omega + lambda I)^{-1} (c - Omega phi)   = W_s (phi_LS - phi),
       instantaneous gradient:  W_w g,          W_w = lambda (Omega + lambda I)^{-1} = I - W_s.

   Directions with eigenvalue >> lambda are identified by integral CL (Approach D); directions with
   eigenvalue << lambda are left to the instantaneous kinetic-metric law (Approach B), which can
   then run at a much larger gain because the two laws no longer act on the same directions.
   The unactuated rows use the same blend in their normalized coordinates (kappa, psi).

3. No all-or-nothing rank gate. A stack is used once it holds `cl_min_windows` windows; the
   damping replaces the lambda_min threshold.

Everything else is Approach D: the auxiliary filter and feedback of Approach A, the ICL windows
built from a model-free Savitzky-Golay smoother, the two-part stack estimate, the projections and
the change detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from src.adaptation.jacobian_engine import (
    LSTMCache, LSTMWeightLayout, LyapunovAdaptationLaw, lstm_forward, sigmoid,
)
from src.observers.el_linear_model import LinearELModel
from src.observers.pi_icl_observer import PIICLObserver, PIICLObserverConfig
from src.observers.pilstm_network import softplus


@dataclass
class HybridObserverConfig(PIICLObserverConfig):
    # Blend of the instantaneous law (Approach B) and integral CL (Approach D)
    gamma_inst: float = 20.0          # instantaneous gain; acts only on weakly excited directions
    blend: bool = True                # False: Approach D's Newton CL with its rank gate
    lambda_damp: float = 0.01         # lambda: eigenvalue that separates "identified" from "weak"
    freeze_own_inertia: bool = True   # M_ii independent of q_i

    # Optional residual: Approach B's dissipative friction LSTM (no measurable effect in simulation)
    residual: bool = False
    residual_hidden: int = 8
    residual_offset: float = -3.0     # d_i(0) = d0_i softplus(-3) = 0.049 d0_i, as in Approach B
    residual_gate_scale: float = 0.5
    gamma_res_gates: float = 5.0
    gamma_res_out: float = 20.0
    w_bar_res: float = 20.0
    b_c: float = 5.0                  # residual LSTM memory bandwidths [1/s]
    b_h: float = 5.0
    velocity_scale: Tuple[float, ...] | None = None   # residual LSTM input normalization


def damped_step(omega: np.ndarray, cross: np.ndarray, phi: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    (delta, W_w) for one stack: delta = (Omega + lam I)^{-1}(c - Omega phi) is the damped Newton
    direction towards the stack solution, W_w = lam (Omega + lam I)^{-1} projects onto the weakly
    excited directions. delta has no component along the null space of Omega.
    """
    inv = np.linalg.inv(omega + lam * np.eye(omega.shape[0]))
    return inv @ (cross - omega @ phi), lam * inv


class FrozenHybridModel:
    """Frozen acceleration model of a HybridObserver, decoupled from the observer (a twin)."""

    def __init__(self, obs: "HybridObserver") -> None:
        self.model = obs.model
        self.theta = np.copy(obs.theta)
        self.theta_res = np.copy(obs.theta_res)
        self.residual = obs.config.residual
        self.layout = obs.res_layout
        self.L = obs.config.residual_hidden if self.residual else 0
        self.n_mem = 2 * self.L
        self.b_c, self.b_h = obs.config.b_c, obs.config.b_h
        self.v_scale, self.d0, self.offset = obs.v_scale, obs.d0, obs.config.residual_offset

    def evaluate(self, q: np.ndarray, q_dot: np.ndarray, u: np.ndarray, mem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(acceleration, d mem / dt) with mem = [c_hat, h_hat] of the residual LSTM."""
        m = self.model
        rho, _ = m.rho(q)
        M = m.inertia(self.theta, q, rho)
        tau = m.B @ np.atleast_1d(u) - m.rest_regressor(q, q_dot) @ self.theta
        if not self.residual:
            return np.linalg.solve(M, tau), mem
        c_hat, h_hat = mem[: self.L], mem[self.L:]
        cache = lstm_forward(self.theta_res, np.concatenate((self.v_scale * q_dot, h_hat, (1.0,))), c_hat, self.layout)
        tau = tau - self.d0 * softplus(cache.phi + self.offset) * q_dot
        return np.linalg.solve(M, tau), np.concatenate((self.b_c * (cache.c - c_hat), self.b_h * (cache.h - h_hat)))


class HybridObserver(PIICLObserver):
    """Approach E observer. Interface of the other observers: reset(x0), update(y, u, dt)."""

    config: HybridObserverConfig

    def __init__(self, config: HybridObserverConfig | None = None) -> None:
        cfg = config or HybridObserverConfig()
        n, L = cfg.n_coords, cfg.residual_hidden
        self.res_layout = LSTMWeightLayout(n + L + 1, L, n)
        self.res_law = LyapunovAdaptationLaw(self.res_layout, cfg.gamma_res_gates, cfg.gamma_res_out, cfg.w_bar_res)
        self.v_scale = np.ones(n) if cfg.velocity_scale is None else np.asarray(cfg.velocity_scale, dtype=np.float64)
        self.d0 = np.asarray(cfg.inertia_init, dtype=np.float64)
        rng = np.random.default_rng(cfg.seed)
        self.theta_res0 = np.zeros(self.res_layout.n_params)
        self.theta_res0[: self.res_layout.n_gate_params] = rng.normal(0.0, cfg.residual_gate_scale, self.res_layout.n_gate_params)
        self.theta_res = self.theta_res0.copy()
        self.frozen = np.zeros(0, dtype=bool)
        super().__init__(cfg)

    # ------------------------------------------------------------------ structure
    def _frozen_mask(self) -> np.ndarray:
        """Feature coefficients of M_ii that depend on q_i (fixed at zero)."""
        m, lay = self.model, self.model.layout
        P = lay.n_features
        frozen = np.zeros(lay.n_params, dtype=bool)
        if not self.config.freeze_own_inertia:
            return frozen
        rng = np.random.default_rng(7)
        depends = np.zeros((P, m.n), dtype=bool)
        for _ in range(8):
            depends |= np.abs(m.rho(rng.uniform(-3.0, 3.0, m.n))[1]) > 1e-12
        start = lay.block_slice("M").start
        for e, (i, j) in enumerate(m.entries):
            if i == j:
                frozen[start + e * P + np.flatnonzero(depends[:, i])] = True
        return frozen

    def _structural_support(self) -> np.ndarray:
        support = super()._structural_support()
        self.frozen = self._frozen_mask()
        support[:, self.frozen] = False                 # frozen parameters enter no stack
        self.metric = self.metric.copy()
        self.metric[self.frozen] = 0.0                  # and are not moved by the inertia projection
        return support

    # ------------------------------------------------------------------ state handling
    def reset(self, initial_state: np.ndarray | None = None, reset_weights: bool = True) -> None:
        super().reset(initial_state, reset_weights)
        L = self.config.residual_hidden
        self.c_res, self.h_res = np.zeros(L), np.zeros(L)
        if reset_weights:
            self.theta_res = self.theta_res0.copy()
        self.residual_force = np.zeros(self.config.n_coords)
        self._residual_power = 0.0
        self._hkey: Tuple[int, ...] | None = None
        self._A: Tuple[np.ndarray, np.ndarray] | None = None
        self._U: List[Tuple[np.ndarray, np.ndarray] | None] = []

    def frozen_model(self) -> FrozenHybridModel:
        return FrozenHybridModel(self)

    def residual_power(self) -> float:
        """q_dot_hat^T F_res >= 0: the residual can only dissipate."""
        return self._residual_power

    # ------------------------------------------------------------------ model
    def _residual(self) -> Tuple[np.ndarray, np.ndarray, LSTMCache | None]:
        """(F_res, dF_res/dphi_F (diagonal), LSTM cache)."""
        n = self.config.n_coords
        if not self.config.residual:
            return np.zeros(n), np.zeros(n), None
        zeta = np.concatenate((self.v_scale * self.q_dot_hat, self.h_res, (1.0,)))
        cache = lstm_forward(self.theta_res, zeta, self.c_res, self.res_layout)
        z = cache.phi + self.config.residual_offset
        return self.d0 * softplus(z) * self.q_dot_hat, self.d0 * sigmoid(z) * self.q_dot_hat, cache

    def _model_terms(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LSTMCache | None]:
        """(Phi_hat, M_hat, Y(q_hat, q_dot_hat, Phi_hat), dF_res/dphi_F, cache)."""
        m = self.model
        blocks = m.regressors(self.q_hat, self.q_dot_hat)
        rho, _ = m.rho(self.q_hat)
        M = m.inertia(self.theta, self.q_hat, rho)
        Y = m._assemble(blocks["cor"], blocks, True)
        F_res, gain, cache = self._residual()
        phi_hat = np.linalg.solve(M, m.B @ u - Y @ self.theta - F_res)
        acc = (m._E_times(phi_hat)[:, :, None] * rho[None, None, :]).reshape(m.n, -1)
        Y[:, : acc.shape[1]] += acc
        self.residual_force = F_res
        self._residual_power = float(self.q_dot_hat @ F_res)
        return phi_hat, M, Y, gain, cache

    # ------------------------------------------------------------------ blend
    def _refresh_stacks(self) -> None:
        """Cache Omega and the cross terms of every stack (recomputed only when a stack changes)."""
        key = tuple(st.version for st in self.stacks)
        if key == self._hkey:
            return
        self._hkey = key
        n_min = self.config.cl_min_windows
        st = self.stack_A
        self._A = (st.information_matrix(), st.cross_term()) if len(st) >= n_min else None
        self._U = []
        for u in self.unact:
            if len(u.stack) < n_min:
                self._U.append(None)
                continue
            R = u.stack.data_matrix()
            self._U.append((R.T @ R, R.T @ u.stack.targets()))

    def information_weights(self) -> Dict[str, np.ndarray]:
        """Per stack: eigenvalues of W_s = Omega (Omega + lambda I)^{-1}, the share given to integral CL."""
        self._refresh_stacks()
        lam, out = self.config.lambda_damp, {}
        if self._A is not None:
            w = np.linalg.eigvalsh(self._A[0])
            out["actuated"] = w / (w + lam)
        phi = self.theta / self.s
        for u, cached in zip(self.unact, self._U):
            if cached is None:
                continue
            T = self._reduction(u, phi)
            w = np.linalg.eigvalsh(T.T @ cached[0] @ T)
            out[f"row {u.row} (normalized)"] = w / (w + lam)
        return out

    @staticmethod
    def _reduction(u, phi: np.ndarray) -> np.ndarray:
        """T with X = R T: column 0 combines the shared columns with the current phi, the rest select the own columns."""
        T = np.zeros((u.cols.size, 1 + u.own.size))
        T[u.shared, 0] = phi[u.shared_cols]
        T[u.own, 1 + np.arange(u.own.size)] = 1.0
        return T

    def _blended_update(self, phi: np.ndarray, grad: np.ndarray, dt: float) -> np.ndarray:
        cfg = self.config
        lam = cfg.lambda_damp
        g = dt * cfg.gamma_cl
        step = g / (1.0 + g)
        self._refresh_stacks()
        cA = self.cols_A
        d_A = None
        if self._A is not None:
            d_A, W_w = damped_step(self._A[0], self._A[1], phi[cA], lam)
            grad[cA] = W_w @ grad[cA]
        increments = []
        for u, cached in zip(self.unact, self._U):
            if cached is None or phi[u.col_J] <= 0.0:
                continue
            own = u.cols[u.own]
            idx = np.concatenate(([u.col_J], own))
            T = self._reduction(u, phi)
            kappa = 1.0 / phi[u.col_J]
            sol = np.concatenate(([kappa], phi[own] * kappa))
            d_sol, W_w = damped_step(T.T @ cached[0] @ T, T.T @ cached[1], sol, lam)
            # phi = (1 / kappa, psi / kappa): map the weak-direction projector to phi coordinates
            J = np.eye(sol.size) / kappa
            J[0, 0] = -1.0 / kappa ** 2
            J[1:, 0] = -sol[1:] / kappa ** 2
            grad[idx] = J @ W_w @ np.linalg.solve(J, grad[idx])
            increments.append((u, own, d_sol))
        phi = phi - dt * cfg.gamma_inst * grad                      # instantaneous law, weak directions
        if d_A is not None:
            phi[cA] = phi[cA] + step * d_A                          # integral CL, identified directions
        for u, own, d_sol in increments:
            if phi[u.col_J] <= 0.0:
                continue
            kappa = 1.0 / phi[u.col_J]
            kappa_new = kappa + step * d_sol[0]
            if kappa_new > 0.0 and self.s[u.col_J] / kappa_new > cfg.eps_M:
                phi[own] = (phi[own] * kappa + step * d_sol[1:]) / kappa_new
                phi[u.col_J] = 1.0 / kappa_new
        return phi

    # ------------------------------------------------------------------ one step
    def _euler_step(self, y: np.ndarray, u: np.ndarray, dt: float) -> None:
        cfg = self.config
        a, k_r = cfg.alpha, cfg.k_r
        q_tilde = y - self.q_hat
        eta = self.p_f - (a + k_r) * q_tilde
        e = q_tilde + self.nu
        phi_hat, M, Y, gain, cache = self._model_terms(u)
        chi = -(3.0 * a + k_r) * eta + self.chi_x1_coeff * q_tilde - self.nu
        robust = cfg.k_s * (np.tanh(e / cfg.tanh_eps) if cfg.sign_mode == "tanh" else np.sign(e))

        d_qd = phi_hat + robust + chi
        d_p = -(k_r + 2.0 * a) * self.p_f - self.nu + ((a + k_r) ** 2 + 1.0) * q_tilde
        d_nu = self.p_f - a * self.nu - (a + k_r) * q_tilde
        self.q_hat = self.q_hat + dt * self.q_dot_hat
        self.q_dot_hat = self.q_dot_hat + dt * d_qd
        self.p_f = self.p_f + dt * d_p
        self.nu = self.nu + dt * d_nu

        if cache is not None:
            if cfg.adapt:
                # kinetic metric: Phi'_F^T M_hat e = (d phi_F / d theta_F)^T (-e * dF_res/dphi_F)
                self.theta_res = self.theta_res + dt * self.res_law.theta_dot(self.theta_res, cache, -e * gain)
                self.res_law.enforce_bound(self.theta_res)
            self.c_res = self.c_res + dt * cfg.b_c * (cache.c - self.c_res)
            self.h_res = self.h_res + dt * cfg.b_h * (cache.h - self.h_res)

        if cfg.adapt:
            phi = self.theta / self.s
            grad = self.s * (Y.T @ e)                               # gradient of the kinetic metric
            grad[self.frozen] = 0.0
            if cfg.cl_enabled and cfg.blend:
                phi = self._blended_update(phi, grad, dt)
            else:
                phi = phi - dt * cfg.gamma_inst * grad
                if cfg.cl_enabled:
                    phi_H, mask = self.stack_estimate()             # Approach D: Newton CL behind a rank gate
                    if phi_H is not None:
                        g = dt * cfg.gamma_cl
                        phi[mask] = (phi[mask] + g * phi_H[mask]) / (1.0 + g)
            self.theta = phi * self.s
            self.theta[self.frozen] = 0.0
            self._project()
        self.phi_hat, self.M_hat, self.e = phi_hat, M, e


__all__ = ["HybridObserver", "HybridObserverConfig", "FrozenHybridModel", "damped_step", "LinearELModel"]
