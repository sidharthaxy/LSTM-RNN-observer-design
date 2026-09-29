"""
Approach D: physics-structured Lyapunov-based observer with integral concurrent learning
(PI-ICL).

Observer (Approach A/B auxiliary filter and feedback, unchanged):
    q_hat_dot     = q_dot_hat
    q_dot_hat_dot = Phi_hat + k_s sgn(e) + chi,   Phi_hat = M_hat^{-1}(B u - Y_rest(q_hat, q_dot_hat) theta_hat)
with the linear-in-parameters Euler-Lagrange model of `el_linear_model.py` (Approach B's
structure and features; every sub-model linear in theta).

Adaptation, in scaled parameters phi = theta / s (s: the unit scale of each parameter):
    phi_dot = -gamma s * Y(q_hat, q_dot_hat, Phi_hat)^T e      (instantaneous; Approach B's kinetic metric)
              + gamma_CL (phi_H - phi)                         (integral concurrent learning)
followed by the projections d_v, d_c >= 0; M(q) >= eps_M I on the revolute-angle grid;
||phi - phi_0|| <= W_bar. The CL term is integrated implicitly (exactly stable for any gain).

Integral concurrent learning (Parikh, Kamalapurkar & Dixon, 2019). Integrating the equations of
motion over a window [t - D, t] removes the acceleration: Y_j theta* = b_j, with
    Y_j = [Y_mom]_{t-D}^{t} + int Y_int dtau,   b_j = int B u dtau,
built from a causal delayed-centre Savitzky-Golay smoother of the encoder data (model-free, so the
stack is usable before the observer has converged).

Two-part stack estimate phi_H (the CL target):
  * Actuated rows (B_i != 0): ordinary least squares on the columns that appear in those rows,
        b_A = Y_A phi_A,   stack A, rank condition lambda_min(Omega_A) > 0.
    The target int B u is noise-free.
  * Each unactuated row i (B_i = 0) is homogeneous, Y_i theta* = 0. Least squares on it is biased
    towards theta_i = 0 by the noise in its velocity regressors (errors-in-variables shrinkage of the
    row scale). It is therefore normalized by its leading inertia coefficient J_i = [M_ii]_const,
    and the columns it shares with the actuated rows (the coupling inertia) are taken from phi_A,
    which fixes the scale:
        kappa z + sum_{c own} (Y_ic s_c) psi_c = -Y_iJ s_J,
        z = sum_{c shared} Y_ic s_c phi_A,c,   kappa = 1 / phi_J,   psi_c = phi_c / phi_J.
    The noisy momentum term Y_iJ (Delta q_dot_i times the feature) is the TARGET; the regressors are
    driven by the other coordinates and by positions and carry little noise. Stack U_i (it stores
    the full row so that z can be re-formed whenever phi_A changes); rank condition on [z, Y_own].
  * phi_H = phi_A on actuated columns, phi_J = 1/kappa on J, phi_J psi_c on the row's own columns.
  gamma_CL (phi_H - phi) is the Newton (information-normalized) form of the classical CL term
  Gamma_CL sum_j Y_j^T (b_j - Y_j phi) = Gamma_CL Omega (phi_LS - phi).

Change detection. A plant change makes new windows inconsistent with the stored ones: the
normalized residual of each new window ||Lambda^{1/2}(b - Y theta)||^2 is averaged (EMA) and
compared with the mean residual of the stored windows; if the ratio stays above `detect_ratio`
for `detect_hold` seconds (outside a refractory period), all stacks are purged and refill with
post-change data.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Literal, Tuple
import numpy as np

from src.concurrent_learning.history_stack import HistoryStack, HistoryStackEntry
from src.concurrent_learning.savitzky_golay import SavitzkyGolay
from src.observers.el_linear_model import LinearELModel


@dataclass
class PIICLObserverConfig:
    # Structure (Approach B defaults)
    n_coords: int = 2
    input_matrix: Tuple[Tuple[float, ...], ...] = ((1.0,), (0.0,))
    revolute: Tuple[bool, ...] = (False, True)
    cyclic: Tuple[bool, ...] = (True, False)
    harmonics: int = 1               # K = 1 represents a rigid pendulum exactly; K = 2 is not identifiable
    n_features: int = 0              # from partial swings under encoder noise (see README, Section 5)
    coulomb: bool = True
    coulomb_velocity: Tuple[float, ...] = (0.01, 0.05)
    inertia_init: Tuple[float, ...] = (2.0, 0.2)     # prior M = diag(inertia_init) (as B)
    eps_M: float = 0.05
    pd_grid: int = 72

    # Auxiliary filter and robust feedback (Approach A/B)
    alpha: float = 4.0
    k_r: float = 8.0
    k_s: float = 0.2
    sign_mode: Literal["sgn", "tanh"] = "sgn"
    tanh_eps: float = 0.01
    chi_x1_coeff: float | None = None

    # Adaptation (scaled parameters phi = theta / s)
    gamma_inst: float = 2.0
    adapt: bool = True
    w_bar: float = 20.0

    # Integral concurrent learning
    cl_enabled: bool = True
    gamma_cl: float = 2.0             # [1/s] rate of the pull towards the stack estimate
    icl_window: float = 0.25          # D [s]
    record_interval: float = 0.05     # [s] candidate spacing
    stack_capacity: int = 300         # windows per stack
    cl_min_windows: int = 60          # windows a stack needs before its estimate is used
    lambda_bar: float = 1e-4          # rank condition with margin: lambda_min(Omega) >= lambda_bar
    novelty_tol: float = 0.05
    min_rel_improvement: float = 1e-3
    sg_window: float = 0.1            # [s] smoother window
    sg_order: int = 3

    # Change detection (stack purge)
    detect_changes: bool = True
    detect_ratio: float = 8.0
    detect_tau: float = 0.5           # [s] EMA time constant of the new-window residual
    detect_hold: float = 0.3          # [s]
    detect_refractory: float = 3.0    # [s] after a purge
    seed: int = 0


@dataclass
class _StreamSample:
    Y_mom: np.ndarray       # (n, p) raw
    cum_int: np.ndarray     # (n, p) raw int_0^t Y_int
    cum_bu: np.ndarray      # (n,)   int_0^t B u
    t: float


@dataclass
class _UnactuatedRow:
    row: int
    col_J: int              # leading inertia coefficient (constant feature of M_ii)
    cols: np.ndarray        # other columns present in the row
    shared: np.ndarray      # positions (within cols) of columns also estimated by the actuated stack
    shared_cols: np.ndarray
    own: np.ndarray         # positions (within cols) of the row's own columns
    shared_in_A: np.ndarray # positions (within cols_A) of the shared columns
    stack: HistoryStack


class PIICLObserver:
    """Approach D observer. Interface of the other observers: reset(x0), update(y, u, dt)."""

    def __init__(self, config: PIICLObserverConfig | None = None) -> None:
        self.config = config or PIICLObserverConfig()
        cfg = self.config
        n = cfg.n_coords
        self.model = LinearELModel(
            n, np.asarray(cfg.input_matrix, dtype=np.float64), cfg.revolute, cfg.cyclic, cfg.harmonics,
            cfg.n_features, coulomb=cfg.coulomb, coulomb_velocity=cfg.coulomb_velocity, seed=cfg.seed,
        )
        lay = self.model.layout
        self.p = lay.n_params
        self.s = self.model.parameter_scale(cfg.inertia_init)
        self.row_weight = 1.0 / np.sqrt(np.asarray(cfg.inertia_init, dtype=np.float64))   # Lambda^{1/2}
        self.theta0 = self.model.prior_parameters(cfg.inertia_init)
        self.phi0 = self.theta0 / self.s
        self.rho_grid = self.model.grid_features(self.model.angle_grid(cfg.pd_grid))
        self.metric = self.s ** 2
        self.nonneg = np.zeros(self.p, dtype=bool)
        self.nonneg[lay.block_slice("Fv")] = True
        self.nonneg[lay.block_slice("Fc")] = True
        self.chi_x1_coeff = 2.0 - cfg.alpha ** 2 if cfg.chi_x1_coeff is None else cfg.chi_x1_coeff
        self._build_stacks()
        self.theta = self.theta0.copy()
        self.reset()

    # ------------------------------------------------------------------ stack structure
    def _structural_support(self) -> np.ndarray:
        """(n, p) boolean: column c can be nonzero in row i of the ICL regressor."""
        rng = np.random.default_rng(12345)
        support = np.zeros((self.model.n, self.p), dtype=bool)
        for _ in range(8):
            q = rng.uniform(-3.0, 3.0, self.model.n)
            qd = rng.normal(size=self.model.n)
            Ym, Yi = self.model.icl_regressors(q, qd)
            support |= (np.abs(Ym) > 1e-12) | (np.abs(Yi) > 1e-12)
        return support

    def _build_stacks(self) -> None:
        cfg = self.config
        support = self._structural_support()
        actuated = np.any(np.abs(self.model.B) > 0.0, axis=1)
        self.act_rows = np.flatnonzero(actuated)
        self.cols_A = np.flatnonzero(support[self.act_rows].any(axis=0))
        self.stack_A = HistoryStack(cfg.stack_capacity, self.cols_A.size, rows_per_entry=self.act_rows.size,
                                    novelty_tol=cfg.novelty_tol, min_rel_improvement=cfg.min_rel_improvement)
        lay, P = self.model.layout, self.model.layout.n_features
        self.unact: List[_UnactuatedRow] = []
        for i in np.flatnonzero(~actuated):
            e = self.model.entries.index((i, i))
            col_J = lay.block_slice("M").start + e * P          # constant feature of M_ii
            cols = np.array([c for c in np.flatnonzero(support[i]) if c != col_J])
            shared_mask = np.isin(cols, self.cols_A)
            if not shared_mask.any():
                raise ValueError(f"Unactuated row {i} shares no parameter with the actuated rows: scale unidentifiable.")
            self.unact.append(_UnactuatedRow(
                row=int(i), col_J=int(col_J), cols=cols, shared=np.flatnonzero(shared_mask),
                shared_cols=cols[shared_mask], own=np.flatnonzero(~shared_mask),
                shared_in_A=np.searchsorted(self.cols_A, cols[shared_mask]),
                stack=HistoryStack(cfg.stack_capacity, cols.size, rows_per_entry=1,
                                   novelty_tol=cfg.novelty_tol, min_rel_improvement=cfg.min_rel_improvement),
            ))

    @property
    def stacks(self) -> List[HistoryStack]:
        return [self.stack_A] + [u.stack for u in self.unact]

    # ------------------------------------------------------------------ state handling
    def reset(self, initial_state: np.ndarray | None = None, reset_weights: bool = True) -> None:
        n = self.config.n_coords
        if initial_state is None:
            self.q_hat, self.q_dot_hat = np.zeros(n), np.zeros(n)
        else:
            x0 = np.asarray(initial_state, dtype=np.float64).reshape(n, 2)
            self.q_hat, self.q_dot_hat = x0[:, 0].copy(), x0[:, 1].copy()
        self.p_f = np.zeros(n)
        self.nu = np.zeros(n)
        self.e = np.zeros(n)
        self.phi_hat = np.zeros(n)
        self.M_hat = np.diag(self.config.inertia_init).astype(np.float64)
        self._filter_initialized = False
        if reset_weights:
            self.theta = self.theta0.copy()
        for st in self.stacks:
            st.clear()
        self.t = 0.0
        self.smoother: SavitzkyGolay | None = None
        self._ybuf: Deque[np.ndarray] = deque()
        self._ubuf: Deque[np.ndarray] = deque()
        self._stream: Deque[_StreamSample] = deque()
        self._prev_int: np.ndarray | None = None
        self._prev_u: np.ndarray | None = None
        self._last_record = -np.inf
        self._target_key: Tuple[int, ...] | None = None
        self._phi_H: np.ndarray | None = None
        self._mask_H = np.zeros(self.p, dtype=bool)
        self._ema: float | None = None
        self._above_since: float | None = None
        self._last_purge = -np.inf
        self.purge_times: List[float] = []
        self.smoothed: Tuple[np.ndarray, np.ndarray] | None = None

    def state_estimate(self) -> np.ndarray:
        return np.column_stack((self.q_hat, self.q_dot_hat)).ravel()

    # ------------------------------------------------------------------ diagnostics
    @property
    def phi(self) -> np.ndarray:
        return self.theta / self.s

    def lambda_min(self) -> float:
        """min over the stacks of lambda_min(Omega) (the joint rank condition)."""
        return min(st.lambda_min() for st in self.stacks)

    def lambda_mins(self) -> Dict[str, float]:
        """lambda_min of the actuated stack and of each normalized, reduced unactuated regression."""
        out = {"actuated": self.stack_A.lambda_min()}
        phi_A = self.stack_A.least_squares()
        for u in self.unact:
            X = self._reduced_regressor(u, phi_A) if phi_A is not None else None
            out[f"row {u.row} (normalized)"] = 0.0 if X is None or X.shape[0] < X.shape[1] else \
                max(float(np.linalg.eigvalsh(X.T @ X)[0]), 0.0)
        return out

    def _reduced_regressor(self, u: _UnactuatedRow, phi_A: np.ndarray) -> np.ndarray | None:
        """[z, Y_own] of the stored windows of row u (z = shared columns combined with phi_A)."""
        if len(u.stack) == 0:
            return None
        R = u.stack.data_matrix()
        z = R[:, u.shared] @ phi_A[u.shared_in_A]
        return np.column_stack((z, R[:, u.own]))

    def physical_parameters(self) -> Dict[str, float]:
        return self.model.physical_parameters(self.theta)

    def inertia_estimate(self, q: np.ndarray | None = None) -> np.ndarray:
        return self.model.inertia(self.theta, self.q_hat if q is None else q)

    def stack_estimate(self) -> Tuple[np.ndarray | None, np.ndarray]:
        """(phi_H, mask): the CL target and which of its entries are determined."""
        key = tuple(st.version for st in self.stacks)
        if key == self._target_key:
            return self._phi_H, self._mask_H
        self._target_key = key
        phi_H = np.zeros(self.p)
        mask = np.zeros(self.p, dtype=bool)
        cfg = self.config
        phi_A = self.stack_A.least_squares()
        if phi_A is None or len(self.stack_A) < cfg.cl_min_windows or self.stack_A.lambda_min() < cfg.lambda_bar:
            self._phi_H, self._mask_H = None, mask
            return None, mask
        phi_H[self.cols_A] = phi_A
        mask[self.cols_A] = True
        for u in self.unact:
            X = self._reduced_regressor(u, phi_A)
            if X is None or X.shape[0] < max(X.shape[1], cfg.cl_min_windows):
                continue
            XtX = X.T @ X
            eig = np.linalg.eigvalsh(XtX)
            if eig[0] < cfg.lambda_bar:
                continue
            sol = np.linalg.solve(XtX, X.T @ u.stack.targets())
            kappa, psi_own = float(sol[0]), sol[1:]
            if kappa <= 0.0 or (1.0 / kappa) * self.s[u.col_J] <= self.config.eps_M:
                continue                                    # physically inadmissible scale: wait for data
            phi_J = 1.0 / kappa
            phi_H[u.col_J] = phi_J
            phi_H[u.cols[u.own]] = phi_J * psi_own
            mask[u.col_J] = True
            mask[u.cols[u.own]] = True
        self._phi_H, self._mask_H = phi_H, mask
        return phi_H, mask

    def stack_solution(self) -> np.ndarray | None:
        """theta_H (all parameters), or None while any stack is rank deficient."""
        phi_H, mask = self.stack_estimate()
        return None if phi_H is None or not mask.all() else phi_H * self.s

    # ------------------------------------------------------------------ adaptation
    def _project(self) -> None:
        cfg = self.config
        phi = self.theta / self.s
        d = phi - self.phi0
        nrm = float(np.linalg.norm(d))
        if nrm > cfg.w_bar:
            phi = self.phi0 + d * (cfg.w_bar / nrm)
        phi[self.nonneg] = np.maximum(phi[self.nonneg], 0.0)
        self.theta = phi * self.s
        self.model.project_positive_inertia(self.theta, self.rho_grid, cfg.eps_M, self.metric)

    def _euler_step(self, y: np.ndarray, u: np.ndarray, dt: float) -> None:
        cfg = self.config
        a, k_r = cfg.alpha, cfg.k_r
        q_tilde = y - self.q_hat
        eta = self.p_f - (a + k_r) * q_tilde
        e = q_tilde + self.nu
        phi_hat, M, Y = self.model.observer_terms(self.theta, self.q_hat, self.q_dot_hat, u)
        chi = -(3.0 * a + k_r) * eta + self.chi_x1_coeff * q_tilde - self.nu
        robust = cfg.k_s * (np.tanh(e / cfg.tanh_eps) if cfg.sign_mode == "tanh" else np.sign(e))

        d_qd = phi_hat + robust + chi
        d_p = -(k_r + 2.0 * a) * self.p_f - self.nu + ((a + k_r) ** 2 + 1.0) * q_tilde
        d_nu = self.p_f - a * self.nu - (a + k_r) * q_tilde
        self.q_hat = self.q_hat + dt * self.q_dot_hat
        self.q_dot_hat = self.q_dot_hat + dt * d_qd
        self.p_f = self.p_f + dt * d_p
        self.nu = self.nu + dt * d_nu

        if cfg.adapt:
            phi = self.theta / self.s
            phi = phi - dt * cfg.gamma_inst * self.s * (Y.T @ e)          # instantaneous, explicit
            if cfg.cl_enabled:
                phi_H, mask = self.stack_estimate()
                if phi_H is not None:
                    g = dt * cfg.gamma_cl
                    phi[mask] = (phi[mask] + g * phi_H[mask]) / (1.0 + g)  # implicit pull
            self.theta = phi * self.s
            self._project()
        self.phi_hat, self.M_hat, self.e = phi_hat, M, e

    # ------------------------------------------------------------------ integral-CL data stream
    def _push_stream(self, y: np.ndarray, u: np.ndarray, dt: float) -> None:
        cfg = self.config
        if self.smoother is None:
            self.smoother = SavitzkyGolay.from_duration(cfg.sg_window, cfg.sg_order, dt)
            self._ybuf = deque(maxlen=self.smoother.window)
            self._ubuf = deque(maxlen=self.smoother.delay + 1)
            self._stream = deque(maxlen=round(cfg.icl_window / dt) + 1)
        self._ybuf.append(y.copy())
        self._ubuf.append(u.copy())
        if len(self._ybuf) < self.smoother.window:
            return
        Yw = np.vstack(self._ybuf)
        q_s = self.smoother.apply(Yw, 0)
        qd_s = self.smoother.apply(Yw, 1)
        self.smoothed = (q_s, qd_s)
        u_c = self._ubuf[0]                                  # input at the centre sample
        t_c = self.t - self.smoother.delay * dt
        Y_mom, Y_int = self.model.icl_regressors(q_s, qd_s)
        if not self._stream:
            cum_int = np.zeros_like(Y_int)
            cum_bu = np.zeros(self.model.n)
        else:
            last = self._stream[-1]
            assert self._prev_int is not None and self._prev_u is not None
            cum_int = last.cum_int + 0.5 * dt * (self._prev_int + Y_int)           # trapezoid
            cum_bu = last.cum_bu + dt * (self.model.B @ self._prev_u)             # ZOH input
        self._prev_int, self._prev_u = Y_int, u_c
        self._stream.append(_StreamSample(Y_mom, cum_int, cum_bu, t_c))
        if len(self._stream) == self._stream.maxlen and t_c - self._last_record >= cfg.record_interval - 1e-12:
            self._record_window(t_c)

    def _record_window(self, t_c: float) -> None:
        cfg = self.config
        first, last = self._stream[0], self._stream[-1]
        Y = (last.Y_mom - first.Y_mom) + (last.cum_int - first.cum_int)    # raw window regressor
        b = last.cum_bu - first.cum_bu
        self._last_record = t_c
        rows = self.act_rows
        Y_A = (self.row_weight[rows, None] * Y[rows]) * self.s[None, :]
        Y_A, b_A = Y_A[:, self.cols_A], self.row_weight[rows] * b[rows]
        if cfg.detect_changes:
            self._detect_change(Y_A, b_A, t_c)
        self.stack_A.consider(HistoryStackEntry(Y_A, b_A, t_c))
        for u in self.unact:
            Yi = Y[u.row] * self.s
            u.stack.consider(HistoryStackEntry(Yi[u.cols][None, :], np.array([-Yi[u.col_J]]), t_c))

    def _detect_change(self, Y_A: np.ndarray, b_A: np.ndarray, t_c: float) -> None:
        """
        Prediction test on the actuated rows: residual of the fresh window under the actuated-stack
        model phi_A (EMA) vs the in-sample residual of the stored windows. The actuated rows have a
        noise-free target (int B u), and a change of the plant's inertia shows up in them even when
        the normalized unactuated rows are invariant to it (m and I scaled together).
        """
        cfg = self.config
        phi_A = self.stack_A.least_squares()
        if phi_A is None or len(self.stack_A) < cfg.cl_min_windows or t_c - self._last_purge < cfg.detect_refractory:
            self._above_since = None
            self._ema = None
            return
        r_new = float(np.sum((b_A - Y_A @ phi_A) ** 2))
        self._ema = r_new if self._ema is None else self._ema + (cfg.record_interval / cfg.detect_tau) * (r_new - self._ema)
        res = self.stack_A.targets() - self.stack_A.data_matrix() @ phi_A
        r_stack = float(np.sum(res ** 2)) / len(self.stack_A)
        if self._ema > cfg.detect_ratio * max(r_stack, 1e-12):
            if self._above_since is None:
                self._above_since = t_c
            elif t_c - self._above_since >= cfg.detect_hold:
                for st in self.stacks:
                    st.clear(purge=True)
                self.purge_times.append(t_c)
                self._last_purge = t_c
                self._above_since = None
                self._ema = None
        else:
            self._above_since = None

    # ------------------------------------------------------------------ main entry
    def update(self, y: np.ndarray, u: float | np.ndarray, dt: float) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float64)
        u_arr = np.atleast_1d(np.asarray(u, dtype=np.float64))
        if not self._filter_initialized:
            cfg = self.config
            self.p_f = (cfg.alpha + cfg.k_r) * (y_arr - self.q_hat)
            self.nu = np.zeros_like(self.p_f)
            self._filter_initialized = True
        self._euler_step(y_arr, u_arr, dt)
        if self.config.cl_enabled or self.config.detect_changes:
            self._push_stream(y_arr, u_arr, dt)
        self.t += dt
        return self.state_estimate()
