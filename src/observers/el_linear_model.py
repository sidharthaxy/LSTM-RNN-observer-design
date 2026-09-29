"""
Linear-in-parameters Euler-Lagrange model for Approach D (physics-structured integral CL).

Same structure and feature layer as Approach B (src/observers/pilstm_network.py), with every
sub-model LINEAR in its parameters, so that concurrent learning's rank condition applies to
the whole parameter vector:

    M(q) q_ddot + C(q, q_dot) q_dot + G(q) + F(q_dot) = B u,

    M(q; theta_M)  = sum_{e=(i<=j)} sum_k theta_{e,k} rho_k(q) E_e,   E_e = e_i e_j^T + e_j e_i^T (i != j), e_i e_i^T
    C q_dot        = M_dot q_dot - 1/2 d/dq (q_dot^T M q_dot)          (Christoffel symbols of M)
    G(q; w_G)      = d P / d q,   P(q) = sum_{k >= 1} w_{G,k} rho_k(q)   (conservative)
    F(q_dot)       = diag(d_v) q_dot + diag(d_c) tanh(q_dot / v_c),   d_v, d_c >= 0 (dissipative)

rho(q) is Approach B's fixed feature layer [1, cos k q_i, sin k q_i, ..., s_i (q_i - mu_i)]
(harmonics of revolute joints; cyclic coordinates are not embedded, so M and P do not depend on
them). The constant feature is left out of the potential: it has zero gradient and would be an
unidentifiable parameter.

Invariants:
    by construction, for every theta: M = M^T; M_dot - 2 C skew-symmetric (Christoffel);
                                     G conservative; power balance d/dt(T + P) = q_dot^T (B u - F)
    by projection (observer):        M(q) >= eps_M I on the revolute-angle grid; d_v, d_c >= 0

Regressors (all n x p, linear in theta):
    Y(q, q_dot, a)      theta = M a + C q_dot + G + F            (torque; instantaneous law)
    Y_rest(q, q_dot)    theta = C q_dot + G + F                  (= Y(q, q_dot, 0))
    Y_mom(q, q_dot)     theta = M q_dot                          (generalized momentum)
    Y_int(q, q_dot)     theta = -1/2 d/dq(q_dot^T M q_dot) + G + F
Integral form of the dynamics (the ICL identity), exact for the true parameters:
    [Y_mom theta]_{t-D}^{t} + int_{t-D}^{t} Y_int theta dtau = int_{t-D}^{t} B u dtau,
because d/dt(M q_dot) - 1/2 d/dq(q_dot^T M q_dot) = M q_ddot + C q_dot. It needs positions and
velocities only, never the acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple
import numpy as np

from src.observers.pilstm_network import ConfigurationFeatures
from src.plant.pendulum_plant import PendulumParameters


@dataclass(frozen=True)
class LinearELLayout:
    """theta = [vec(W_M) (P x R, column-major), w_G (P-1), d_v (n), d_c (n or 0)]."""
    n: int
    n_features: int         # P
    coulomb: bool

    @property
    def n_entries(self) -> int:
        return self.n * (self.n + 1) // 2

    @property
    def sizes(self) -> Dict[str, int]:
        return {"M": self.n_features * self.n_entries, "G": self.n_features - 1,
                "Fv": self.n, "Fc": self.n if self.coulomb else 0}

    def block_slice(self, name: str) -> slice:
        start = 0
        for b, size in self.sizes.items():
            if b == name:
                return slice(start, start + size)
            start += size
        raise KeyError(name)

    @property
    def n_params(self) -> int:
        return sum(self.sizes.values())

    def W_M(self, theta: np.ndarray) -> np.ndarray:
        return theta[self.block_slice("M")].reshape((self.n_features, self.n_entries), order="F")


class LinearELModel:
    """Linear-in-parameters Euler-Lagrange acceleration model with analytic regressors."""

    def __init__(
        self,
        n_coords: int = 2,
        input_matrix: np.ndarray | None = None,
        revolute: Sequence[bool] = (False, True),
        cyclic: Sequence[bool] = (True, False),
        harmonics: int = 2,
        n_features: int = 0,
        feature_scale: float = 1.0,
        position_scale: Sequence[float] | None = None,
        position_offset: Sequence[float] | None = None,
        coulomb: bool = True,
        coulomb_velocity: Sequence[float] = (0.01, 0.05),
        seed: int = 0,
    ) -> None:
        n = n_coords
        self.n = n
        self.B = np.array([[1.0]] + [[0.0]] * (n - 1)) if input_matrix is None else np.atleast_2d(
            np.asarray(input_matrix, dtype=np.float64))
        self.revolute = np.asarray(revolute, dtype=bool)
        self.cyclic = np.asarray(cyclic, dtype=bool)
        if self.revolute.size != n or self.cyclic.size != n or self.cyclic.all():
            raise ValueError("revolute / cyclic need n entries, not all cyclic.")
        self.features = ConfigurationFeatures(
            revolute, n_features, feature_scale, position_scale, position_offset,
            linear_skip=True, active=~self.cyclic, harmonics=harmonics, seed=seed + 1000,
        )
        self.harmonics = harmonics
        self.v_c = np.asarray(coulomb_velocity, dtype=np.float64)
        self.layout = LinearELLayout(n, self.features.dim, coulomb)
        self.entries = [(i, j) for i in range(n) for j in range(i, n)]   # (i <= j), column order of W_M
        self._ei = np.array([i for i, _ in self.entries])
        self._ej = np.array([j for _, j in self.entries])
        self._off = self._ei != self._ej

    # ------------------------------------------------------------------ building blocks
    def rho(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.features(q)

    def _E_times(self, v: np.ndarray) -> np.ndarray:
        """(n, R): column e = E_e v."""
        out = np.zeros((self.n, len(self.entries)))
        r = np.arange(len(self.entries))
        out[self._ei, r] += v[self._ej]
        out[self._ej[self._off], r[self._off]] += v[self._ei[self._off]]
        return out

    def _quad(self, v: np.ndarray) -> np.ndarray:
        """(R,): v^T E_e v."""
        return np.where(self._off, 2.0, 1.0) * v[self._ei] * v[self._ej]

    def inertia(self, theta: np.ndarray, q: np.ndarray, rho: np.ndarray | None = None) -> np.ndarray:
        if rho is None:
            rho, _ = self.rho(q)
        m = self.layout.W_M(theta).T @ rho          # (R,)
        M = np.zeros((self.n, self.n))
        M[self._ei, self._ej] = m
        M[self._ej, self._ei] = m
        return M

    def inertia_derivative(self, theta: np.ndarray, q: np.ndarray) -> np.ndarray:
        """dM[k] = dM / dq_k, shape (n, n, n)."""
        _, drho = self.rho(q)
        dm = self.layout.W_M(theta).T @ drho       # (R, n)
        dM = np.zeros((self.n, self.n, self.n))
        dM[:, self._ei, self._ej] = dm.T
        dM[:, self._ej, self._ei] = dm.T
        return dM

    # ------------------------------------------------------------------ regressors
    def regressors(self, q: np.ndarray, q_dot: np.ndarray, a: np.ndarray | None = None) -> Dict[str, np.ndarray]:
        """
        Blocks of the regressors (n x block size): "acc" (M a), "cor" (C q_dot), "mom" (M q_dot),
        "kin" (-1/2 d/dq q_dot^T M q_dot), "G", "Fv", "Fc".
        """
        rho, drho = self.rho(q)
        n, P = self.n, rho.size
        R = len(self.entries)
        Eq = self._E_times(q_dot)                   # (n, R)
        quad = self._quad(q_dot)                    # (R,)
        rdot = drho @ q_dot                         # (P,)  d rho / dt
        kin = -0.5 * drho.T[:, None, :] * quad[None, :, None]          # (n, R, P)
        cor = Eq[:, :, None] * rdot[None, None, :] + kin                # M_dot q_dot + kin
        mom = Eq[:, :, None] * rho[None, None, :]
        out = {
            "cor": cor.reshape(n, R * P),
            "mom": mom.reshape(n, R * P),
            "kin": kin.reshape(n, R * P),
            "G": drho[1:, :].T.copy(),
            "Fv": np.diag(q_dot),
            "Fc": np.diag(np.tanh(q_dot / self.v_c)) if self.layout.coulomb else np.zeros((n, 0)),
        }
        if a is not None:
            out["acc"] = (self._E_times(a)[:, :, None] * rho[None, None, :]).reshape(n, R * P)
        return out

    def _assemble(self, M_part: np.ndarray, blocks: Dict[str, np.ndarray], with_rest: bool) -> np.ndarray:
        n = self.n
        parts = [M_part]
        if with_rest:
            parts += [blocks["G"], blocks["Fv"], blocks["Fc"]]
        else:
            parts += [np.zeros((n, self.layout.sizes[b])) for b in ("G", "Fv", "Fc")]
        return np.hstack(parts)

    def torque_regressor(self, q: np.ndarray, q_dot: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Y(q, q_dot, a): Y theta = M a + C q_dot + G + F."""
        b = self.regressors(q, q_dot, a)
        return self._assemble(b["acc"] + b["cor"], b, True)

    def rest_regressor(self, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        """Y_rest(q, q_dot): Y_rest theta = C q_dot + G + F."""
        b = self.regressors(q, q_dot)
        return self._assemble(b["cor"], b, True)

    def momentum_regressor(self, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        """Y_mom theta = M(q) q_dot."""
        b = self.regressors(q, q_dot)
        return self._assemble(b["mom"], b, False)

    def integrand_regressor(self, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        """Y_int theta = -1/2 d/dq(q_dot^T M q_dot) + G + F."""
        b = self.regressors(q, q_dot)
        return self._assemble(b["kin"], b, True)

    def icl_regressors(self, q: np.ndarray, q_dot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(Y_mom, Y_int) from one feature evaluation."""
        b = self.regressors(q, q_dot)
        return self._assemble(b["mom"], b, False), self._assemble(b["kin"], b, True)

    # ------------------------------------------------------------------ model outputs
    def acceleration(self, theta: np.ndarray, q: np.ndarray, q_dot: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(Phi_hat, M_hat): Phi_hat = M^{-1} (B u - Y_rest theta)."""
        rho, _ = self.rho(q)
        M = self.inertia(theta, q, rho)
        tau = self.B @ np.atleast_1d(u) - self.rest_regressor(q, q_dot) @ theta
        return np.linalg.solve(M, tau), M

    def observer_terms(
        self, theta: np.ndarray, q: np.ndarray, q_dot: np.ndarray, u: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        (Phi_hat, M_hat, Y(q, q_dot, Phi_hat)) from one feature evaluation. The torque regressor at
        a = Phi_hat gives the Jacobian d Phi_hat / d theta = -M_hat^{-1} Y(q, q_dot, Phi_hat).
        """
        b = self.regressors(q, q_dot)
        rho, _ = self.rho(q)
        M = self.inertia(theta, q, rho)
        Y = self._assemble(b["cor"], b, True)
        phi = np.linalg.solve(M, self.B @ np.atleast_1d(u) - Y @ theta)
        acc = (self._E_times(phi)[:, :, None] * rho[None, None, :]).reshape(self.n, -1)
        Y[:, : acc.shape[1]] += acc
        return phi, M, Y

    def conservative_acceleration(self, theta: np.ndarray, q: np.ndarray, q_dot: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Rigid-body part only (friction removed): M^{-1}(B u - C q_dot - G)."""
        th = theta.copy()
        th[self.layout.block_slice("Fv")] = 0.0
        th[self.layout.block_slice("Fc")] = 0.0
        return self.acceleration(th, q, q_dot, u)[0]

    def kinetic_energy(self, theta: np.ndarray, q: np.ndarray, q_dot: np.ndarray) -> float:
        return float(0.5 * q_dot @ self.inertia(theta, q) @ q_dot)

    def potential_energy(self, theta: np.ndarray, q: np.ndarray) -> float:
        rho, _ = self.rho(q)
        return float(theta[self.layout.block_slice("G")] @ rho[1:])

    def skew_residual(self, theta: np.ndarray, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        """N = M_dot - 2 C with C the Christoffel matrix; skew-symmetric by construction."""
        dM = self.inertia_derivative(theta, q)
        M_dot = np.einsum("kij,k->ij", dM, q_dot)
        t2 = np.einsum("jik,k->ij", dM, q_dot)
        t3 = np.einsum("ijk,k->ij", dM, q_dot)
        C = 0.5 * (M_dot + t2 - t3)
        return M_dot - 2.0 * C

    # ------------------------------------------------------------------ parameters
    def prior_parameters(self, inertia_init: Sequence[float]) -> np.ndarray:
        """M(q) = diag(inertia_init) for all q, no potential, no friction (Approach B's prior)."""
        theta = np.zeros(self.layout.n_params)
        W_M = self.layout.W_M(theta)
        for e, (i, j) in enumerate(self.entries):
            if i == j:
                W_M[0, e] = inertia_init[i]
        return theta

    def parameter_scale(self, inertia_init: Sequence[float], gravity_scale: float = 5.0, friction_rate: float = 0.1) -> np.ndarray:
        """
        Per-parameter unit scale s (theta = s * phi with phi = O(1)): sqrt(M0_ii M0_jj) for
        inertia entries, M0_nn * gravity_scale for the potential (gravity_scale ~ omega^2 [1/s^2]),
        M0_ii * friction_rate for the friction coefficients.
        """
        m0 = np.asarray(inertia_init, dtype=np.float64)
        lay = self.layout
        s = np.zeros(lay.n_params)
        s_M = lay.W_M(s)
        for e, (i, j) in enumerate(self.entries):
            s_M[:, e] = np.sqrt(m0[i] * m0[j])
        s[lay.block_slice("G")] = m0[~self.cyclic].max() * gravity_scale
        s[lay.block_slice("Fv")] = m0 * friction_rate
        if lay.coulomb:
            s[lay.block_slice("Fc")] = m0 * friction_rate
        return s

    def _cart_pendulum_indices(self) -> Tuple[int, int]:
        if self.n != 2 or not self.cyclic[0] or not self.revolute[1] or self.harmonics < 1:
            raise NotImplementedError("Physical mapping implemented for the cart-pendulum layout only.")
        return 0, 1   # rho index of the constant and of cos(theta)

    def true_parameters(self, p: PendulumParameters) -> np.ndarray:
        """theta* realizing the Feedback 33-936S equations of motion exactly."""
        k0, kc = self._cart_pendulum_indices()
        theta = np.zeros(self.layout.n_params)
        W_M = self.layout.W_M(theta)
        W_M[k0, 0] = p.M + p.m                   # m11
        W_M[kc, 1] = p.m * p.l                   # m12 = m l cos(theta)
        W_M[k0, 2] = p.I + p.m * p.l ** 2        # m22
        theta[self.layout.block_slice("G")][kc - 1] = p.m * p.g * p.l   # P = m g l cos(theta)
        theta[self.layout.block_slice("Fv")] = [p.b, p.d]
        return theta

    def physical_parameters(self, theta: np.ndarray) -> Dict[str, float]:
        """Physical reading of theta (cart-pendulum): the coefficients a rigid cart-pendulum would have."""
        k0, kc = self._cart_pendulum_indices()
        W_M = self.layout.W_M(theta)
        wG = theta[self.layout.block_slice("G")]
        dv = theta[self.layout.block_slice("Fv")]
        out = {
            "M+m": float(W_M[k0, 0]), "ml": float(W_M[kc, 1]), "I+ml^2": float(W_M[k0, 2]),
            "mgl": float(wG[kc - 1]), "b": float(dv[0]), "d": float(dv[1]),
        }
        return out

    @staticmethod
    def physical_truth(p: PendulumParameters) -> Dict[str, float]:
        return {"M+m": p.M + p.m, "ml": p.m * p.l, "I+ml^2": p.I + p.m * p.l ** 2,
                "mgl": p.m * p.g * p.l, "b": p.b, "d": p.d}

    # ------------------------------------------------------------------ positive-definiteness
    def angle_grid(self, n_grid: int = 72) -> np.ndarray:
        """Configurations covering the non-cyclic revolute coordinate (one such coordinate)."""
        idx = np.flatnonzero(~self.cyclic & self.revolute)
        if idx.size != 1 or np.any(~self.cyclic & ~self.revolute):
            raise NotImplementedError("PD grid implemented for one non-cyclic revolute coordinate.")
        grid = np.zeros((n_grid, self.n))
        grid[:, idx[0]] = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
        return grid

    def grid_features(self, grid: np.ndarray) -> np.ndarray:
        return np.vstack([self.rho(q)[0] for q in grid])

    def min_inertia_eigenvalue(self, theta: np.ndarray, rho_grid: np.ndarray) -> Tuple[float, int, np.ndarray]:
        """(min_g lambda_min(M(q_g)), argmin g, unit eigenvector) over a feature grid (G, P)."""
        m = rho_grid @ self.layout.W_M(theta)                     # (G, R)
        Ms = np.zeros((len(m), self.n, self.n))
        Ms[:, self._ei, self._ej] = m
        Ms[:, self._ej, self._ei] = m
        w, v = np.linalg.eigh(Ms)
        g = int(np.argmin(w[:, 0]))
        return float(w[g, 0]), g, v[g, :, 0]

    def project_positive_inertia(
        self, theta: np.ndarray, rho_grid: np.ndarray, eps_M: float, metric: np.ndarray, max_iter: int = 20,
    ) -> int:
        """
        In-place projection onto {theta : M(q_g; theta) >= eps_M I on the grid}, a convex set
        (intersection of LMIs, each linear in theta). Each iteration projects, in the metric
        diag(metric), onto the supporting half-space of the most violated constraint
        v^T M(q_g) v >= eps_M (v = its minimum eigenvector). Returns the iteration count.
        """
        sl = self.layout.block_slice("M")
        P, R = self.layout.n_features, len(self.entries)
        for it in range(max_iter):
            lam, g, v = self.min_inertia_eigenvalue(theta, rho_grid)
            if lam >= eps_M:
                return it
            grad = (self._quad(v)[None, :] * rho_grid[g][:, None]).ravel(order="F")   # d(v^T M v)/d vec(W_M)
            w = metric[sl] * grad
            theta[sl] += (eps_M - lam) * 1.0001 * w / float(grad @ w)
        return max_iter
