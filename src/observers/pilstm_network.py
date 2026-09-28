"""
Structured Euler-Lagrange sub-networks of the physics-informed Lb-LSTM (PI-LSTM)
(Hart, Griffis, Patil & Dixon, 2024).

The acceleration field of an n-DOF Euler-Lagrange system
    M(q) q_ddot + V_m(q, q_dot) q_dot + G(q) + F(q_dot) = B u
is approximated by four sub-networks whose *structure* enforces the mechanical invariants,
for every value of their parameters theta = [theta_M, theta_V, theta_G, theta_F]:

1. Inertia (theta_M = W_M), modified Cholesky factorization
       M_hat(q) = L(q) L(q)^T + eps_M I,   L = D L~ lower triangular,
       L~_ii = softplus(a_ii) > 0,  L~_ij = a_ij (i > j),  a = W_M^T rho(q),
   with a fixed positive row scale D = diag(d_i) (the units of coordinate i; d_i^2 is the
   prior inertia scale), so W_M is dimensionless and one adaptation gain suits every row.
   => M_hat = M_hat^T and lambda_min(M_hat) >= eps_M > 0 everywhere.

2. Centripetal-Coriolis (theta_V = W_V)
       V_m_hat = C_hat(q, q_dot; theta_M) + S(q, q_dot; theta_V),
   C_hat from the Christoffel symbols of the first kind of M_hat,
       C_ij = sum_k 1/2 (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i) q_dot_k,
   and S = sum_{i<j} s_ij (E_ij - E_ji) a skew-symmetric (gyroscopic, power-neutral) term with
   s_ij = W_V[:, ij]^T (rho(q) kron q_dot).
   => M_hat_dot - 2 V_m_hat = (M_hat_dot - 2 C_hat) - 2 S is skew-symmetric.

3. Gravity (theta_G = w_G): gradient of a learned potential energy
       P_hat(q) = w_G^T rho(q),   G_hat(q) = dP_hat/dq = Drho(q)^T w_G.
   => G_hat is conservative (curl-free); the learned model satisfies the power balance
      d/dt [1/2 q_dot^T M_hat q_dot + P_hat] = q_dot^T (B u - F_hat).

4. Friction (theta_F): continuous-time LSTM (same cell as Approach A) driven by q_dot, whose
   readout phi_F = W_h^T h sets a velocity- and history-dependent diagonal damping
       F_hat = D_hat q_dot,   D_hat = diag(d_i),   d_i = d0_i softplus(phi_F,i + beta_F) > 0.
   => q_dot^T F_hat = sum_i d_i q_dot_i^2 >= 0: the learned friction can only dissipate, so the
      whole learned model is passive, d/dt [T_hat + P_hat] <= q_dot^T B u. Stribeck and Coulomb
      laws are of this form (d_i(v) = F(v)/v); the LSTM memory makes d_i depend on the velocity
      history. `dissipative_friction=False` uses the unconstrained force F_hat = phi_F instead.

Cyclic coordinates (`cyclic[i] = True`) are absent from the learned Lagrangian: M_hat, S and
P_hat are invariant to them by construction (for the cart on a level track, x is cyclic, so
the conjugate momentum is conserved when B u = F = 0).

The configuration-dependent sub-networks share a fixed feature layer
    rho(q) = [1, xi(q), tanh(V_0^T xi(q) + b_0)],
    xi(q)  = [cos k q_i, sin k q_i]_{k=1..K}  (revolute joints: Fourier features on the circle),
             s_i (q_i - mu_i)                 (prismatic joints),
whose Jacobian Drho = d rho / dq is analytic. The harmonic features are orthogonal over a
revolution, which keeps the adaptation well conditioned; the optional random tanh layer adds
capacity on top. Only the output layers adapt, which keeps the second derivatives
d^2 M_hat / dq d theta_M (needed by the Christoffel Jacobian) closed-form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Sequence, Tuple
import numpy as np

from src.adaptation.jacobian_engine import LSTMCache, LSTMWeightLayout, lstm_forward, sigmoid


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def softplus_inverse(y: float) -> float:
    """a such that softplus(a) = y, y > 0."""
    return float(y + np.log(-np.expm1(-y)))


# ---------------------------------------------------------------------------- features
class ConfigurationFeatures:
    """
    Fixed feature layer rho(q) = [1, xi(q), tanh(V_0^T xi(q) + b_0)] in R^P and its analytic
    Jacobian Drho(q) = d rho / dq in R^{P x n}. xi(q) holds K harmonics [cos k q_i, sin k q_i]
    per revolute joint and s_i (q_i - mu_i) per prismatic joint. The skip connection keeps
    xi(q) directly available to every readout; `linear_skip=False` drops it (then the random
    layer must have n_features > 0). Coordinates with `active[i] = False` are not embedded,
    so rho is invariant to them (Drho[:, i] = 0).
    """

    def __init__(
        self,
        revolute: Sequence[bool],
        n_features: int,
        weight_scale: float = 1.0,
        position_scale: Sequence[float] | None = None,
        position_offset: Sequence[float] | None = None,
        linear_skip: bool = True,
        active: Sequence[bool] | np.ndarray | None = None,
        harmonics: int = 1,
        seed: int = 0,
    ) -> None:
        if harmonics < 1:
            raise ValueError("harmonics must be >= 1.")
        self.revolute = np.asarray(revolute, dtype=bool)
        self.n = int(self.revolute.size)
        self.active = np.ones(self.n, dtype=bool) if active is None else np.asarray(active, dtype=bool)
        self.harmonics = harmonics
        self.xi_dim = int(np.sum(np.where(self.revolute, 2 * harmonics, 1)[self.active]))
        self.scale = np.ones(self.n) if position_scale is None else np.asarray(position_scale, dtype=np.float64)
        self.offset = np.zeros(self.n) if position_offset is None else np.asarray(position_offset, dtype=np.float64)
        rng = np.random.default_rng(seed)
        self.V0 = rng.normal(0.0, weight_scale, size=(self.xi_dim, n_features))
        self.b0 = rng.uniform(-1.0, 1.0, size=n_features)
        self.linear_skip = linear_skip
        self.dim = 1 + (self.xi_dim if linear_skip else 0) + n_features

    def embed(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """xi(q) and D xi = d xi / dq in R^{xi_dim x n}."""
        xi = np.zeros(self.xi_dim)
        dxi = np.zeros((self.xi_dim, self.n))
        row = 0
        for i in np.flatnonzero(self.active):
            if self.revolute[i]:
                for k in range(1, self.harmonics + 1):
                    c, s = np.cos(k * q[i]), np.sin(k * q[i])
                    xi[row], xi[row + 1] = c, s
                    dxi[row, i], dxi[row + 1, i] = -k * s, k * c
                    row += 2
            else:
                xi[row] = self.scale[i] * (q[i] - self.offset[i])
                dxi[row, i] = self.scale[i]
                row += 1
        return xi, dxi

    def __call__(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xi, dxi = self.embed(q)
        z = np.tanh(self.V0.T @ xi + self.b0)
        dz = (1.0 - z ** 2)[:, None] * (self.V0.T @ dxi)
        if self.linear_skip:
            return np.concatenate(((1.0,), xi, z)), np.vstack((np.zeros((1, self.n)), dxi, dz))
        return np.concatenate(((1.0,), z)), np.vstack((np.zeros((1, self.n)), dz))


# ---------------------------------------------------------------------------- layout
@dataclass(frozen=True)
class PILSTMLayout:
    """
    Index bookkeeping for theta = [vec(W_M), vec(W_V), w_G, theta_F] (column-major vecs).
        W_M in R^{P x R},       R = n(n+1)/2 Cholesky entries (row-major lower triangle)
        W_V in R^{Pn x n_pairs}, one column per coordinate pair i < j
        w_G in R^P
        theta_F: Approach A LSTM layout (gates + readout)
    """
    n: int
    n_features: int                 # P = dim(rho) (constant + skip + random features)
    friction: LSTMWeightLayout
    gyroscopic: bool = True

    BLOCKS: ClassVar[Tuple[str, ...]] = ("M", "V", "G", "F")

    @property
    def n_chol(self) -> int:
        return self.n * (self.n + 1) // 2

    @property
    def n_pairs(self) -> int:
        return self.n * (self.n - 1) // 2 if self.gyroscopic else 0

    def block_size(self, name: str) -> int:
        P = self.n_features
        return {
            "M": P * self.n_chol,
            "V": P * self.n * self.n_pairs,
            "G": P,
            "F": self.friction.n_params,
        }[name]

    def block_slice(self, name: str) -> slice:
        start = 0
        for b in self.BLOCKS:
            size = self.block_size(b)
            if b == name:
                return slice(start, start + size)
            start += size
        raise KeyError(name)

    @property
    def n_params(self) -> int:
        return sum(self.block_size(b) for b in self.BLOCKS)

    def W_M(self, theta: np.ndarray) -> np.ndarray:
        return theta[self.block_slice("M")].reshape((self.n_features, self.n_chol), order="F")

    def W_V(self, theta: np.ndarray) -> np.ndarray:
        return theta[self.block_slice("V")].reshape((self.n_features * self.n, self.n_pairs), order="F")

    def w_G(self, theta: np.ndarray) -> np.ndarray:
        return theta[self.block_slice("G")]

    def theta_F(self, theta: np.ndarray) -> np.ndarray:
        return theta[self.block_slice("F")]


# ---------------------------------------------------------------------------- caches
@dataclass
class InertiaCache:
    """Cholesky inertia M_hat = L L^T + eps I and its configuration derivatives."""
    a: np.ndarray       # (R,)   pre-activations W_M^T rho
    gp: np.ndarray      # (R,)   g'(a):  sigmoid on the diagonal, 1 off-diagonal
    gpp: np.ndarray     # (R,)   g''(a): sigmoid' on the diagonal, 0 off-diagonal
    beta: np.ndarray    # (R, n) W_M^T Drho  (d a / d q)
    L: np.ndarray       # (n, n)
    M: np.ndarray       # (n, n)
    dL: np.ndarray      # (n, n, n)  dL[k] = dL / dq_k
    dM: np.ndarray      # (n, n, n)  dM[k] = dM / dq_k


@dataclass
class PILSTMCache:
    """Every intermediate of one PI-LSTM evaluation (reused by the Jacobian engine)."""
    q: np.ndarray
    q_dot: np.ndarray
    u: np.ndarray
    rho: np.ndarray         # rho(q), (P,)
    drho: np.ndarray        # (P, n)
    inertia: InertiaCache
    coriolis: np.ndarray    # C_hat q_dot (Christoffel part)
    gyro_basis: np.ndarray  # rho kron q_dot, (Pn,)
    gyro: np.ndarray        # S q_dot
    potential: float        # P_hat(q)
    gravity: np.ndarray     # G_hat(q)
    friction: LSTMCache     # friction LSTM (readout phi_F)
    friction_force: np.ndarray  # F_hat
    friction_gain: np.ndarray   # dF_hat_i / dphi_F,i (diagonal)
    tau: np.ndarray         # B u - V_m q_dot - G - F
    phi: np.ndarray         # M_hat^{-1} tau


# ---------------------------------------------------------------------------- sub-networks
def cholesky_inertia(
    W_M: np.ndarray, rho: np.ndarray, drho: np.ndarray, eps_M: float,
    rows: np.ndarray, cols: np.ndarray, diag: np.ndarray, row_scale: np.ndarray,
) -> InertiaCache:
    """
    Evaluates M_hat(q) = L L^T + eps_M I and dM/dq_k = dL_k L^T + L dL_k^T, where
    L_r = d_{i_r} g_r(a_r) (softplus on the diagonal, identity below it). The row scale is
    folded into g' and g'', so the Jacobian engine sees L_r = g_r(a_r) with scaled g.
    """
    n = drho.shape[1]
    a = W_M.T @ rho
    beta = W_M.T @ drho
    sig = sigmoid(a)
    d = row_scale[rows]
    ell = d * np.where(diag, softplus(a), a)
    gp = d * np.where(diag, sig, 1.0)
    gpp = d * np.where(diag, sig * (1.0 - sig), 0.0)

    L = np.zeros((n, n))
    L[rows, cols] = ell
    dL = np.zeros((n, n, n))
    dL[:, rows, cols] = (gp[:, None] * beta).T
    M = L @ L.T + eps_M * np.eye(n)
    dLLt = dL @ L.T
    dM = dLLt + np.transpose(dLLt, (0, 2, 1))
    return InertiaCache(a=a, gp=gp, gpp=gpp, beta=beta, L=L, M=M, dL=dL, dM=dM)


def inertia_time_derivative(dM: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
    """M_hat_dot = sum_k dM/dq_k q_dot_k."""
    return np.einsum("kij,k->ij", dM, q_dot)


def christoffel_matrix(dM: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
    """C_ij = sum_k 1/2 (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i) q_dot_k."""
    t1 = np.einsum("kij,k->ij", dM, q_dot)
    t2 = np.einsum("jik,k->ij", dM, q_dot)
    t3 = np.einsum("ijk,k->ij", dM, q_dot)
    return 0.5 * (t1 + t2 - t3)


def christoffel_vector(dM: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
    """C_hat q_dot = M_hat_dot q_dot - 1/2 d/dq (q_dot^T M_hat q_dot)."""
    return inertia_time_derivative(dM, q_dot) @ q_dot - 0.5 * np.einsum("i,kij,j->k", q_dot, dM, q_dot)


def gyroscopic_matrix(s: np.ndarray, pairs: Sequence[Tuple[int, int]], n: int) -> np.ndarray:
    """S = sum_p s_p (E_{i_p j_p} - E_{j_p i_p}) (skew-symmetric)."""
    S = np.zeros((n, n))
    for sp, (i, j) in zip(s, pairs):
        S[i, j] += sp
        S[j, i] -= sp
    return S


# ---------------------------------------------------------------------------- network
class PILSTMNetwork:
    """
    Physics-informed acceleration model
        Phi_hat(q, q_dot, u, c_hat; theta) = M_hat^{-1} [B u - V_m_hat q_dot - G_hat - F_hat].
    Stateless: parameters theta and the friction memory (c_hat, h_hat) are passed in.
    """

    def __init__(
        self,
        n_coords: int = 2,
        input_matrix: np.ndarray | None = None,
        revolute: Sequence[bool] = (False, True),
        n_features: int = 0,
        harmonics: int = 2,
        feature_scale: float = 1.0,
        position_scale: Sequence[float] | None = None,
        position_offset: Sequence[float] | None = None,
        linear_skip: bool = True,
        cyclic: Sequence[bool] | None = None,
        inertia_scale: Sequence[float] | None = None,
        eps_M: float = 0.05,
        gyroscopic: bool = True,
        friction_hidden: int = 8,
        velocity_scale: Sequence[float] | None = None,
        dissipative_friction: bool = True,
        damping_scale: Sequence[float] | None = None,
        damping_offset: float = -3.0,
        seed: int = 0,
    ) -> None:
        if len(revolute) != n_coords:
            raise ValueError("revolute must have one entry per coordinate.")
        if eps_M <= 0.0:
            raise ValueError("eps_M must be strictly positive.")
        n = n_coords
        self.n = n
        self.B = np.array([[1.0]] + [[0.0]] * (n - 1)) if input_matrix is None else np.atleast_2d(
            np.asarray(input_matrix, dtype=np.float64))
        if self.B.shape[0] != n:
            raise ValueError("input_matrix must have n rows.")
        self.eps_M = eps_M
        scale = np.ones(n) if inertia_scale is None else np.asarray(inertia_scale, dtype=np.float64)
        if scale.shape != (n,) or np.any(scale <= 0.0):
            raise ValueError("inertia_scale needs n strictly positive entries.")
        self.row_scale = np.sqrt(scale)
        self.cyclic = np.zeros(n, dtype=bool) if cyclic is None else np.asarray(cyclic, dtype=bool)
        if self.cyclic.size != n or self.cyclic.all():
            raise ValueError("cyclic needs n entries, at least one of them False.")
        self.features = ConfigurationFeatures(
            revolute, n_features, feature_scale, position_scale, position_offset, linear_skip,
            active=~self.cyclic, harmonics=harmonics, seed=seed + 1000,
        )
        self.rows, self.cols = np.tril_indices(n)
        self.diag = self.rows == self.cols
        self.pairs = [(i, j) for i in range(n) for j in range(i + 1, n)] if gyroscopic else []
        self.velocity_scale = np.ones(n) if velocity_scale is None else np.asarray(velocity_scale, dtype=np.float64)
        self.dissipative_friction = dissipative_friction
        # d0_i [1/s x inertia units]: defaults to the inertia scale, so d_i(0) = 0.049 d0_i
        self.damping_scale = scale if damping_scale is None else np.asarray(damping_scale, dtype=np.float64)
        self.damping_offset = damping_offset
        friction_layout = LSTMWeightLayout(n + friction_hidden + 1, friction_hidden, n)
        self.layout = PILSTMLayout(n, self.features.dim, friction_layout, gyroscopic)

    # -------------------------------------------------------------- parameters
    def initial_parameters(
        self, inertia_init: Sequence[float], gate_scale: float = 0.5, seed: int = 0
    ) -> np.ndarray:
        """
        theta(0): M_hat(q) = diag(inertia_init) for all q (a configuration-independent prior),
        S = 0, P_hat = 0, friction gates N(0, gate_scale^2), friction readout 0.
        """
        lay = self.layout
        if len(inertia_init) != self.n or min(inertia_init) <= self.eps_M:
            raise ValueError("inertia_init needs n entries, each larger than eps_M.")
        theta = np.zeros(lay.n_params)
        W_M = lay.W_M(theta)
        for r in np.flatnonzero(self.diag):
            i = self.rows[r]
            W_M[0, r] = softplus_inverse(float(np.sqrt(inertia_init[i] - self.eps_M) / self.row_scale[i]))
        rng = np.random.default_rng(seed)
        fl = lay.friction
        theta[lay.block_slice("F")][: fl.n_gate_params] = rng.normal(0.0, gate_scale, fl.n_gate_params)
        return theta

    # -------------------------------------------------------------- forward pass
    def friction_input(self, q_dot: np.ndarray, h_hat: np.ndarray) -> np.ndarray:
        return np.concatenate((self.velocity_scale * q_dot, h_hat, (1.0,)))

    def inertia(self, theta: np.ndarray, q: np.ndarray) -> InertiaCache:
        rho, drho = self.features(q)
        return cholesky_inertia(self.layout.W_M(theta), rho, drho, self.eps_M, self.rows, self.cols, self.diag, self.row_scale)

    def forward(
        self,
        theta: np.ndarray,
        q: np.ndarray,
        q_dot: np.ndarray,
        u: np.ndarray,
        c_hat: np.ndarray,
        h_hat: np.ndarray,
    ) -> PILSTMCache:
        lay = self.layout
        rho, drho = self.features(q)
        inertia = cholesky_inertia(lay.W_M(theta), rho, drho, self.eps_M, self.rows, self.cols, self.diag, self.row_scale)
        coriolis = christoffel_vector(inertia.dM, q_dot)

        gyro_basis = np.kron(rho, q_dot)
        gyro = np.zeros(self.n)
        if self.pairs:
            s = lay.W_V(theta).T @ gyro_basis
            for sp, (i, j) in zip(s, self.pairs):
                gyro[i] += sp * q_dot[j]
                gyro[j] -= sp * q_dot[i]

        w_G = lay.w_G(theta)
        potential = float(w_G @ rho)
        gravity = drho.T @ w_G

        friction = lstm_forward(lay.theta_F(theta), self.friction_input(q_dot, h_hat), c_hat, lay.friction)
        if self.dissipative_friction:
            z = friction.phi + self.damping_offset
            friction_force = self.damping_scale * softplus(z) * q_dot
            friction_gain = self.damping_scale * sigmoid(z) * q_dot
        else:
            friction_force = friction.phi
            friction_gain = np.ones(self.n)

        tau = self.B @ u - coriolis - gyro - gravity - friction_force
        phi = np.linalg.solve(inertia.M, tau)
        return PILSTMCache(
            q=q, q_dot=q_dot, u=u, rho=rho, drho=drho, inertia=inertia, coriolis=coriolis,
            gyro_basis=gyro_basis, gyro=gyro, potential=potential, gravity=gravity,
            friction=friction, friction_force=friction_force, friction_gain=friction_gain, tau=tau, phi=phi,
        )

    # -------------------------------------------------------------- diagnostics
    def coriolis_matrix(self, theta: np.ndarray, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        """V_m_hat = C_hat (Christoffel) + S (gyroscopic)."""
        rho, _ = self.features(q)
        C = christoffel_matrix(self.inertia(theta, q).dM, q_dot)
        if self.pairs:
            s = self.layout.W_V(theta).T @ np.kron(rho, q_dot)
            C = C + gyroscopic_matrix(s, self.pairs, self.n)
        return C

    def skew_residual(self, theta: np.ndarray, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        """N = M_hat_dot - 2 V_m_hat; skew-symmetric by construction."""
        M_dot = inertia_time_derivative(self.inertia(theta, q).dM, q_dot)
        return M_dot - 2.0 * self.coriolis_matrix(theta, q, q_dot)

    def kinetic_energy(self, theta: np.ndarray, q: np.ndarray, q_dot: np.ndarray) -> float:
        return float(0.5 * q_dot @ self.inertia(theta, q).M @ q_dot)

    def potential_energy(self, theta: np.ndarray, q: np.ndarray) -> float:
        rho, _ = self.features(q)
        return float(self.layout.w_G(theta) @ rho)

    def conservative_acceleration(self, theta: np.ndarray, q: np.ndarray, q_dot: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Learned rigid-body model without friction: M_hat^{-1}(B u - V_m_hat q_dot - G_hat)."""
        _, drho = self.features(q)
        V = self.coriolis_matrix(theta, q, q_dot)
        tau = self.B @ u - V @ q_dot - drho.T @ self.layout.w_G(theta)
        return np.linalg.solve(self.inertia(theta, q).M, tau)
