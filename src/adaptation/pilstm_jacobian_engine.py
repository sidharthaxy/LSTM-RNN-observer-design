"""
Analytical sensitivity Jacobians and blockwise Lyapunov adaptation for the PI-LSTM
(src/observers/pilstm_network.py).

Acceleration model:  Phi_hat = M_hat^{-1} tau,  tau = B u - C_hat q_dot - S q_dot - G_hat - F_hat.
For any parameter block beta,
    d Phi_hat / d theta_beta = M_hat^{-1} ( d tau / d theta_beta - (d M_hat / d theta_beta) Phi_hat ),
so with the adjoint vector lambda = M_hat^{-1} e (M_hat symmetric) the adaptation signal is
    Phi'_beta^T e = d/d theta_beta [ lambda^T tau - lambda^T M_hat Phi_hat ]     (lambda, Phi_hat held).
It is evaluated in O(p) without forming Phi'. Block by block:

  G:  lambda^T tau contains -lambda^T Drho^T w_G              =>  g_G = -Drho lambda
  V:  -lambda^T S q_dot, s_p = W_V[:, p]^T (rho kron q_dot)   =>  g_V[:, p] = -(lambda_i q_dot_j - lambda_j q_dot_i)(rho kron q_dot)
  F:  -lambda^T F_hat, F_hat_i = F_i(phi_F,i) (diagonal map of the Approach A LSTM readout)
                                                              =>  g_F = Phi_F'^T (-lambda * dF/dphi_F)
  M:  M_hat enters twice: through M_hat^{-1} and through the Christoffel vector
          c = C_hat q_dot = sum_k [a_k^T dM_k q_dot],  a_k = q_dot_k lambda - 1/2 lambda_k q_dot  (as lambda^T c),
      dM_k = dL_k L^T + L dL_k^T. Differentiating w.r.t. the Cholesky factor and its q-derivative:
          d(lambda^T M Phi)/dL     = (lambda Phi^T + Phi lambda^T) L                 =: K
          d(lambda^T c)/dL         = sum_k (q_dot a_k^T + a_k q_dot^T) dL_k          =: Z
          d(lambda^T c)/d(dL_k)    = a_k q_dot^T L + q_dot a_k^T L                   =: Y_k
      Chain rule through L_r = g_r(a_r), a_r = W_r^T rho, dL_k,r = g_r'(a_r) W_r^T Drho_k:
          d L_r / d W_r      = g_r' rho
          d dL_k,r / d W_r   = g_r'' (W_r^T Drho_k) rho + g_r' Drho_k
      =>  g_{W_r} = -(Z + K)_r g_r' rho - sum_k Y_k,r (g_r'' beta_rk rho + g_r' Drho_k).

Adaptation law (per block, with its own gain and projection ball):
    theta_beta_hat_dot = proj_beta( Gamma_beta Phi'_beta^T W e ),   ||theta_beta_hat|| <= W_bar_beta,
with the error metric W chosen by `metric`:
    "euclidean":  W = I,      lambda = M_hat^{-1} e   (gradient of 1/2 |e|^2)
    "kinetic":    W = M_hat,  lambda = e              (gradient of 1/2 e^T M_hat e, the kinetic-energy metric)
Both are descent directions for the prediction error since Phi'^T W Phi' >= 0. The kinetic metric
removes one factor M_hat^{-1} from the loop gain, whose Euclidean value scales like M_hat^{-2}
and so explodes for a light (small-inertia) coordinate or while M_hat is being learned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Literal, Mapping
import numpy as np

from src.adaptation.jacobian_engine import phi_jacobian_transpose_vec, smooth_projection

if TYPE_CHECKING:  # annotations only: src.observers imports this module
    from src.observers.pilstm_network import PILSTMCache, PILSTMLayout, PILSTMNetwork

BLOCKS = ("M", "V", "G", "F")  # == PILSTMLayout.BLOCKS
Metric = Literal["euclidean", "kinetic"]


def adjoint(cache: PILSTMCache, e: np.ndarray, metric: Metric = "euclidean") -> np.ndarray:
    """lambda = M_hat^{-1} W e: M_hat^{-1} e (euclidean) or e (kinetic)."""
    if metric == "kinetic":
        return e
    return np.linalg.solve(cache.inertia.M, e)


def inertia_block_grad(net: PILSTMNetwork, cache: PILSTMCache, lam: np.ndarray) -> np.ndarray:
    """Phi'_M^T e as vec(g_W_M) (column-major, P x R)."""
    inr = cache.inertia
    qd, phi, L = cache.q_dot, cache.phi, inr.L
    rows, cols = net.rows, net.cols

    # a_k = q_dot_k lambda - 1/2 lambda_k q_dot, stacked as rows A[k] (n x n)
    A = np.outer(qd, lam) - 0.5 * np.outer(lam, qd)
    K = (np.outer(lam, phi) + np.outer(phi, lam)) @ L
    # Z = sum_k (q_dot a_k^T + a_k q_dot^T) dL_k
    Z = np.einsum("i,kj,kjl->il", qd, A, inr.dL) + np.einsum("ki,j,kjl->il", A, qd, inr.dL)
    # Y_k = a_k (L^T q_dot)^T + q_dot (L^T a_k)^T, restricted to the lower-triangular entries
    Ltq = L.T @ qd
    LtA = A @ L                                             # row k = (L^T a_k)^T
    Y = A[:, :, None] * Ltq[None, None, :] + qd[None, :, None] * LtA[:, None, :]
    Yr = Y[:, rows, cols]                                   # (n, R)

    gL = (Z + K)[rows, cols]                                # (R,)
    coef_rho = gL * inr.gp + np.einsum("kr,r,rk->r", Yr, inr.gpp, inr.beta)
    grad = -np.outer(cache.rho, coef_rho) - cache.drho @ (Yr * inr.gp[None, :])
    return grad.ravel(order="F")


def gyroscopic_block_grad(net: PILSTMNetwork, cache: PILSTMCache, lam: np.ndarray) -> np.ndarray:
    """Phi'_V^T e as vec(g_W_V) (column-major, Pn x n_pairs)."""
    qd = cache.q_dot
    power = np.array([lam[i] * qd[j] - lam[j] * qd[i] for i, j in net.pairs])
    return -np.outer(cache.gyro_basis, power).ravel(order="F")


def gravity_block_grad(cache: PILSTMCache, lam: np.ndarray) -> np.ndarray:
    """Phi'_G^T e = -Drho lambda."""
    return -cache.drho @ lam


def friction_block_grad(layout: PILSTMLayout, theta: np.ndarray, cache: PILSTMCache, lam: np.ndarray) -> np.ndarray:
    """Phi'_F^T e = (d phi_F / d theta_F)^T (-lambda * dF_hat/dphi_F) (Approach A Kronecker form)."""
    return phi_jacobian_transpose_vec(cache.friction, layout.theta_F(theta), layout.friction, -lam * cache.friction_gain)


def block_gradients(
    net: PILSTMNetwork, theta: np.ndarray, cache: PILSTMCache, e: np.ndarray, metric: Metric = "euclidean"
) -> Dict[str, np.ndarray]:
    """{beta: Phi'_beta^T W e} for beta in (M, V, G, F)."""
    lam = adjoint(cache, e, metric)
    grads = {
        "M": inertia_block_grad(net, cache, lam),
        "V": gyroscopic_block_grad(net, cache, lam) if net.pairs else np.zeros(0),
        "G": gravity_block_grad(cache, lam),
        "F": friction_block_grad(net.layout, theta, cache, lam),
    }
    return grads


def pilstm_jacobian_transpose_vec(
    net: PILSTMNetwork, theta: np.ndarray, cache: PILSTMCache, e: np.ndarray
) -> np.ndarray:
    """Phi'^T e in R^p, blocks ordered as theta."""
    g = block_gradients(net, theta, cache, e)
    return np.concatenate([g[b] for b in BLOCKS])


def pilstm_jacobian(net: PILSTMNetwork, theta: np.ndarray, cache: PILSTMCache) -> np.ndarray:
    """Explicit Phi' = d Phi_hat / d theta in R^{n x p} (row i = Phi'^T unit_i). For tests / analysis."""
    eye = np.eye(net.n)
    return np.vstack([pilstm_jacobian_transpose_vec(net, theta, cache, eye[i]) for i in range(net.n)])


class PILSTMAdaptationLaw:
    """
    theta_beta_hat_dot = proj_beta(Gamma_beta Phi'_beta^T W e) for beta in {M, V, G, F}, with
    Gamma_F = diag(gamma_F_gates I, gamma_F_out I). All blocks adapt simultaneously from the
    same filtered error e; each block is confined to its own ball ||theta_beta|| <= W_bar_beta.
    """

    def __init__(
        self,
        net: PILSTMNetwork,
        gammas: Mapping[str, float],
        w_bars: Mapping[str, float],
        proj_eps: float = 0.1,
        metric: Metric = "euclidean",
    ) -> None:
        required = ("M", "V", "G", "F_gates", "F_out")
        if any(gammas[k] < 0.0 for k in required):
            raise ValueError("Adaptation gains must be non-negative.")
        if any(w_bars[b] <= 0.0 for b in BLOCKS) or proj_eps <= 0.0:
            raise ValueError("Projection radii and proj_eps must be strictly positive.")
        self.net = net
        self.layout = net.layout
        self.w_bars = {b: float(w_bars[b]) for b in BLOCKS}
        self.proj_eps = proj_eps
        self.metric: Metric = metric
        fl = self.layout.friction
        self.gamma = {
            "M": np.full(self.layout.block_size("M"), gammas["M"]),
            "V": np.full(self.layout.block_size("V"), gammas["V"]),
            "G": np.full(self.layout.block_size("G"), gammas["G"]),
            "F": np.concatenate((np.full(fl.n_gate_params, gammas["F_gates"]), np.full(fl.n_out_params, gammas["F_out"]))),
        }

    def theta_dot(self, theta: np.ndarray, cache: PILSTMCache, e: np.ndarray) -> np.ndarray:
        grads = block_gradients(self.net, theta, cache, e, self.metric)
        out = np.zeros_like(theta)
        for b in BLOCKS:
            sl = self.layout.block_slice(b)
            if sl.stop == sl.start:
                continue
            tau = self.gamma[b] * grads[b]
            out[sl] = smooth_projection(theta[sl], tau, self.gamma[b], self.w_bars[b], self.proj_eps)
        return out

    def enforce_bound(self, theta: np.ndarray) -> None:
        """Discrete-time safeguard: radial rescale of any block that overshot its ball."""
        for b in BLOCKS:
            blk = theta[self.layout.block_slice(b)]
            norm = float(np.linalg.norm(blk))
            if norm > self.w_bars[b]:
                blk *= self.w_bars[b] / norm

    def block_norms(self, theta: np.ndarray) -> Dict[str, float]:
        return {b: float(np.linalg.norm(theta[self.layout.block_slice(b)])) for b in BLOCKS}
