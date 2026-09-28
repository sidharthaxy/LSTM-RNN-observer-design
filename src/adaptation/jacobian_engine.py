"""
Analytical Jacobian engine and Lyapunov adaptation law for the continuous-time
Lb-LSTM (Griffis, Patil, Hart & Dixon, IEEE Control Systems Letters, 2024).

LSTM cell (static map of the augmented input zeta and the filtered cell memory c_hat):
    f  = sigma_g(W_f^T zeta)          i  = sigma_g(W_i^T zeta)
    c* = sigma_c(W_c^T zeta)          o  = sigma_g(W_o^T zeta)
    c  = f * c_hat + i * c*                      (instantaneous cell state)
    h  = o * sigma_c(c)                          (instantaneous hidden state)
    Phi_hat = W_h^T h
with sigma_g the logistic sigmoid, sigma_c = tanh, W_{c,i,f,o} in R^{d x L}, W_h in R^{L x n}.

Parameter vector (column-major vec, i.e. vec stacks the columns of each matrix):
    theta = [vec(W_c)^T, vec(W_i)^T, vec(W_f)^T, vec(W_o)^T, vec(W_h)^T]^T in R^{4dL + Ln}

Because the four gate blocks are stored contiguously, theta[:4dL] viewed as a
column-major d x 4L matrix is exactly [W_c | W_i | W_f | W_o], so all four gate
pre-activations are a single mat-vec.

Exact Jacobian Phi' = d Phi_hat / d theta in R^{n x p}, holding (zeta, c_hat) fixed:
    d Phi_hat / d vec(W_g) = (W_h^T diag(delta_g)) kron zeta^T,   g in {c, i, f, o}
    d Phi_hat / d vec(W_h) = I_n kron h^T
with the gate sensitivity vectors (s = o * (1 - tanh(c)^2) = dh/dc):
    delta_c = s * i * (1 - c*^2)
    delta_i = s * c* * i * (1 - i)
    delta_f = s * c_hat * f * (1 - f)
    delta_o = tanh(c) * o * (1 - o)

Adaptation law:  theta_hat_dot = proj( Gamma Phi'^T e ),  ||theta_hat|| <= W_bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Tuple
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically safe logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


@dataclass(frozen=True)
class LSTMWeightLayout:
    """Index bookkeeping for theta = [vec(W_c), vec(W_i), vec(W_f), vec(W_o), vec(W_h)]."""
    zeta_dim: int     # d = 2n + m + L + 1
    hidden_dim: int   # L
    output_dim: int   # n

    GATE_ORDER: ClassVar[Tuple[str, ...]] = ("c", "i", "f", "o")

    @property
    def gate_block_size(self) -> int:
        return self.zeta_dim * self.hidden_dim

    @property
    def n_gate_params(self) -> int:
        return 4 * self.gate_block_size

    @property
    def n_out_params(self) -> int:
        return self.hidden_dim * self.output_dim

    @property
    def n_params(self) -> int:
        return self.n_gate_params + self.n_out_params

    def block_slice(self, name: str) -> slice:
        """Slice of theta holding vec(W_name), name in {c, i, f, o, h}."""
        if name == "h":
            return slice(self.n_gate_params, self.n_params)
        k = self.GATE_ORDER.index(name)
        return slice(k * self.gate_block_size, (k + 1) * self.gate_block_size)

    def unpack(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns column-major *views* into theta:
            W_gates = [W_c | W_i | W_f | W_o] in R^{d x 4L},  W_h in R^{L x n}.
        """
        W_gates = theta[: self.n_gate_params].reshape((self.zeta_dim, 4 * self.hidden_dim), order="F")
        W_h = theta[self.n_gate_params:].reshape((self.hidden_dim, self.output_dim), order="F")
        return W_gates, W_h

    def weight_matrix(self, theta: np.ndarray, name: str) -> np.ndarray:
        """Individual weight matrix W_name (view), name in {c, i, f, o, h}."""
        if name == "h":
            return self.unpack(theta)[1]
        return theta[self.block_slice(name)].reshape((self.zeta_dim, self.hidden_dim), order="F")


@dataclass
class LSTMCache:
    """Intermediate quantities of one LSTM cell evaluation (reused by the Jacobian)."""
    zeta: np.ndarray
    c_hat: np.ndarray
    f: np.ndarray
    i: np.ndarray
    c_star: np.ndarray
    o: np.ndarray
    c: np.ndarray
    tanh_c: np.ndarray
    h: np.ndarray
    phi: np.ndarray


def lstm_forward(
    theta: np.ndarray,
    zeta: np.ndarray,
    c_hat: np.ndarray,
    layout: LSTMWeightLayout,
) -> LSTMCache:
    """Evaluates the LSTM cell (Eqs. 7-8) and caches every intermediate."""
    L = layout.hidden_dim
    W_gates, W_h = layout.unpack(theta)
    pre = W_gates.T @ zeta  # [a_c, a_i, a_f, a_o] in R^{4L}

    c_star = np.tanh(pre[0:L])
    i_gate = sigmoid(pre[L:2 * L])
    f_gate = sigmoid(pre[2 * L:3 * L])
    o_gate = sigmoid(pre[3 * L:4 * L])

    c = f_gate * c_hat + i_gate * c_star
    tanh_c = np.tanh(c)
    h = o_gate * tanh_c
    phi = W_h.T @ h

    return LSTMCache(
        zeta=zeta, c_hat=c_hat, f=f_gate, i=i_gate, c_star=c_star,
        o=o_gate, c=c, tanh_c=tanh_c, h=h, phi=phi,
    )


def gate_sensitivities(cache: LSTMCache) -> np.ndarray:
    """
    Stacked sensitivities [delta_c, delta_i, delta_f, delta_o] in R^{4L}, where
    delta_g = d h / d a_g (element-wise) and a_g = W_g^T zeta is the gate pre-activation.
    """
    dh_dc = cache.o * (1.0 - cache.tanh_c ** 2)
    delta_c = dh_dc * cache.i * (1.0 - cache.c_star ** 2)
    delta_i = dh_dc * cache.c_star * cache.i * (1.0 - cache.i)
    delta_f = dh_dc * cache.c_hat * cache.f * (1.0 - cache.f)
    delta_o = cache.tanh_c * cache.o * (1.0 - cache.o)
    return np.concatenate((delta_c, delta_i, delta_f, delta_o))


def phi_jacobian(
    cache: LSTMCache,
    theta: np.ndarray,
    layout: LSTMWeightLayout,
) -> np.ndarray:
    """
    Explicit Jacobian Phi' = d Phi_hat / d theta in R^{n x p} (Kronecker form).
    O(n p) memory; use `phi_jacobian_transpose_vec` inside the observer loop.
    """
    _, W_h = layout.unpack(theta)
    delta = gate_sensitivities(cache)
    # (W_h^T diag(delta)) kron zeta^T, for all four gates at once: W_h^T is tiled per gate.
    W_h_tiled = np.tile(W_h.T, (1, 4))                       # n x 4L
    J_gates = np.kron(W_h_tiled * delta[None, :], cache.zeta[None, :])  # n x 4dL
    J_out = np.kron(np.eye(layout.output_dim), cache.h[None, :])        # n x Ln
    return np.hstack((J_gates, J_out))


def phi_jacobian_transpose_vec(
    cache: LSTMCache,
    theta: np.ndarray,
    layout: LSTMWeightLayout,
    e: np.ndarray,
) -> np.ndarray:
    """
    Phi'^T e in R^p without forming Phi'. Using (A kron zeta)(e kron 1) = (A e) kron zeta:
        gate block g:  vec( zeta (delta_g * (W_h e))^T )
        output block:  vec( h e^T )
    """
    _, W_h = layout.unpack(theta)
    delta = gate_sensitivities(cache)
    back = np.tile(W_h @ e, 4) * delta                # R^{4L}
    grad_gates = np.outer(back, cache.zeta).ravel()   # column-major vec of zeta back^T
    grad_out = np.outer(e, cache.h).ravel()           # column-major vec of h e^T
    return np.concatenate((grad_gates, grad_out))


def smooth_projection(
    theta: np.ndarray,
    tau: np.ndarray,
    gamma: np.ndarray,
    w_bar: float,
    eps: float,
) -> np.ndarray:
    """
    Smooth (Lipschitz) parameter projection onto the ball ||theta|| <= w_bar
    (Pomet & Praly 1992; Lavretsky & Wise 2013), for tau = Gamma y with diagonal Gamma.

    Convex boundary function, zero on ||theta|| = w_bar / sqrt(1 + eps), one on ||theta|| = w_bar:
        f(theta) = ((1 + eps) ||theta||^2 - w_bar^2) / (eps w_bar^2)
    Projection (grad f is parallel to theta):
        proj = tau - f(theta) * Gamma theta (theta^T tau) / (theta^T Gamma theta)
               if f(theta) > 0 and theta^T tau > 0, else tau.
    At the outer boundary the outward radial component of tau is removed exactly, so the
    continuous-time flow satisfies ||theta(t)|| <= w_bar whenever ||theta(0)|| <= w_bar.
    """
    theta_sq = float(theta @ theta)
    f_val = ((1.0 + eps) * theta_sq - w_bar ** 2) / (eps * w_bar ** 2)
    radial = float(theta @ tau)
    if f_val <= 0.0 or radial <= 0.0:
        return tau
    gamma_theta = gamma * theta
    return tau - min(f_val, 1.0) * gamma_theta * (radial / float(theta @ gamma_theta))


class LyapunovAdaptationLaw:
    """
    theta_hat_dot = proj( Gamma Phi'^T e ) with block-diagonal gain
    Gamma = diag(gamma_gates I_{4dL}, gamma_out I_{Ln}).
    """

    def __init__(
        self,
        layout: LSTMWeightLayout,
        gamma_gates: float,
        gamma_out: float,
        w_bar: float,
        proj_eps: float = 0.1,
    ) -> None:
        if gamma_gates < 0.0 or gamma_out < 0.0:
            raise ValueError("Adaptation gains must be non-negative.")
        if w_bar <= 0.0 or proj_eps <= 0.0:
            raise ValueError("w_bar and proj_eps must be strictly positive.")
        self.layout = layout
        self.w_bar = w_bar
        self.proj_eps = proj_eps
        self.gamma = np.concatenate((
            np.full(layout.n_gate_params, gamma_gates, dtype=np.float64),
            np.full(layout.n_out_params, gamma_out, dtype=np.float64),
        ))

    def theta_dot(self, theta: np.ndarray, cache: LSTMCache, e: np.ndarray) -> np.ndarray:
        tau = self.gamma * phi_jacobian_transpose_vec(cache, theta, self.layout, e)
        return smooth_projection(theta, tau, self.gamma, self.w_bar, self.proj_eps)

    def enforce_bound(self, theta: np.ndarray) -> None:
        """
        Discrete-time safeguard applied after each integration step: the projection
        guarantees invariance of the ball for the continuous flow, but a finite Euler
        step can overshoot it by O(dt). Rescale radially in place if that happens.
        """
        norm = float(np.linalg.norm(theta))
        if norm > self.w_bar:
            theta *= self.w_bar / norm
