"""
Approach A: Unconstrained Continuous-Time Lb-LSTM Observer.
Approximates the complete nonlinear acceleration vector field [x_ddot, theta_ddot]^T.

Continuous-Time LSTM formulation:
Input: z(t) = [x_hat(t), u(t)] in R^5
Gates:
    i(t) = sigmoid(W_i * z + b_i)
    f(t) = sigmoid(W_f * z + b_f)
    o(t) = sigmoid(W_o * z + b_o)
    g(t) = tanh(W_g * z + b_g)
Cell state differential equation:
    dc/dt = -(1 - f(t)) * c(t) + i(t) * g(t)
Hidden state:
    h(t) = o(t) * tanh(c(t)) in R^{d_h}

Acceleration vector field:
    [x_hat_ddot, theta_hat_ddot]^T = W_a * h(t) + b_a

Observer dynamics with output injection:
    dx_hat_1 / dt = x_hat_2 + L1 * (y1 - x_hat_1)
    dx_hat_2 / dt = a_hat_1(t) + L2 * (y1 - x_hat_1)
    dx_hat_3 / dt = x_hat_4 + L3 * (y2 - x_hat_3)
    dx_hat_4 / dt = a_hat_2(t) + L4 * (y2 - x_hat_3)

Lyapunov Adaptive Weight Law (Krasovskii / e-modification):
    dW_a / dt = Gamma * [ (y - y_hat) * h(t)^T ] - sigma_leak * W_a
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np


@dataclass
class BlackBoxLSTMConfig:
    """Hyperparameters for Continuous Black-Box Lb-LSTM Observer."""
    hidden_dim: int = 16
    gamma_a: float = 15.0         # Adaptation gain for acceleration readout weights
    sigma_leak: float = 0.005     # Leakage coefficient for UUB boundedness
    # Output injection gains (Luenberger gains)
    l1: float = 50.0              # Cart pos error -> x_dot
    l2: float = 300.0             # Cart pos error -> x_ddot
    l3: float = 50.0              # Pendulum angle error -> theta_dot
    l4: float = 300.0             # Pendulum angle error -> theta_ddot
    seed: int = 42


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class BlackBoxLSTMObserver:
    """
    Unconstrained Continuous-Time Lb-LSTM Observer.
    Evaluated in continuous time using NumPy and analytical Lyapunov update laws.
    """

    def __init__(self, config: BlackBoxLSTMConfig | None = None) -> None:
        self.config = config or BlackBoxLSTMConfig()
        h = self.config.hidden_dim
        rng = np.random.default_rng(self.config.seed)

        # Dimension: input z in R^5: [x1, x2, x3, x4, u]
        d_in = 5

        # Orthogonal / Xavier initialization for gate projections
        self.W_i = rng.normal(0.0, 0.3, size=(h, d_in)).astype(np.float64)
        self.b_i = np.zeros(h, dtype=np.float64)

        self.W_f = rng.normal(0.0, 0.3, size=(h, d_in)).astype(np.float64)
        self.b_f = np.ones(h, dtype=np.float64)  # Initialize forget gate open

        self.W_o = rng.normal(0.0, 0.3, size=(h, d_in)).astype(np.float64)
        self.b_o = np.zeros(h, dtype=np.float64)

        self.W_g = rng.normal(0.0, 0.3, size=(h, d_in)).astype(np.float64)
        self.b_g = np.zeros(h, dtype=np.float64)

        # Adaptive acceleration readout weights: W_a in R^{2 x h}
        # Maps hidden representation h(t) -> [x_ddot, theta_ddot]^T
        self.W_a = np.zeros((2, h), dtype=np.float64)
        self.b_a = np.zeros(2, dtype=np.float64)

        # Continuous states
        self.x_hat = np.zeros(4, dtype=np.float64)
        self.c = np.zeros(h, dtype=np.float64)      # LSTM cell state
        self.h = np.zeros(h, dtype=np.float64)      # LSTM hidden state

    def reset(self, initial_state: np.ndarray | None = None) -> None:
        if initial_state is not None:
            self.x_hat = np.copy(initial_state).astype(np.float64)
        else:
            self.x_hat = np.zeros(4, dtype=np.float64)
        self.c.fill(0.0)
        self.h.fill(0.0)
        self.W_a.fill(0.0)

    def continuous_derivatives(
        self,
        x_hat: np.ndarray,
        c: np.ndarray,
        W_a: np.ndarray,
        y: np.ndarray,
        u: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes continuous derivatives:
            dx_hat/dt in R^4, dc/dt in R^h, dW_a/dt in R^{2 x h}
        """
        cfg = self.config

        # 1. Feature input z = [x_hat, u]
        z = np.array([x_hat[0], x_hat[1], x_hat[2], x_hat[3], u], dtype=np.float64)

        # 2. Gate activations
        i_gate = _sigmoid(self.W_i @ z + self.b_i)
        f_gate = _sigmoid(self.W_f @ z + self.b_f)
        o_gate = _sigmoid(self.W_o @ z + self.b_o)
        g_gate = np.tanh(self.W_g @ z + self.b_g)

        # 3. Cell state derivative: dc/dt = -(1 - f)*c + i*g
        dc = -(1.0 - f_gate) * c + i_gate * g_gate

        # 4. Hidden state: h = o * tanh(c)
        h_state = o_gate * np.tanh(c)

        # 5. Acceleration prediction: a_hat = W_a * h + b_a
        a_hat = W_a @ h_state + self.b_a  # shape (2,)

        # 6. Innovation / output errors
        e1 = float(y[0] - x_hat[0])
        e2 = _wrap_angle(float(y[1] - x_hat[2]))
        e_y = np.array([e1, e2], dtype=np.float64)

        # 7. State derivatives
        dx_hat = np.zeros(4, dtype=np.float64)
        dx_hat[0] = x_hat[1] + cfg.l1 * e1
        dx_hat[1] = a_hat[0] + cfg.l2 * e1
        dx_hat[2] = x_hat[3] + cfg.l3 * e2
        dx_hat[3] = a_hat[1] + cfg.l4 * e2

        # 8. Analytical Lyapunov adaptation law for W_a:
        # dW_a / dt = Gamma * [ e_y * h^T ] - sigma_leak * W_a
        dW_a = cfg.gamma_a * np.outer(e_y, h_state) - cfg.sigma_leak * W_a

        return dx_hat, dc, dW_a

    def update(self, y: np.ndarray, u: float, dt: float) -> np.ndarray:
        """
        Advances the continuous Lb-LSTM observer over time interval dt using RK4.
        """
        # Stage 1
        dx1, dc1, dwa1 = self.continuous_derivatives(self.x_hat, self.c, self.W_a, y, u)

        # Stage 2
        x2 = self.x_hat + 0.5 * dt * dx1
        c2 = self.c + 0.5 * dt * dc1
        wa2 = self.W_a + 0.5 * dt * dwa1
        dx2, dc2, dwa2 = self.continuous_derivatives(x2, c2, wa2, y, u)

        # Stage 3
        x3 = self.x_hat + 0.5 * dt * dx2
        c3 = self.c + 0.5 * dt * dc2
        wa3 = self.W_a + 0.5 * dt * dwa2
        dx3, dc3, dwa3 = self.continuous_derivatives(x3, c3, wa3, y, u)

        # Stage 4
        x4 = self.x_hat + dt * dx3
        c4 = self.c + dt * dc3
        wa4 = self.W_a + dt * dwa3
        dx4, dc4, dwa4 = self.continuous_derivatives(x4, c4, wa4, y, u)

        # Assemble RK4 updates
        self.x_hat += (dt / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
        self.c += (dt / 6.0) * (dc1 + 2.0 * dc2 + 2.0 * dc3 + dc4)
        self.W_a += (dt / 6.0) * (dwa1 + 2.0 * dwa2 + 2.0 * dwa3 + dwa4)

        # Update cached hidden state
        z = np.array([self.x_hat[0], self.x_hat[1], self.x_hat[2], self.x_hat[3], u])
        o_gate = _sigmoid(self.W_o @ z + self.b_o)
        self.h = o_gate * np.tanh(self.c)

        self.x_hat[2] = _wrap_angle(self.x_hat[2])
        return np.copy(self.x_hat)
