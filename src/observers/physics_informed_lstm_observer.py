"""
Approach B: Physics-Informed PI-LSTM Observer for Feedback 33-936S Rig.

Enforces:
1. Inertia Matrix Positive-Definiteness:
   M_hat(q) = M_hat(q)^T > 0 strictly everywhere via analytical Euler-Lagrange structure.
2. Coriolis Skew-Symmetry:
   z^T * (dM/dt - 2*C(q, q_dot)) * z = 0 for all z in R^2.
3. Neural Residual Identification:
   Continuous-time LSTM approximates unmodeled generalized forces tau_Delta(q, q_dot, u).
4. Energy-Passivity Lyapunov Stability:
   Lyapunov function V = 0.5 * e_v^T * M(q) * e_v + 0.5 * e_p^T * K_p * e_p + (1/2*Gamma)*tr(W_tilde^T * W_tilde).
   Coriolis skew-symmetry causes (dM/dt - 2*C) cross term to identically vanish!
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np

from src.plant.pendulum_plant import PendulumParameters


@dataclass
class PhysicsInformedLSTMConfig:
    """Hyperparameters for Physics-Informed PI-LSTM Observer."""
    hidden_dim: int = 16
    gamma_delta: float = 20.0       # Adaptation rate for residual neural weights
    sigma_leak: float = 0.005       # Leakage coefficient for weight boundedness
    # Passivity-based injection gains
    kp_cart: float = 150.0          # Position gain for cart
    kv_cart: float = 40.0           # Velocity / damping injection for cart
    kp_pend: float = 250.0          # Angle gain for pendulum
    kv_pend: float = 50.0           # Angular velocity injection for pendulum
    seed: int = 42


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class PhysicsInformedLSTMObserver:
    """
    Physics-Informed Continuous-Time LSTM Adaptive Observer.
    Enforces positive-definite inertia M(q) and Coriolis skew-symmetry (dM/dt - 2*C).
    Learns unmodeled residual generalized forces tau_Delta via continuous LSTM.
    """

    def __init__(
        self,
        plant_params: PendulumParameters | None = None,
        config: PhysicsInformedLSTMConfig | None = None,
    ) -> None:
        self.params = plant_params or PendulumParameters()
        self.config = config or PhysicsInformedLSTMConfig()
        h = self.config.hidden_dim
        rng = np.random.default_rng(self.config.seed)

        # Input to LSTM: z(t) = [q_hat, q_dot_hat, u] in R^5
        d_in = 5
        self.W_i = rng.normal(0.0, 0.3, size=(h, d_in)).astype(np.float64)
        self.b_i = np.zeros(h, dtype=np.float64)

        self.W_f = rng.normal(0.0, 0.3, size=(h, d_in)).astype(np.float64)
        self.b_f = np.ones(h, dtype=np.float64)

        self.W_o = rng.normal(0.0, 0.3, size=(h, d_in)).astype(np.float64)
        self.b_o = np.zeros(h, dtype=np.float64)

        self.W_g = rng.normal(0.0, 0.3, size=(h, d_in)).astype(np.float64)
        self.b_g = np.zeros(h, dtype=np.float64)

        # Adaptive residual force readout: W_delta in R^{2 x h}
        # maps h(t) -> tau_delta = [tau_delta_x, tau_delta_theta]^T
        self.W_delta = np.zeros((2, h), dtype=np.float64)

        # Observer states
        self.x_hat = np.zeros(4, dtype=np.float64)  # [x, x_dot, theta, theta_dot]
        self.c = np.zeros(h, dtype=np.float64)      # LSTM cell state
        self.h = np.zeros(h, dtype=np.float64)      # LSTM hidden state

    def reset(self, initial_state: np.ndarray | None = None) -> None:
        if initial_state is not None:
            self.x_hat = np.copy(initial_state).astype(np.float64)
        else:
            self.x_hat = np.zeros(4, dtype=np.float64)
        self.c.fill(0.0)
        self.h.fill(0.0)
        self.W_delta.fill(0.0)

    def inertia_matrix(self, theta: float) -> np.ndarray:
        """
        Symmetric positive-definite inertia matrix M(theta):
        [[M + m, m*l*cos(theta)], [m*l*cos(theta), I + m*l^2]]
        """
        p = self.params
        m11 = p.M + p.m
        m12 = p.m * p.l * np.cos(theta)
        m22 = p.I + p.m * (p.l ** 2)
        return np.array([[m11, m12], [m12, m22]], dtype=np.float64)

    def coriolis_matrix(self, theta: float, theta_dot: float) -> np.ndarray:
        """
        Coriolis matrix C(q, q_dot) derived from Christoffel symbols:
        C = [[0, -m*l*theta_dot*sin(theta)], [0, 0]]
        Satisfies skew-symmetry: z^T * (dM/dt - 2*C) * z = 0 for all z.
        """
        p = self.params
        c12 = -p.m * p.l * theta_dot * np.sin(theta)
        return np.array([[0.0, c12], [0.0, 0.0]], dtype=np.float64)

    def gravity_vector(self, theta: float) -> np.ndarray:
        """Gravitational torque G(q) = [0, -m*g*l*sin(theta)]^T."""
        p = self.params
        g2 = -p.m * p.g * p.l * np.sin(theta)
        return np.array([0.0, g2], dtype=np.float64)

    def viscous_damping_matrix(self) -> np.ndarray:
        """Viscous friction matrix D = diag(b, d)."""
        p = self.params
        return np.diag([p.b, p.d]).astype(np.float64)

    def continuous_derivatives(
        self,
        x_hat: np.ndarray,
        c: np.ndarray,
        W_delta: np.ndarray,
        y: np.ndarray,
        u: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Physics-informed ODE derivatives:
        dx_hat/dt in R^4, dc/dt in R^h, dW_delta/dt in R^{2 x h}
        """
        cfg = self.config
        p = self.params

        # 1. CT-LSTM forward step
        z = np.array([x_hat[0], x_hat[1], x_hat[2], x_hat[3], u], dtype=np.float64)
        i_gate = _sigmoid(self.W_i @ z + self.b_i)
        f_gate = _sigmoid(self.W_f @ z + self.b_f)
        o_gate = _sigmoid(self.W_o @ z + self.b_o)
        g_gate = np.tanh(self.W_g @ z + self.b_g)

        dc = -(1.0 - f_gate) * c + i_gate * g_gate
        h_state = o_gate * np.tanh(c)

        # Residual force prediction from LSTM: tau_delta in R^2
        tau_delta = W_delta @ h_state

        # 2. Position tracking errors e_q = [x_meas - x_hat, theta_meas - theta_hat]
        e_pos_x = float(y[0] - x_hat[0])
        e_pos_th = _wrap_angle(float(y[1] - x_hat[2]))
        e_q = np.array([e_pos_x, e_pos_th], dtype=np.float64)

        # Approximate velocity innovation e_v via passivity output injection
        # e_v_eff = K_p / K_v * e_q
        tau_injection = np.array([
            cfg.kp_cart * e_pos_x,
            cfg.kp_pend * e_pos_th,
        ], dtype=np.float64)

        # 3. Structural Euler-Lagrange equations:
        # M(theta) * q_ddot + C(q, q_dot)*q_dot + G(q) + D*q_dot = B*u + tau_delta + tau_injection
        th = x_hat[2]
        q_dot = np.array([x_hat[1], x_hat[3]], dtype=np.float64)

        M = self.inertia_matrix(th)
        C = self.coriolis_matrix(th, q_dot[1])
        G = self.gravity_vector(th)
        D = self.viscous_damping_matrix()

        # Actuator input B*u
        Bu = np.array([float(np.clip(u, -p.F_max, p.F_max)), 0.0], dtype=np.float64)

        # Net generalized torque
        tau_rhs = Bu + tau_delta + tau_injection - (C @ q_dot) - G - (D @ q_dot)

        # Solve for accelerations: q_ddot = M^-1 * tau_rhs
        q_ddot = np.linalg.solve(M, tau_rhs)

        # Assemble state derivatives
        dx_hat = np.zeros(4, dtype=np.float64)
        dx_hat[0] = x_hat[1] + (cfg.kp_cart / cfg.kv_cart) * e_pos_x
        dx_hat[1] = q_ddot[0]
        dx_hat[2] = x_hat[3] + (cfg.kp_pend / cfg.kv_pend) * e_pos_th
        dx_hat[3] = q_ddot[1]

        # 4. Energy-Passivity Lyapunov adaptation law for W_delta:
        # dW_delta/dt = Gamma * [ e_q * h^T ] - sigma_leak * W_delta
        dW_delta = cfg.gamma_delta * np.outer(e_q, h_state) - cfg.sigma_leak * W_delta

        return dx_hat, dc, dW_delta

    def update(self, y: np.ndarray, u: float, dt: float) -> np.ndarray:
        """
        Advances the PI-LSTM observer by dt using RK4.
        """
        # RK4 Integration
        dx1, dc1, dwd1 = self.continuous_derivatives(self.x_hat, self.c, self.W_delta, y, u)

        x2 = self.x_hat + 0.5 * dt * dx1
        c2 = self.c + 0.5 * dt * dc1
        wd2 = self.W_delta + 0.5 * dt * dwd1
        dx2, dc2, dwd2 = self.continuous_derivatives(x2, c2, wd2, y, u)

        x3 = self.x_hat + 0.5 * dt * dx2
        c3 = self.c + 0.5 * dt * dc2
        wd3 = self.W_delta + 0.5 * dt * dwd2
        dx3, dc3, dwd3 = self.continuous_derivatives(x3, c3, wd3, y, u)

        x4 = self.x_hat + dt * dx3
        c4 = self.c + dt * dc3
        wd4 = self.W_delta + dt * dwd3
        dx4, dc4, dwd4 = self.continuous_derivatives(x4, c4, wd4, y, u)

        self.x_hat += (dt / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
        self.c += (dt / 6.0) * (dc1 + 2.0 * dc2 + 2.0 * dc3 + dc4)
        self.W_delta += (dt / 6.0) * (dwd1 + 2.0 * dwd2 + 2.0 * dwd3 + dwd4)

        z = np.array([self.x_hat[0], self.x_hat[1], self.x_hat[2], self.x_hat[3], u])
        o_gate = _sigmoid(self.W_o @ z + self.b_o)
        self.h = o_gate * np.tanh(self.c)

        self.x_hat[2] = _wrap_angle(self.x_hat[2])
        return np.copy(self.x_hat)
