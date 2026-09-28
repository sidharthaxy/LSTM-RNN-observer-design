"""
Feedback Instruments 33-936S Digital Cart-Inverted Pendulum Rig.
Mathematical Ground Truth Plant Simulation.

Equations of Motion from Manufacturer Manual:
(1) (M + m)*x_ddot + b*x_dot + m*l*theta_ddot*cos(theta) - m*l*theta_dot^2*sin(theta) = F
(2) (I + m*l^2)*theta_ddot - m*g*l*sin(theta) + m*l*x_ddot*cos(theta) + d*theta_dot = 0

System Constants:
- M = 2.4 kg (cart mass)
- m = 0.23 kg (pendulum rod mass)
- l = 0.36 m (distance from pivot to center of mass)
- I = 0.099 kg*m^2 (pendulum moment of inertia about pivot)
- b = 0.05 Ns/m (cart viscous damping)
- d = 0.005 Nms/rad (pendulum joint viscous damping)
- g = 9.81 m/s^2 (gravitational acceleration)

Actuator and Physical Constraints:
- Force saturation: F in [-20.0, +20.0] N
- Cart track limits: x in [-0.5, +0.5] m
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np


@dataclass(frozen=True)
class PendulumParameters:
    """Rigorous physical parameters for Feedback 33-936S pendulum rig."""
    M: float = 2.4          # Cart mass [kg]
    m: float = 0.23         # Pendulum mass [kg]
    l: float = 0.36         # Center of mass distance from pivot [m]
    I: float = 0.099        # Pendulum inertia about pivot [kg*m^2]
    b: float = 0.05         # Cart viscous damping [N*s/m]
    d: float = 0.005        # Pendulum joint damping [N*m*s/rad]
    g: float = 9.81         # Gravitational acceleration [m/s^2]
    F_max: float = 20.0     # Actuator force limit (+/-) [N]
    x_limit: float = 0.5    # Track limit (+/-) [m]
    restitution: float = 0.0 # Bumper coefficient of restitution (0 = inelastic)

    @property
    def total_cart_mass(self) -> float:
        return self.M + self.m

    @property
    def total_pendulum_inertia(self) -> float:
        return self.I + self.m * (self.l ** 2)


@dataclass
class PlantState:
    """State vector container for cart-pendulum system."""
    x: float = 0.0          # Cart position [m]
    x_dot: float = 0.0      # Cart velocity [m/s]
    theta: float = 0.0      # Pendulum angle from upright [rad] (0 = upright)
    theta_dot: float = 0.0  # Pendulum angular velocity [rad/s]

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.x_dot, self.theta, self.theta_dot], dtype=np.float64)

    @classmethod
    def from_array(cls, state: np.ndarray) -> PlantState:
        return cls(
            x=float(state[0]),
            x_dot=float(state[1]),
            theta=float(state[2]),
            theta_dot=float(state[3]),
        )


class PendulumPlant:
    """
    Continuous-time dynamic model and forward ODE simulator of the Feedback 33-936S rig.
    State representation:
        x_vec = [x, x_dot, theta, theta_dot]^T = [x_1, x_2, x_3, x_4]^T
    """

    def __init__(self, params: PendulumParameters | None = None) -> None:
        self.params = params or PendulumParameters()
        self.state = PlantState()

    def reset(self, initial_state: PlantState | np.ndarray | None = None) -> PlantState:
        """Reset the internal plant state."""
        if initial_state is None:
            self.state = PlantState()
        elif isinstance(initial_state, np.ndarray):
            self.state = PlantState.from_array(initial_state)
        else:
            self.state = initial_state
        return self.state

    def mass_matrix(self, theta: float) -> np.ndarray:
        """
        Inertia matrix M(theta) = [[M + m, m*l*cos(theta)], [m*l*cos(theta), I + m*l^2]].
        Strictly positive definite for all theta.
        """
        p = self.params
        m11 = p.M + p.m
        m12 = p.m * p.l * np.cos(theta)
        m22 = p.I + p.m * (p.l ** 2)
        return np.array([[m11, m12], [m12, m22]], dtype=np.float64)

    def mass_matrix_determinant(self, theta: float) -> float:
        """
        Analytical determinant:
        det(M) = (M + m)*(I + m*l^2) - (m*l*cos(theta))^2 > 0
        """
        p = self.params
        return (p.M + p.m) * (p.I + p.m * (p.l ** 2)) - (p.m * p.l * np.cos(theta)) ** 2

    def coriolis_gravity_vector(self, state: np.ndarray) -> np.ndarray:
        """
        Returns nonlinear force vector [C(q, q_dot)*q_dot + G(q) + F_viscous].
        Row 1: b*x_dot - m*l*theta_dot^2*sin(theta)
        Row 2: -m*g*l*sin(theta) + d*theta_dot
        """
        p = self.params
        _, x_dot, theta, theta_dot = state
        sin_t = np.sin(theta)
        f1 = p.b * x_dot - p.m * p.l * (theta_dot ** 2) * sin_t
        f2 = -p.m * p.g * p.l * sin_t + p.d * theta_dot
        return np.array([f1, f2], dtype=np.float64)

    def forward_dynamics(
        self, state: np.ndarray, force: float
    ) -> Tuple[float, float]:
        """
        Computes accelerations [x_ddot, theta_ddot] given state and applied force.
        Solves:
        (M + m)*x_ddot + (m*l*cos(theta))*theta_ddot = F - b*x_dot + m*l*theta_dot^2*sin(theta)
        (m*l*cos(theta))*x_ddot + (I + m*l^2)*theta_ddot = m*g*l*sin(theta) - d*theta_dot
        """
        p = self.params
        F_applied = float(np.clip(force, -p.F_max, p.F_max))

        _, x_dot, theta, theta_dot = state
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Right-hand side terms
        h1 = F_applied - p.b * x_dot + p.m * p.l * (theta_dot ** 2) * sin_t
        h2 = p.m * p.g * p.l * sin_t - p.d * theta_dot

        # Mass matrix elements
        m11 = p.M + p.m
        m12 = p.m * p.l * cos_t
        m22 = p.I + p.m * (p.l ** 2)
        det_m = m11 * m22 - m12 * m12

        # Analytic inversion: M^-1 * [h1, h2]^T
        x_ddot = (m22 * h1 - m12 * h2) / det_m
        theta_ddot = (-m12 * h1 + m11 * h2) / det_m

        return float(x_ddot), float(theta_ddot)

    def state_derivative(self, state: np.ndarray, force: float) -> np.ndarray:
        """
        Continuous vector field f(x, u) = [x_dot, x_ddot, theta_dot, theta_ddot]^T.
        """
        x_dot = state[1]
        theta_dot = state[3]
        x_ddot, theta_ddot = self.forward_dynamics(state, force)
        return np.array([x_dot, x_ddot, theta_dot, theta_ddot], dtype=np.float64)

    def analytical_jacobian(
        self, state: np.ndarray, force: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes analytical continuous-time Jacobian matrices:
        A = df/dx in R^{4x4}, B = df/du in R^{4x1}
        Used for Continuous EKF and linearized control designs.
        """
        p = self.params
        F_clipped = float(np.clip(force, -p.F_max, p.F_max))

        _, x_dot, theta, theta_dot = state
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        m11 = p.M + p.m
        m22 = p.I + p.m * (p.l ** 2)
        ml = p.m * p.l
        mgl = ml * p.g

        det_m = m11 * m22 - (ml * cos_t) ** 2
        d_det_m_dtheta = 2.0 * (ml ** 2) * sin_t * cos_t  # = ml^2 * sin(2*theta)

        h1 = F_clipped - p.b * x_dot + ml * (theta_dot ** 2) * sin_t
        h2 = mgl * sin_t - p.d * theta_dot

        # Numerators
        N1 = m22 * h1 - (ml * cos_t) * h2
        N2 = -(ml * cos_t) * h1 + m11 * h2

        # Partial derivatives wrt x_dot (x2)
        # dh1/dx_dot = -b, dh2/dx_dot = 0
        dx_ddot_dxdot = -p.b * m22 / det_m
        dtheta_ddot_dxdot = p.b * (ml * cos_t) / det_m

        # Partial derivatives wrt theta_dot (x4)
        # dh1/dtheta_dot = 2*ml*theta_dot*sin_t, dh2/dtheta_dot = -d
        dh1_dthetadot = 2.0 * ml * theta_dot * sin_t
        dh2_dthetadot = -p.d
        dN1_dthetadot = m22 * dh1_dthetadot - (ml * cos_t) * dh2_dthetadot
        dN2_dthetadot = -(ml * cos_t) * dh1_dthetadot + m11 * dh2_dthetadot
        dx_ddot_dthetadot = dN1_dthetadot / det_m
        dtheta_ddot_dthetadot = dN2_dthetadot / det_m

        # Partial derivatives wrt theta (x3)
        dh1_dtheta = ml * (theta_dot ** 2) * cos_t
        dh2_dtheta = mgl * cos_t
        dN1_dtheta = m22 * dh1_dtheta - ((-ml * sin_t) * h2 + (ml * cos_t) * dh2_dtheta)
        dN2_dtheta = -((-ml * sin_t) * h1 + (ml * cos_t) * dh1_dtheta) + m11 * dh2_dtheta

        dx_ddot_dtheta = (det_m * dN1_dtheta - N1 * d_det_m_dtheta) / (det_m ** 2)
        dtheta_ddot_dtheta = (det_m * dN2_dtheta - N2 * d_det_m_dtheta) / (det_m ** 2)

        # Assemble A matrix
        A = np.zeros((4, 4), dtype=np.float64)
        A[0, 1] = 1.0
        A[1, 1] = dx_ddot_dxdot
        A[1, 2] = dx_ddot_dtheta
        A[1, 3] = dx_ddot_dthetadot
        A[2, 3] = 1.0
        A[3, 1] = dtheta_ddot_dxdot
        A[3, 2] = dtheta_ddot_dtheta
        A[3, 3] = dtheta_ddot_dthetadot

        # Assemble B matrix (df/du)
        # dh1/dF = 1.0, dh2/dF = 0.0
        B = np.zeros((4, 1), dtype=np.float64)
        B[1, 0] = m22 / det_m
        B[3, 0] = -(ml * cos_t) / det_m

        return A, B

    def step(
        self,
        force: float,
        dt: float,
        integrator: Literal["rk4", "euler"] = "rk4",
    ) -> PlantState:
        """
        Advance plant state by dt using chosen numerical integrator,
        enforcing track boundaries and restitution.
        """
        state_arr = self.state.to_array()

        if integrator == "rk4":
            k1 = self.state_derivative(state_arr, force)
            k2 = self.state_derivative(state_arr + 0.5 * dt * k1, force)
            k3 = self.state_derivative(state_arr + 0.5 * dt * k2, force)
            k4 = self.state_derivative(state_arr + dt * k3, force)
            next_state = state_arr + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        elif integrator == "euler":
            k1 = self.state_derivative(state_arr, force)
            next_state = state_arr + dt * k1
        else:
            raise ValueError(f"Unknown integrator '{integrator}'. Choose 'rk4' or 'euler'.")

        # Boundary enforcement on track limits [-x_limit, +x_limit]
        p = self.params
        x = next_state[0]
        x_dot = next_state[1]

        if x > p.x_limit:
            next_state[0] = p.x_limit
            next_state[1] = -p.restitution * x_dot if x_dot > 0 else x_dot
        elif x < -p.x_limit:
            next_state[0] = -p.x_limit
            next_state[1] = -p.restitution * x_dot if x_dot < 0 else x_dot

        self.state = PlantState.from_array(next_state)
        return self.state

    def total_energy(self, state: np.ndarray | None = None) -> float:
        """
        Computes total mechanical energy (Kinetic + Potential).
        Reference: upright inverted position (theta = 0) has V = m*g*l.
        """
        if state is None:
            state = self.state.to_array()
        p = self.params
        _, x_dot, theta, theta_dot = state

        # Kinetic energy:
        # T_cart = 0.5 * M * x_dot^2
        # T_pendulum = 0.5 * m * (x_dot^2 + 2*l*x_dot*theta_dot*cos(theta) + l^2*theta_dot^2) + 0.5 * I * theta_dot^2
        T_cart = 0.5 * p.M * (x_dot ** 2)
        T_pend = (
            0.5 * p.m * (x_dot ** 2 + 2.0 * p.l * x_dot * theta_dot * np.cos(theta) + (p.l ** 2) * (theta_dot ** 2))
            + 0.5 * p.I * (theta_dot ** 2)
        )
        # Potential energy (upright theta=0 is V = +m*g*l, hanging theta=pi is -m*g*l)
        V = p.m * p.g * p.l * np.cos(theta)

        return float(T_cart + T_pend + V)
