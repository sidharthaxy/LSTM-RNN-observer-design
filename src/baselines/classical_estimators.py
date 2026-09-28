"""
Classical and Neural Baseline Estimators for Cart-Inverted Pendulum Rig.

Implements:
1. DirtyDerivativeFilter:
   Standard 1st-order dirty derivative: s / (tau_d * s + 1).
2. ButterworthDifferentiator:
   2nd-order Butterworth filter differentiator matching the Feedback 33-936S manual:
   H(s) = (1e4 * s) / (s^2 + 70.7 * s + 1e4).
3. ContinuousDiscreteEKF:
   Continuous-Discrete Extended Kalman Filter utilizing analytical Jacobians.
4. ContinuousShallowRNNObserver:
   Continuous Shallow Dynamic RNN Observer based on Dinh et al. (2014)
   with real-time analytical Lyapunov weight adaptation law.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from src.plant.pendulum_plant import PendulumPlant


class DirtyDerivativeFilter:
    """
    First-order dirty derivative filter:
        H(s) = s / (tau_d * s + 1)
    Approximates derivative at low frequencies while attenuating high-frequency noise.
    State-space realization:
        dz/dt = -(1 / tau_d) * z + (1 / tau_d) * u
        y = (u - z) / tau_d
    """

    def __init__(self, tau_d: float = 0.02) -> None:
        if tau_d <= 0:
            raise ValueError("tau_d must be strictly positive.")
        self.tau_d = tau_d
        self.z: float = 0.0
        self.initialized: bool = False

    def reset(self, initial_u: float = 0.0) -> None:
        self.z = initial_u
        self.initialized = True

    def update(self, u: float, dt: float) -> float:
        if not self.initialized:
            self.reset(u)
            return 0.0

        # Continuous ODE update via RK2
        dz1 = -(self.z - u) / self.tau_d
        z_mid = self.z + 0.5 * dt * dz1
        dz2 = -(z_mid - u) / self.tau_d
        self.z += dt * dz2

        derivative_estimate = (u - self.z) / self.tau_d
        return float(derivative_estimate)


class ButterworthDifferentiator:
    """
    Second-order Butterworth filter differentiator matching the Feedback 33-936S manual:
        H(s) = s * [ 1e4 / (s^2 + 70.7*s + 1e4) ]
             = (10000 * s) / (s^2 + 70.7*s + 10000)

    Natural frequency: omega_n = 100 rad/s (~15.9 Hz)
    Damping ratio: zeta = 70.7 / (2 * 100) = 0.3535 * 2 ~ 0.7071 (Butterworth maximally flat)

    Continuous-time state-space realization:
        z1_dot = z2
        z2_dot = -10000*z1 - 70.7*z2 + 10000*u
        y = z2  (the estimated derivative du/dt)
    """

    def __init__(
        self,
        omega_n_sq: float = 1.0e4,
        two_zeta_omega_n: float = 70.7,
    ) -> None:
        self.omega_n_sq = omega_n_sq
        self.two_zeta_omega_n = two_zeta_omega_n
        self.z1: float = 0.0
        self.z2: float = 0.0
        self.initialized: bool = False

    def reset(self, initial_u: float = 0.0) -> None:
        self.z1 = initial_u
        self.z2 = 0.0
        self.initialized = True

    def state_derivative(self, z: np.ndarray, u: float) -> np.ndarray:
        dz1 = z[1]
        dz2 = -self.omega_n_sq * z[0] - self.two_zeta_omega_n * z[1] + self.omega_n_sq * u
        return np.array([dz1, dz2], dtype=np.float64)

    def update(self, u: float, dt: float) -> float:
        """Advance filter by dt using RK4 integration."""
        if not self.initialized:
            self.reset(u)
            return 0.0

        z = np.array([self.z1, self.z2], dtype=np.float64)
        k1 = self.state_derivative(z, u)
        k2 = self.state_derivative(z + 0.5 * dt * k1, u)
        k3 = self.state_derivative(z + 0.5 * dt * k2, u)
        k4 = self.state_derivative(z + dt * k3, u)

        z_next = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.z1 = float(z_next[0])
        self.z2 = float(z_next[1])
        return self.z2


@dataclass
class EKFConfig:
    """Continuous-Discrete Extended Kalman Filter tuning matrices."""
    # Process noise covariance diagonal: [x, x_dot, theta, theta_dot]
    q_diag: Tuple[float, float, float, float] = (1e-4, 1.0, 1e-4, 5.0)
    # Measurement noise covariance diagonal: [x_meas, theta_meas]
    r_diag: Tuple[float, float] = (1e-5, 1e-4)
    # Initial error covariance diagonal
    p0_diag: Tuple[float, float, float, float] = (1e-3, 0.1, 1e-3, 0.5)


class ContinuousDiscreteEKF:
    """
    Continuous-Discrete Extended Kalman Filter for the Feedback 33-936S rig.
    - Continuous propagation via RK4 integration of non-linear plant equations.
    - Error covariance propagation via analytical Jacobians A(x, u).
    - Discrete measurement update from optical encoder measurements y = [x, theta]^T.
    - Angle unwrapping / wrapping to prevent discontinuity at +/- pi.
    - Joseph-form covariance update for numerical symmetry and positive-definiteness.
    """

    def __init__(
        self,
        plant: PendulumPlant | None = None,
        config: EKFConfig | None = None,
    ) -> None:
        self.plant = plant or PendulumPlant()
        self.config = config or EKFConfig()

        self.x_hat = np.zeros(4, dtype=np.float64)
        self.P = np.diag(self.config.p0_diag).astype(np.float64)
        self.Q = np.diag(self.config.q_diag).astype(np.float64)
        self.R = np.diag(self.config.r_diag).astype(np.float64)

        # Measurement matrix: y = [x, theta]^T -> H = [[1, 0, 0, 0], [0, 0, 1, 0]]
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=np.float64)

    def reset(
        self,
        initial_state: np.ndarray | None = None,
        initial_cov: np.ndarray | None = None,
    ) -> None:
        if initial_state is not None:
            self.x_hat = np.copy(initial_state).astype(np.float64)
        else:
            self.x_hat = np.zeros(4, dtype=np.float64)

        if initial_cov is not None:
            self.P = np.copy(initial_cov).astype(np.float64)
        else:
            self.P = np.diag(self.config.p0_diag).astype(np.float64)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi)."""
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def predict(self, force: float, dt: float) -> np.ndarray:
        """
        Continuous propagation of state estimate and error covariance over dt.
        """
        # 1. State propagation via RK4
        k1 = self.plant.state_derivative(self.x_hat, force)
        k2 = self.plant.state_derivative(self.x_hat + 0.5 * dt * k1, force)
        k3 = self.plant.state_derivative(self.x_hat + 0.5 * dt * k2, force)
        k4 = self.plant.state_derivative(self.x_hat + dt * k3, force)
        self.x_hat += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # 2. Covariance propagation: dP/dt = A*P + P*A^T + Q
        # Second-order matrix transition: Phi = I + A*dt + 0.5*(A*dt)^2
        A, _ = self.plant.analytical_jacobian(self.x_hat, force)
        I4 = np.eye(4, dtype=np.float64)
        Phi = I4 + A * dt + 0.5 * (A @ A) * (dt ** 2)

        self.P = Phi @ self.P @ Phi.T + self.Q * dt
        # Symmetrize
        self.P = 0.5 * (self.P + self.P.T)
        return np.copy(self.x_hat)

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Discrete measurement correction:
        measurement = [x_meas, theta_meas]^T
        """
        # Innovation
        y_pred = self.H @ self.x_hat
        residual = measurement - y_pred
        # Wrap theta residual
        residual[1] = self._wrap_angle(residual[1])

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x_hat += K @ residual
        self.x_hat[2] = self._wrap_angle(self.x_hat[2])

        # Joseph form covariance update for positive-definiteness
        I4 = np.eye(4, dtype=np.float64)
        I_KH = I4 - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return np.copy(self.x_hat)

    def step(self, force: float, measurement: np.ndarray, dt: float) -> np.ndarray:
        """Single Predict-Update cycle."""
        self.predict(force, dt)
        return self.update(measurement)


@dataclass
class RNNConfig:
    """Tuning parameters for Continuous Shallow Dynamic RNN Observer (Dinh et al., 2014)."""
    hidden_dim: int = 16
    gamma1: float = 5.0          # Adaptation gain for cart velocity network
    gamma2: float = 10.0         # Adaptation gain for pendulum angular velocity network
    sigma_leak: float = 0.01     # Weight leakage coefficient (e-modification for boundedness)
    # Output injection gains (Luenberger gains for position errors)
    l1: float = 40.0             # Cart position error gain -> x_dot
    l2: float = 200.0            # Cart position error gain -> x_ddot
    l3: float = 40.0             # Pendulum angle error gain -> theta_dot
    l4: float = 200.0            # Pendulum angle error gain -> theta_ddot
    seed: int = 42


class ContinuousShallowRNNObserver:
    """
    Continuous Shallow Dynamic RNN Observer (Dinh et al., 2014)
    State structure:
        x_hat_dot_1 = x_hat_2 + l1 * (y1 - x_hat_1)
        x_hat_dot_2 = f_nom_1(x_hat, u) + W1^T * tanh(V1 * x_hat) + l2 * (y1 - x_hat_1)
        x_hat_dot_3 = x_hat_4 + l3 * (y2 - x_hat_3)
        x_hat_dot_4 = f_nom_2(x_hat, u) + W2^T * tanh(V2 * x_hat) + l4 * (y2 - x_hat_3)

    Weight adaptation law:
        W1_dot = gamma1 * tanh(V1 * x_hat) * (y1 - x_hat_1) - sigma_leak * W1
        W2_dot = gamma2 * tanh(V2 * x_hat) * (y2 - x_hat_3) - sigma_leak * W2
    """

    def __init__(
        self,
        plant: PendulumPlant | None = None,
        config: RNNConfig | None = None,
    ) -> None:
        self.plant = plant or PendulumPlant()
        self.config = config or RNNConfig()

        rng = np.random.default_rng(self.config.seed)
        h = self.config.hidden_dim

        # Fixed input projection weights V in R^{h x 4}
        self.V1 = rng.normal(0.0, 0.5, size=(h, 4)).astype(np.float64)
        self.V2 = rng.normal(0.0, 0.5, size=(h, 4)).astype(np.float64)

        # Adaptive output weights W in R^h
        self.W1 = np.zeros(h, dtype=np.float64)
        self.W2 = np.zeros(h, dtype=np.float64)

        self.x_hat = np.zeros(4, dtype=np.float64)

    def reset(self, initial_state: np.ndarray | None = None) -> None:
        if initial_state is not None:
            self.x_hat = np.copy(initial_state).astype(np.float64)
        else:
            self.x_hat = np.zeros(4, dtype=np.float64)
        self.W1.fill(0.0)
        self.W2.fill(0.0)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def derivatives(
        self,
        x_hat: np.ndarray,
        W1: np.ndarray,
        W2: np.ndarray,
        y: np.ndarray,
        force: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes continuous-time derivatives for state and weights.
        y = [x_meas, theta_meas]
        """
        # Nominal model forward accelerations
        f_nom_x_ddot, f_nom_theta_ddot = self.plant.forward_dynamics(x_hat, force)

        # Neural activations
        phi1 = np.tanh(self.V1 @ x_hat)  # shape (h,)
        phi2 = np.tanh(self.V2 @ x_hat)  # shape (h,)

        # Output errors
        e1 = float(y[0] - x_hat[0])
        e2 = self._wrap_angle(float(y[1] - x_hat[2]))

        cfg = self.config
        # State derivatives
        dx_hat = np.zeros(4, dtype=np.float64)
        dx_hat[0] = x_hat[1] + cfg.l1 * e1
        dx_hat[1] = f_nom_x_ddot + float(W1 @ phi1) + cfg.l2 * e1
        dx_hat[2] = x_hat[3] + cfg.l3 * e2
        dx_hat[3] = f_nom_theta_ddot + float(W2 @ phi2) + cfg.l4 * e2

        # Weight adaptation derivatives
        dW1 = cfg.gamma1 * phi1 * e1 - cfg.sigma_leak * W1
        dW2 = cfg.gamma2 * phi2 * e2 - cfg.sigma_leak * W2

        return dx_hat, dW1, dW2

    def update(self, y: np.ndarray, force: float, dt: float) -> np.ndarray:
        """
        Advances the RNN observer by dt using RK4 integration.
        y = [x_meas, theta_meas]
        """
        # RK4 step
        dx1, dw1_1, dw2_1 = self.derivatives(self.x_hat, self.W1, self.W2, y, force)

        x_k2 = self.x_hat + 0.5 * dt * dx1
        w1_k2 = self.W1 + 0.5 * dt * dw1_1
        w2_k2 = self.W2 + 0.5 * dt * dw2_1
        dx2, dw1_2, dw2_2 = self.derivatives(x_k2, w1_k2, w2_k2, y, force)

        x_k3 = self.x_hat + 0.5 * dt * dx2
        w1_k3 = self.W1 + 0.5 * dt * dw1_2
        w2_k3 = self.W2 + 0.5 * dt * dw2_2
        dx3, dw1_3, dw2_3 = self.derivatives(x_k3, w1_k3, w2_k3, y, force)

        x_k4 = self.x_hat + dt * dx3
        w1_k4 = self.W1 + dt * dw1_3
        w2_k4 = self.W2 + dt * dw2_3
        dx4, dw1_4, dw2_4 = self.derivatives(x_k4, w1_k4, w2_k4, y, force)

        # Update
        self.x_hat += (dt / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
        self.W1 += (dt / 6.0) * (dw1_1 + 2.0 * dw1_2 + 2.0 * dw1_3 + dw1_4)
        self.W2 += (dt / 6.0) * (dw2_1 + 2.0 * dw2_2 + 2.0 * dw2_3 + dw2_4)

        self.x_hat[2] = self._wrap_angle(self.x_hat[2])
        return np.copy(self.x_hat)
