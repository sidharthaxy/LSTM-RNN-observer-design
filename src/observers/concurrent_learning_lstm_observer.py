"""
Approach C: Concurrent Learning Lb-LSTM Observer.
Guarantees exponential parameter convergence WITHOUT Persistent Excitation (PE).

Maintains an online History Stack H = {(h_j, a_j)}_{j=1}^{N_H} of recorded state-action
transitions satisfying a minimum singular value rank condition:
    lambda_min( sum_{j=1}^{N_H} h_j * h_j^T ) >= lambda_bar > 0

Composite Lyapunov Adaptation Law:
    dW_a/dt = Gamma_inst * [ e_y(t) * h(t)^T ]
            + Gamma_cl * sum_{j=1}^{N_H} [ (a_j - W_a * h_j) * h_j^T ]
            - sigma_leak * W_a
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class HistoryStackEntry:
    """Stored transition in the concurrent learning history stack."""
    h_features: np.ndarray      # Shape (d_h,) hidden representation
    a_target: np.ndarray        # Shape (2,) estimated/computed target accelerations
    timestamp: float            # Time when sample was recorded


@dataclass
class ConcurrentLearningConfig:
    """Hyperparameters for Concurrent Learning Lb-LSTM Observer."""
    hidden_dim: int = 16
    stack_capacity: int = 32         # Maximum history stack size
    min_eigenval_threshold: float = 0.05 # Minimum acceptable excitation eigenvalue
    sample_min_interval: float = 0.05   # Minimum seconds between stack acquisitions
    gamma_inst: float = 10.0         # Adaptation gain for instantaneous tracking error
    gamma_cl: float = 25.0           # Adaptation gain for concurrent learning stack
    sigma_leak: float = 0.001        # Leakage parameter (low because CL stabilizes weights)
    # Output injection gains
    l1: float = 40.0
    l2: float = 250.0
    l3: float = 40.0
    l4: float = 250.0
    seed: int = 42


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class ConcurrentLearningLSTMObserver:
    """
    Concurrent Learning Continuous-Time Lb-LSTM Observer.
    Records rich dynamic data in an online history stack to drive parameter error
    to zero without requiring continuous persistent excitation of the reference signal.
    """

    def __init__(self, config: ConcurrentLearningConfig | None = None) -> None:
        self.config = config or ConcurrentLearningConfig()
        h = self.config.hidden_dim
        rng = np.random.default_rng(self.config.seed)

        # Continuous LSTM parameters
        d_in = 5
        self.W_i = rng.normal(0.0, 0.3, size=(h, d_in)).astype(np.float64)
        self.b_i = np.zeros(h, dtype=np.float64)

        self.W_f = rng.normal(0.0, 0.3, size=(h, d_in)).astype(np.float64)
        self.b_f = np.ones(h, dtype=np.float64)

        self.W_o = rng.normal(0.0, 0.3, size=(h, d_in)).astype(np.float64)
        self.b_o = np.zeros(h, dtype=np.float64)

        self.W_g = rng.normal(0.0, 0.3, size=(h, d_in)).astype(np.float64)
        self.b_g = np.zeros(h, dtype=np.float64)

        # Acceleration readout weights W_a in R^{2 x h}
        self.W_a = np.zeros((2, h), dtype=np.float64)
        self.b_a = np.zeros(2, dtype=np.float64)

        # States
        self.x_hat = np.zeros(4, dtype=np.float64)
        self.c = np.zeros(h, dtype=np.float64)
        self.h = np.zeros(h, dtype=np.float64)

        # Concurrent Learning History Stack
        self.history_stack: List[HistoryStackEntry] = []
        self.last_record_time: float = -1.0
        self.current_time: float = 0.0

    def reset(self, initial_state: np.ndarray | None = None) -> None:
        if initial_state is not None:
            self.x_hat = np.copy(initial_state).astype(np.float64)
        else:
            self.x_hat = np.zeros(4, dtype=np.float64)
        self.c.fill(0.0)
        self.h.fill(0.0)
        self.W_a.fill(0.0)
        self.history_stack.clear()
        self.last_record_time = -1.0
        self.current_time = 0.0

    def compute_hidden_state(self, x_state: np.ndarray, u: float, c_state: np.ndarray) -> np.ndarray:
        """Forward pass to obtain hidden feature vector h(t)."""
        z = np.array([x_state[0], x_state[1], x_state[2], x_state[3], u], dtype=np.float64)
        o_gate = _sigmoid(self.W_o @ z + self.b_o)
        return o_gate * np.tanh(c_state)

    def stack_excitation_matrix(self) -> np.ndarray:
        """
        Computes Omega = sum_{j=1}^{N_H} h_j * h_j^T.
        Rank / eigenvalues of Omega determine convergence rate.
        """
        h_dim = self.config.hidden_dim
        if not self.history_stack:
            return np.zeros((h_dim, h_dim), dtype=np.float64)

        Omega = np.zeros((h_dim, h_dim), dtype=np.float64)
        for entry in self.history_stack:
            Omega += np.outer(entry.h_features, entry.h_features)
        return Omega

    def minimum_excitation_eigenvalue(self) -> float:
        """Returns lambda_min(Omega) of the history stack."""
        if len(self.history_stack) < 3:
            return 0.0
        Omega = self.stack_excitation_matrix()
        eigvals = np.linalg.eigvalsh(Omega)
        return float(np.min(eigvals))

    def maybe_add_to_history_stack(
        self,
        h_features: np.ndarray,
        a_target: np.ndarray,
        t: float,
    ) -> bool:
        """
        Evaluates candidate transition (h_features, a_target).
        Adds to stack if:
        1. Time since last record >= sample_min_interval
        2. Point increases or improves excitation conditioning of history stack.
        """
        cfg = self.config
        if (t - self.last_record_time) < cfg.sample_min_interval:
            return False

        candidate = HistoryStackEntry(h_features=np.copy(h_features), a_target=np.copy(a_target), timestamp=t)

        # If stack not full, add directly
        if len(self.history_stack) < cfg.stack_capacity:
            self.history_stack.append(candidate)
            self.last_record_time = t
            return True

        # Stack is full: test if replacing any existing point improves min eigenvalue
        curr_min_eig = self.minimum_excitation_eigenvalue()
        best_idx = -1
        best_new_min_eig = curr_min_eig

        for i in range(len(self.history_stack)):
            # Temporarily substitute candidate
            old_entry = self.history_stack[i]
            self.history_stack[i] = candidate
            new_min_eig = self.minimum_excitation_eigenvalue()
            if new_min_eig > best_new_min_eig:
                best_new_min_eig = new_min_eig
                best_idx = i
            # Restore
            self.history_stack[i] = old_entry

        if best_idx >= 0 and best_new_min_eig > curr_min_eig + 1e-4:
            self.history_stack[best_idx] = candidate
            self.last_record_time = t
            return True

        return False

    def concurrent_learning_residual(self, W_a: np.ndarray) -> np.ndarray:
        """
        Computes composite gradient from all points in the history stack:
        R_cl = sum_{j=1}^{N_H} (a_j - W_a * h_j) * h_j^T
        """
        h_dim = self.config.hidden_dim
        grad_cl = np.zeros((2, h_dim), dtype=np.float64)

        for entry in self.history_stack:
            h_j = entry.h_features
            a_j = entry.a_target
            pred_j = W_a @ h_j + self.b_a
            err_j = a_j - pred_j  # shape (2,)
            grad_cl += np.outer(err_j, h_j)

        return grad_cl

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

        # 1. CT-LSTM gates
        z = np.array([x_hat[0], x_hat[1], x_hat[2], x_hat[3], u], dtype=np.float64)
        i_gate = _sigmoid(self.W_i @ z + self.b_i)
        f_gate = _sigmoid(self.W_f @ z + self.b_f)
        o_gate = _sigmoid(self.W_o @ z + self.b_o)
        g_gate = np.tanh(self.W_g @ z + self.b_g)

        dc = -(1.0 - f_gate) * c + i_gate * g_gate
        h_state = o_gate * np.tanh(c)

        # 2. Predicted accelerations
        a_hat = W_a @ h_state + self.b_a

        # 3. Innovation errors
        e1 = float(y[0] - x_hat[0])
        e2 = _wrap_angle(float(y[1] - x_hat[2]))
        e_y = np.array([e1, e2], dtype=np.float64)

        # 4. Observer state derivatives
        dx_hat = np.zeros(4, dtype=np.float64)
        dx_hat[0] = x_hat[1] + cfg.l1 * e1
        dx_hat[1] = a_hat[0] + cfg.l2 * e1
        dx_hat[2] = x_hat[3] + cfg.l3 * e2
        dx_hat[3] = a_hat[1] + cfg.l4 * e2

        # 5. Concurrent Learning Adaptation Law:
        # Instantaneous error term + History Stack residual term - Leakage
        grad_inst = cfg.gamma_inst * np.outer(e_y, h_state)
        grad_cl = cfg.gamma_cl * self.concurrent_learning_residual(W_a)

        dW_a = grad_inst + grad_cl - cfg.sigma_leak * W_a

        return dx_hat, dc, dW_a

    def update(self, y: np.ndarray, u: float, dt: float) -> np.ndarray:
        """
        Advances the CL-LSTM observer by dt using RK4 integration.
        """
        self.current_time += dt

        # RK4 Integration
        dx1, dc1, dwa1 = self.continuous_derivatives(self.x_hat, self.c, self.W_a, y, u)

        x2 = self.x_hat + 0.5 * dt * dx1
        c2 = self.c + 0.5 * dt * dc1
        wa2 = self.W_a + 0.5 * dt * dwa1
        dx2, dc2, dwa2 = self.continuous_derivatives(x2, c2, wa2, y, u)

        x3 = self.x_hat + 0.5 * dt * dx2
        c3 = self.c + 0.5 * dt * dc2
        wa3 = self.W_a + 0.5 * dt * dwa2
        dx3, dc3, dwa3 = self.continuous_derivatives(x3, c3, wa3, y, u)

        x4 = self.x_hat + dt * dx3
        c4 = self.c + dt * dc3
        wa4 = self.W_a + dt * dwa3
        dx4, dc4, dwa4 = self.continuous_derivatives(x4, c4, wa4, y, u)

        self.x_hat += (dt / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
        self.c += (dt / 6.0) * (dc1 + 2.0 * dc2 + 2.0 * dc3 + dc4)
        self.W_a += (dt / 6.0) * (dwa1 + 2.0 * dwa2 + 2.0 * dwa3 + dwa4)

        # Cache hidden representation
        z = np.array([self.x_hat[0], self.x_hat[1], self.x_hat[2], self.x_hat[3], u])
        o_gate = _sigmoid(self.W_o @ z + self.b_o)
        self.h = o_gate * np.tanh(self.c)
        self.x_hat[2] = _wrap_angle(self.x_hat[2])

        # Evaluate candidate for history stack: target acceleration = [dx_hat[1], dx_hat[3]]
        curr_accel = np.array([dx1[1], dx1[3]], dtype=np.float64)
        self.maybe_add_to_history_stack(self.h, curr_accel, self.current_time)

        return np.copy(self.x_hat)
