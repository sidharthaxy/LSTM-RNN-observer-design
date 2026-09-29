"""
Concurrent-learning history stack with singular-value-maximizing data selection
(Chowdhary & Johnson, "A singular value maximizing data recording algorithm for
concurrent learning", ACC 2011; Chowdhary, Yucelen, Muhlegg & Johnson, IJACSP 2013).

Each entry j stores the raw data needed to re-evaluate the network at the recorded
operating point,
    zeta(t_j)     LSTM input  [s * ([x_hat1, x_hat2, u] - mu), h_hat, 1]
    c_hat(t_j)    LSTM cell memory (makes the LSTM a static map of (zeta, c_hat))
    u(t_j)        applied input
    x_meas(t_j)   measured generalized coordinates
    a(t_j)        acceleration target from the causal smoothing proxy
together with the regressor row r_j used for the rank condition.

Rank condition. With the concurrently learned parameters theta_h = vec(W_h) entering
Phi_hat = W_h^T h linearly, Phi'_j = d Phi_hat / d theta_h = I_n kron h_j^T and

    Omega = sum_j Phi'_j^T Phi'_j = I_n kron (sum_j h_j h_j^T) = I_n kron (R^T R),

where R = [r_1, ..., r_N]^T (N x p) is the data matrix with r_j = h_j. Hence

    lambda_min(Omega) = lambda_min(R^T R) = sigma_min(R)^2,

and the rank condition lambda_min(Omega) >= lambda_bar > 0 is a condition on the smallest
singular value of the data matrix. It needs N >= p = dim(r_j) recorded points.

Recording policy (per candidate r):
    1. Novelty gate: ||r - r_last|| / ||r|| >= novelty_tol, with r_last the most recently
       recorded regressor. Rejects near-duplicates from a slowly moving trajectory.
    2. Stack not full: append.
    3. Stack full: evaluate every single swap j -> r in one batched eigen-decomposition
           Omega_j' = Omega - r_j r_j^T + r r^T,
       and replace the j* that maximizes lambda_min(Omega_j'), provided it strictly
       increases lambda_min(Omega). While the stack is still rank deficient (every swap
       leaves lambda_min = 0), the D-optimal criterion log det(Omega + delta I) ranks the
       swaps instead, which drives the stack towards full rank.
For fixed regressors lambda_min(Omega) is therefore non-decreasing over time: a swap is
accepted only if it does not decrease lambda_min (strictly increases it once the stack
has full rank). This monotonicity is what the switched-system Lyapunov argument in the
README relies on.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Tuple
import numpy as np


@dataclass
class HistoryStackEntry:
    """One recorded operating point of the concurrent-learning history stack."""
    zeta: np.ndarray          # (d,)  LSTM input at t_j
    c_hat: np.ndarray         # (L,)  LSTM cell memory at t_j
    u: np.ndarray             # (m,)  applied input at t_j
    x_meas: np.ndarray        # (n,)  measured generalized coordinates at t_j
    accel_target: np.ndarray  # (n,)  acceleration proxy x_ddot(t_j)
    t: float                  # recording time t_j [s]
    regressor: np.ndarray     # (p,)  row r_j of the data matrix


class StackDecision(Enum):
    APPENDED = "appended"
    REPLACED = "replaced"
    REJECTED_NOVELTY = "rejected_novelty"
    REJECTED_NO_GAIN = "rejected_no_gain"


@dataclass
class StackStatistics:
    considered: int = 0
    appended: int = 0
    replaced: int = 0
    rejected_novelty: int = 0
    rejected_no_gain: int = 0


class HistoryStack:
    """Fixed-capacity history stack maximizing sigma_min of the data matrix."""

    def __init__(
        self,
        capacity: int,
        regressor_dim: int,
        novelty_tol: float = 0.02,
        min_rel_improvement: float = 1e-3,
        rank_tol: float = 1e-10,
        logdet_delta: float = 1e-6,
    ) -> None:
        if regressor_dim <= 0:
            raise ValueError("regressor_dim must be positive.")
        if capacity < regressor_dim:
            raise ValueError(
                f"capacity ({capacity}) < regressor_dim ({regressor_dim}): the rank condition "
                "lambda_min(Omega) > 0 needs at least as many points as regressor dimensions."
            )
        if novelty_tol < 0.0 or min_rel_improvement < 0.0:
            raise ValueError("novelty_tol and min_rel_improvement must be non-negative.")
        self.capacity = capacity
        self.regressor_dim = regressor_dim
        self.novelty_tol = novelty_tol
        self.min_rel_improvement = min_rel_improvement
        self.rank_tol = rank_tol
        self.logdet_delta = logdet_delta

        self.entries: List[HistoryStackEntry] = []
        self._R = np.zeros((0, regressor_dim), dtype=np.float64)
        self._omega = np.zeros((regressor_dim, regressor_dim), dtype=np.float64)
        self._last_regressor: np.ndarray | None = None
        self.stats = StackStatistics()
        self.version = 0  # bumped on every content change (lets callers cache Omega-derived terms)

    # ------------------------------------------------------------------ inspection
    def __len__(self) -> int:
        return len(self.entries)

    @property
    def is_full(self) -> bool:
        return len(self.entries) >= self.capacity

    def data_matrix(self) -> np.ndarray:
        """R = [r_1, ..., r_N]^T, shape (N, p)."""
        return self._R.copy()

    def information_matrix(self) -> np.ndarray:
        """Omega_r = R^T R = sum_j r_j r_j^T (p x p). The full Omega is I_n kron Omega_r."""
        return self._omega.copy()

    def targets(self) -> np.ndarray:
        """Acceleration targets A = [a_1, ..., a_N]^T, shape (N, n)."""
        if not self.entries:
            return np.zeros((0, 0), dtype=np.float64)
        return np.vstack([e.accel_target for e in self.entries])

    def lambda_min(self) -> float:
        """lambda_min(Omega) = sigma_min(R)^2 (zero while N < p)."""
        if len(self.entries) < self.regressor_dim:
            return 0.0
        return max(float(np.linalg.eigvalsh(self._omega)[0]), 0.0)

    def lambda_max(self) -> float:
        if not self.entries:
            return 0.0
        return float(np.linalg.eigvalsh(self._omega)[-1])

    def min_singular_value(self) -> float:
        return float(np.sqrt(self.lambda_min()))

    def rank(self) -> int:
        if not self.entries:
            return 0
        eig = np.linalg.eigvalsh(self._omega)
        return int(np.sum(eig > self.rank_tol * max(float(eig[-1]), 1.0)))

    def rank_condition_satisfied(self, lambda_bar: float) -> bool:
        return self.lambda_min() >= lambda_bar > 0.0

    # ------------------------------------------------------------------ recording policy
    def _novel(self, r: np.ndarray) -> bool:
        if self._last_regressor is None:
            return True
        return float(np.linalg.norm(r - self._last_regressor)) >= self.novelty_tol * (float(np.linalg.norm(r)) + 1e-12)

    def _swap_scores(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(lambda_min, logdet) of Omega - r_j r_j^T + r r^T for every j, in one batch."""
        base = self._omega + np.outer(r, r)
        swapped = base[None, :, :] - self._R[:, :, None] * self._R[:, None, :]
        eig = np.linalg.eigvalsh(swapped)                      # (N, p) ascending
        lam_min = np.maximum(eig[:, 0], 0.0)
        logdet = np.sum(np.log(np.maximum(eig, 0.0) + self.logdet_delta), axis=1)
        return lam_min, logdet

    def consider(self, entry: HistoryStackEntry) -> StackDecision:
        """Applies the recording policy to a candidate entry."""
        r = np.asarray(entry.regressor, dtype=np.float64)
        if r.shape != (self.regressor_dim,):
            raise ValueError(f"regressor must have shape ({self.regressor_dim},).")
        self.stats.considered += 1

        if not self._novel(r):
            self.stats.rejected_novelty += 1
            return StackDecision.REJECTED_NOVELTY

        if not self.is_full:
            self.entries.append(entry)
            self._R = np.vstack((self._R, r[None, :]))
            self._omega += np.outer(r, r)
            self._last_regressor = r.copy()
            self.stats.appended += 1
            self.version += 1
            return StackDecision.APPENDED

        eig_cur = np.linalg.eigvalsh(self._omega)
        lam_cur = max(float(eig_cur[0]), 0.0)
        lam_new, logdet_new = self._swap_scores(r)

        full_rank = lam_cur > self.rank_tol * max(float(eig_cur[-1]), 1.0)
        if full_rank:
            j = int(np.argmax(lam_new))
            accept = lam_new[j] > lam_cur * (1.0 + self.min_rel_improvement)
        else:
            # Rank-deficient phase: D-optimal tie-breaking among swaps that keep lambda_min.
            logdet_cur = float(np.sum(np.log(np.maximum(eig_cur, 0.0) + self.logdet_delta)))
            admissible = lam_new >= lam_cur
            scores = np.where(admissible, logdet_new, -np.inf)
            j = int(np.argmax(scores))
            accept = bool(admissible[j]) and (
                lam_new[j] > lam_cur * (1.0 + self.min_rel_improvement) + self.rank_tol
                or logdet_new[j] > logdet_cur + 1e-9
            )

        if not accept:
            self.stats.rejected_no_gain += 1
            return StackDecision.REJECTED_NO_GAIN

        self.entries[j] = entry
        self._R[j] = r
        # Rebuild instead of rank-2 downdate: avoids round-off accumulation over many swaps.
        self._omega = self._R.T @ self._R
        self._last_regressor = r.copy()
        self.stats.replaced += 1
        self.version += 1
        return StackDecision.REPLACED

    # ------------------------------------------------------------------ feature refresh
    def refresh_regressors(self, feature_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
        """
        Re-evaluates r_j = feature_fn(Z, C) for all entries (Z: (N, d) zetas, C: (N, L) cell
        memories). Needed when the feature map itself adapts (adapted gate weights).
        """
        if not self.entries:
            return
        Z = np.vstack([e.zeta for e in self.entries])
        C = np.vstack([e.c_hat for e in self.entries])
        R = np.asarray(feature_fn(Z, C), dtype=np.float64)
        if R.shape != self._R.shape:
            raise ValueError("feature_fn returned regressors of the wrong shape.")
        for e, r in zip(self.entries, R):
            e.regressor = r.copy()
        self._R = R
        self._omega = R.T @ R
        self.version += 1

    def clear(self) -> None:
        self.entries.clear()
        self._R = np.zeros((0, self.regressor_dim), dtype=np.float64)
        self._omega = np.zeros((self.regressor_dim, self.regressor_dim), dtype=np.float64)
        self._last_regressor = None
        self.stats = StackStatistics()
        self.version += 1
