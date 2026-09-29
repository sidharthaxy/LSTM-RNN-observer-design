"""
Concurrent-learning history stack with singular-value-maximizing data selection
(Chowdhary & Johnson, ACC 2011), for block regressors.

Each entry j stores a regressor block Y_j in R^{r x p} and a target b_j in R^r of a linear
relation b_j = Y_j theta* + noise. For integral concurrent learning (Approach D) one entry is
one integration window, with r = n rows (one per generalized coordinate); a single-row
regressor (r = 1) recovers the classical stack. The information matrix is

    Omega = sum_j Y_j^T Y_j = R^T R,    R = [Y_1; ...; Y_N]   (rN x p),

so lambda_min(Omega) = sigma_min(R)^2, and the rank condition lambda_min(Omega) > 0 needs
rN >= p.

Recording policy, per candidate Y:
    1. Novelty gate: ||Y - Y_last||_F >= novelty_tol ||Y||_F (Y_last = last recorded block).
    2. Stack not full: append.
    3. Stack full: evaluate every single swap j -> Y in one batched eigen-decomposition,
           Omega_j' = Omega - Y_j^T Y_j + Y^T Y,
       and replace the maximizer of lambda_min(Omega_j') if it increases lambda_min(Omega) by
       the relative margin min_rel_improvement. While the stack is rank deficient, swaps that do
       not decrease lambda_min are ranked by log det(Omega + delta I) (D-optimal tie-break).
For fixed data lambda_min(Omega) is non-decreasing (monotone recording); `clear()` restarts it
(used by Approach D's change detector after a plant change).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List
import numpy as np


@dataclass
class HistoryStackEntry:
    """One recorded window / operating point."""
    regressor: np.ndarray                 # (r, p) block Y_j (a (p,) vector is treated as r = 1)
    target: np.ndarray                    # (r,)   b_j
    t: float                              # recording time [s]
    info: Dict[str, float] = field(default_factory=dict)  # optional diagnostics
    payload: Any = None                   # optional caller data (e.g. the full window)

    def __post_init__(self) -> None:
        self.regressor = np.atleast_2d(np.asarray(self.regressor, dtype=np.float64))
        self.target = np.atleast_1d(np.asarray(self.target, dtype=np.float64))
        if self.target.shape != (self.regressor.shape[0],):
            raise ValueError("target must have one entry per regressor row.")


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
    purges: int = 0


class HistoryStack:
    """Fixed-capacity history stack maximizing sigma_min of the stacked data matrix."""

    def __init__(
        self,
        capacity: int,
        regressor_dim: int,
        rows_per_entry: int = 1,
        novelty_tol: float = 0.02,
        min_rel_improvement: float = 1e-3,
        rank_tol: float = 1e-10,
        logdet_delta: float = 1e-6,
    ) -> None:
        if regressor_dim <= 0 or rows_per_entry <= 0:
            raise ValueError("regressor_dim and rows_per_entry must be positive.")
        if capacity * rows_per_entry < regressor_dim:
            raise ValueError(
                f"capacity x rows ({capacity * rows_per_entry}) < regressor_dim ({regressor_dim}): "
                "the rank condition lambda_min(Omega) > 0 would be unattainable."
            )
        if novelty_tol < 0.0 or min_rel_improvement < 0.0:
            raise ValueError("novelty_tol and min_rel_improvement must be non-negative.")
        self.capacity = capacity
        self.regressor_dim = regressor_dim
        self.rows = rows_per_entry
        self.novelty_tol = novelty_tol
        self.min_rel_improvement = min_rel_improvement
        self.rank_tol = rank_tol
        self.logdet_delta = logdet_delta
        self.entries: List[HistoryStackEntry] = []
        self._blocks = np.zeros((0, rows_per_entry, regressor_dim))
        self._omega = np.zeros((regressor_dim, regressor_dim))
        self._last: np.ndarray | None = None
        self.stats = StackStatistics()
        self.version = 0   # bumped on every content change (lets callers cache Omega-derived terms)

    # ------------------------------------------------------------------ inspection
    def __len__(self) -> int:
        return len(self.entries)

    @property
    def is_full(self) -> bool:
        return len(self.entries) >= self.capacity

    def data_matrix(self) -> np.ndarray:
        """R = [Y_1; ...; Y_N], shape (rN, p)."""
        return self._blocks.reshape(-1, self.regressor_dim).copy()

    def targets(self) -> np.ndarray:
        """b = [b_1; ...; b_N], shape (rN,)."""
        if not self.entries:
            return np.zeros(0)
        return np.concatenate([e.target for e in self.entries])

    def information_matrix(self) -> np.ndarray:
        return self._omega.copy()

    def cross_term(self) -> np.ndarray:
        """B = R^T b = sum_j Y_j^T b_j, shape (p,)."""
        if not self.entries:
            return np.zeros(self.regressor_dim)
        return self.data_matrix().T @ self.targets()

    def eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvalsh(self._omega)

    def lambda_min(self) -> float:
        if len(self.entries) * self.rows < self.regressor_dim:
            return 0.0
        return max(float(self.eigenvalues()[0]), 0.0)

    def lambda_max(self) -> float:
        return float(self.eigenvalues()[-1]) if self.entries else 0.0

    def rank(self) -> int:
        if not self.entries:
            return 0
        eig = self.eigenvalues()
        return int(np.sum(eig > self.rank_tol * max(float(eig[-1]), 1.0)))

    def rank_condition_satisfied(self, lambda_bar: float) -> bool:
        return self.lambda_min() >= lambda_bar > 0.0

    def least_squares(self, ridge: float = 0.0) -> np.ndarray | None:
        """theta_H = (Omega + ridge I)^{-1} R^T b, or None while (Omega + ridge I) is singular."""
        if not self.entries:
            return None
        A = self._omega + ridge * np.eye(self.regressor_dim)
        if np.linalg.eigvalsh(A)[0] <= self.rank_tol * max(float(np.linalg.eigvalsh(A)[-1]), 1.0):
            return None
        return np.linalg.solve(A, self.cross_term())

    # ------------------------------------------------------------------ recording policy
    def _novel(self, Y: np.ndarray) -> bool:
        if self._last is None:
            return True
        return float(np.linalg.norm(Y - self._last)) >= self.novelty_tol * (float(np.linalg.norm(Y)) + 1e-12)

    def consider(self, entry: HistoryStackEntry) -> StackDecision:
        Y = entry.regressor
        if Y.shape != (self.rows, self.regressor_dim):
            raise ValueError(f"regressor must have shape ({self.rows}, {self.regressor_dim}).")
        self.stats.considered += 1
        if not self._novel(Y):
            self.stats.rejected_novelty += 1
            return StackDecision.REJECTED_NOVELTY

        gram = Y.T @ Y
        if not self.is_full:
            self.entries.append(entry)
            self._blocks = np.concatenate((self._blocks, Y[None]), axis=0)
            self._omega = self._omega + gram
            self._last = Y.copy()
            self.stats.appended += 1
            self.version += 1
            return StackDecision.APPENDED

        eig_cur = self.eigenvalues()
        lam_cur = max(float(eig_cur[0]), 0.0)
        grams = np.einsum("jri,jrk->jik", self._blocks, self._blocks)
        eig = np.linalg.eigvalsh(self._omega[None] - grams + gram[None])
        lam_new = np.maximum(eig[:, 0], 0.0)
        full_rank = lam_cur > self.rank_tol * max(float(eig_cur[-1]), 1.0)
        if full_rank:
            j = int(np.argmax(lam_new))
            accept = lam_new[j] > lam_cur * (1.0 + self.min_rel_improvement)
        else:
            logdet_cur = float(np.sum(np.log(np.maximum(eig_cur, 0.0) + self.logdet_delta)))
            logdet_new = np.sum(np.log(np.maximum(eig, 0.0) + self.logdet_delta), axis=1)
            admissible = lam_new >= lam_cur
            j = int(np.argmax(np.where(admissible, logdet_new, -np.inf)))
            accept = bool(admissible[j]) and (
                lam_new[j] > lam_cur * (1.0 + self.min_rel_improvement) + self.rank_tol
                or logdet_new[j] > logdet_cur + 1e-9
            )
        if not accept:
            self.stats.rejected_no_gain += 1
            return StackDecision.REJECTED_NO_GAIN

        self.entries[j] = entry
        self._blocks[j] = Y
        R = self._blocks.reshape(-1, self.regressor_dim)
        self._omega = R.T @ R    # rebuild: no round-off accumulation over many swaps
        self._last = Y.copy()
        self.stats.replaced += 1
        self.version += 1
        return StackDecision.REPLACED

    def clear(self, purge: bool = False) -> None:
        """Empties the stack. `purge=True` counts it as a change-detection purge (stats kept)."""
        self.entries.clear()
        self._blocks = np.zeros((0, self.rows, self.regressor_dim))
        self._omega = np.zeros((self.regressor_dim, self.regressor_dim))
        self._last = None
        if purge:
            self.stats.purges += 1
        else:
            self.stats = StackStatistics()
        self.version += 1
