"""
Unit tests for the concurrent-learning history stack (singular-value-maximizing recording).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.concurrent_learning.point_history_stack import HistoryStack, HistoryStackEntry, StackDecision


def _entry(r: np.ndarray, t: float = 0.0, a: np.ndarray | None = None) -> HistoryStackEntry:
    return HistoryStackEntry(
        zeta=np.zeros(3), c_hat=np.zeros(2), u=np.zeros(1), x_meas=np.zeros(2),
        accel_target=np.zeros(2) if a is None else a, t=t, regressor=r,
    )


def test_capacity_below_regressor_dim_is_rejected() -> None:
    with pytest.raises(ValueError):
        HistoryStack(capacity=3, regressor_dim=4)


def test_information_matrix_and_lambda_min_match_data_matrix() -> None:
    rng = np.random.default_rng(0)
    stack = HistoryStack(capacity=8, regressor_dim=4, novelty_tol=0.0)
    for k in range(8):
        assert stack.consider(_entry(rng.normal(size=4), k)) is StackDecision.APPENDED
    R = stack.data_matrix()
    np.testing.assert_allclose(stack.information_matrix(), R.T @ R, atol=1e-12)
    sigma_min = np.linalg.svd(R, compute_uv=False)[-1]
    assert stack.lambda_min() == pytest.approx(sigma_min ** 2, rel=1e-10)
    assert stack.min_singular_value() == pytest.approx(sigma_min, rel=1e-10)
    assert stack.rank() == 4 and stack.rank_condition_satisfied(0.5 * stack.lambda_min())


def test_lambda_min_is_zero_until_enough_points() -> None:
    stack = HistoryStack(capacity=5, regressor_dim=3, novelty_tol=0.0)
    stack.consider(_entry(np.array([1.0, 0.0, 0.0])))
    stack.consider(_entry(np.array([0.0, 1.0, 0.0])))
    assert stack.lambda_min() == 0.0
    assert not stack.rank_condition_satisfied(1e-6)


def test_novelty_gate_rejects_near_duplicates() -> None:
    stack = HistoryStack(capacity=4, regressor_dim=2, novelty_tol=0.1)
    assert stack.consider(_entry(np.array([1.0, 0.0]))) is StackDecision.APPENDED
    assert stack.consider(_entry(np.array([1.0, 0.01]))) is StackDecision.REJECTED_NOVELTY
    assert stack.consider(_entry(np.array([1.0, 0.5]))) is StackDecision.APPENDED
    assert stack.stats.rejected_novelty == 1 and len(stack) == 2


def test_lambda_min_never_decreases_once_full() -> None:
    """For fixed regressors every accepted swap increases lambda_min (monotone recording)."""
    rng = np.random.default_rng(1)
    stack = HistoryStack(capacity=6, regressor_dim=4, novelty_tol=0.0)
    history = []
    for k in range(400):
        stack.consider(_entry(rng.normal(size=4) * rng.uniform(0.1, 2.0), k))
        if stack.is_full:
            history.append(stack.lambda_min())
    assert np.all(np.diff(history) >= -1e-12)
    assert stack.stats.replaced > 0 and stack.stats.rejected_no_gain > 0


def test_swap_picks_the_best_replacement() -> None:
    stack = HistoryStack(capacity=2, regressor_dim=2, novelty_tol=0.0, min_rel_improvement=0.0)
    stack.consider(_entry(np.array([1.0, 0.0])))
    stack.consider(_entry(np.array([1.0, 0.1])))       # nearly collinear with the first point
    lam_before = stack.lambda_min()
    assert stack.consider(_entry(np.array([0.0, 1.0]))) is StackDecision.REPLACED
    assert stack.lambda_min() > lam_before
    # The brute-force optimum over both single swaps is lambda_min = 1 (orthonormal pair).
    assert stack.lambda_min() == pytest.approx(1.0, abs=1e-2)


def test_rank_deficient_stack_recovers_full_rank() -> None:
    """Filled with rank-1 data, the D-optimal tie-breaker still admits informative points."""
    stack = HistoryStack(capacity=3, regressor_dim=3, novelty_tol=0.0)
    for s in (1.0, 2.0, 3.0):
        stack.consider(_entry(np.array([s, 0.0, 0.0])))
    assert stack.rank() == 1 and stack.lambda_min() == 0.0
    stack.consider(_entry(np.array([0.0, 1.0, 0.0])))
    stack.consider(_entry(np.array([0.0, 0.0, 1.0])))
    assert stack.rank() == 3 and stack.lambda_min() > 0.0


def test_svm_selection_beats_fifo_on_decaying_excitation() -> None:
    """Rich early data then a collapsed regressor: the stack keeps the early information."""
    rng = np.random.default_rng(2)
    p, N = 4, 8
    stream = [rng.normal(size=p) for _ in range(40)]                          # informative transient
    stream += [np.array([1.0, 0.5, 0.0, 0.0]) * rng.uniform(0.5, 1.5) + 1e-3 * rng.normal(size=p)
               for _ in range(400)]                                           # non-PE tail
    stack = HistoryStack(capacity=N, regressor_dim=p, novelty_tol=0.0)
    for k, r in enumerate(stream):
        stack.consider(_entry(r, k))
    fifo = np.vstack(stream[-N:])
    lam_fifo = np.linalg.eigvalsh(fifo.T @ fifo)[0]
    assert stack.lambda_min() > 1e3 * max(lam_fifo, 1e-12)


def test_refresh_regressors_rebuilds_information_matrix() -> None:
    rng = np.random.default_rng(3)
    stack = HistoryStack(capacity=5, regressor_dim=2, novelty_tol=0.0)
    for k in range(5):
        e = _entry(rng.normal(size=2), k)
        e.zeta = rng.normal(size=3)
        stack.consider(e)
    version = stack.version
    stack.refresh_regressors(lambda Z, C: Z[:, :2] * 2.0)
    R = np.vstack([e.zeta[:2] * 2.0 for e in stack.entries])
    np.testing.assert_allclose(stack.data_matrix(), R)
    np.testing.assert_allclose(stack.information_matrix(), R.T @ R)
    assert stack.version > version


def test_targets_and_clear() -> None:
    stack = HistoryStack(capacity=2, regressor_dim=1, novelty_tol=0.0)
    stack.consider(_entry(np.array([1.0]), a=np.array([0.1, 0.2])))
    stack.consider(_entry(np.array([2.0]), a=np.array([0.3, 0.4])))
    np.testing.assert_allclose(stack.targets(), [[0.1, 0.2], [0.3, 0.4]])
    stack.clear()
    assert len(stack) == 0 and stack.lambda_min() == 0.0 and stack.stats.considered == 0
