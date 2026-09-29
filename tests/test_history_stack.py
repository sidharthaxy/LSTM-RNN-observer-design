"""
Unit tests for the block-regressor history stack (singular-value-maximizing recording).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.concurrent_learning.history_stack import HistoryStack, HistoryStackEntry, StackDecision


def _entry(Y: np.ndarray, b: np.ndarray | None = None, t: float = 0.0) -> HistoryStackEntry:
    Y = np.atleast_2d(Y)
    return HistoryStackEntry(Y, np.zeros(Y.shape[0]) if b is None else b, t)


def test_capacity_must_allow_full_rank() -> None:
    with pytest.raises(ValueError):
        HistoryStack(capacity=3, regressor_dim=7, rows_per_entry=2)
    HistoryStack(capacity=4, regressor_dim=7, rows_per_entry=2)


def test_block_information_matrix_and_least_squares() -> None:
    rng = np.random.default_rng(0)
    theta = rng.normal(size=5)
    stack = HistoryStack(capacity=6, regressor_dim=5, rows_per_entry=2, novelty_tol=0.0)
    for k in range(6):
        Y = rng.normal(size=(2, 5))
        assert stack.consider(_entry(Y, Y @ theta, k)) is StackDecision.APPENDED
    R = stack.data_matrix()
    assert R.shape == (12, 5)
    np.testing.assert_allclose(stack.information_matrix(), R.T @ R, atol=1e-12)
    assert stack.lambda_min() == pytest.approx(np.linalg.svd(R, compute_uv=False)[-1] ** 2, rel=1e-10)
    sol = stack.least_squares()
    assert sol is not None
    np.testing.assert_allclose(sol, theta, atol=1e-10)
    np.testing.assert_allclose(stack.cross_term(), R.T @ stack.targets())


def test_lambda_min_zero_until_enough_rows() -> None:
    stack = HistoryStack(capacity=4, regressor_dim=3, rows_per_entry=1, novelty_tol=0.0)
    stack.consider(_entry(np.array([1.0, 0.0, 0.0])))
    stack.consider(_entry(np.array([0.0, 1.0, 0.0])))
    assert stack.lambda_min() == 0.0 and stack.least_squares() is None


def test_novelty_gate() -> None:
    stack = HistoryStack(capacity=4, regressor_dim=2, novelty_tol=0.1)
    assert stack.consider(_entry(np.array([1.0, 0.0]))) is StackDecision.APPENDED
    assert stack.consider(_entry(np.array([1.0, 0.01]))) is StackDecision.REJECTED_NOVELTY
    assert stack.consider(_entry(np.array([1.0, 0.5]))) is StackDecision.APPENDED


def test_lambda_min_is_monotone_once_full() -> None:
    rng = np.random.default_rng(1)
    stack = HistoryStack(capacity=5, regressor_dim=4, rows_per_entry=2, novelty_tol=0.0)
    hist = []
    for k in range(300):
        stack.consider(_entry(rng.normal(size=(2, 4)) * rng.uniform(0.1, 2.0), t=k))
        if stack.is_full:
            hist.append(stack.lambda_min())
    assert np.all(np.diff(hist) >= -1e-12)
    assert stack.stats.replaced > 0 and stack.stats.rejected_no_gain > 0


def test_rank_deficient_stack_recovers() -> None:
    stack = HistoryStack(capacity=3, regressor_dim=3, novelty_tol=0.0)
    for s in (1.0, 2.0, 3.0):
        stack.consider(_entry(np.array([s, 0.0, 0.0])))
    assert stack.rank() == 1
    stack.consider(_entry(np.array([0.0, 1.0, 0.0])))
    stack.consider(_entry(np.array([0.0, 0.0, 1.0])))
    assert stack.rank() == 3 and stack.lambda_min() > 0.0


def test_selection_beats_fifo_on_collapsing_excitation() -> None:
    rng = np.random.default_rng(2)
    stream = [rng.normal(size=(1, 4)) for _ in range(40)]
    stream += [np.array([[1.0, 0.5, 0.0, 0.0]]) * rng.uniform(0.5, 1.5) + 1e-3 * rng.normal(size=(1, 4)) for _ in range(400)]
    stack = HistoryStack(capacity=8, regressor_dim=4, novelty_tol=0.0)
    for k, Y in enumerate(stream):
        stack.consider(_entry(Y, t=k))
    fifo = np.vstack(stream[-8:])
    assert stack.lambda_min() > 1e3 * max(np.linalg.eigvalsh(fifo.T @ fifo)[0], 1e-12)


def test_purge_keeps_statistics() -> None:
    stack = HistoryStack(capacity=2, regressor_dim=1, novelty_tol=0.0)
    stack.consider(_entry(np.array([1.0])))
    stack.clear(purge=True)
    assert len(stack) == 0 and stack.stats.purges == 1 and stack.stats.appended == 1
    stack.clear()
    assert stack.stats.purges == 0
