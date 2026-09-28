"""
Unit tests for estimation evaluation metrics.
"""

from __future__ import annotations

import numpy as np

from src.utils.metrics import (
    compute_rmse,
    compute_steady_state_error_norm,
    compute_transient_peak_overshoot,
    compute_thd,
    compute_chattering_index,
    evaluate_estimation_performance,
)


def test_compute_rmse() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_hat = np.array([1.1, 1.9, 3.1, 3.9])
    # Errors are all +/- 0.1, squared error is 0.01, mean 0.01, sqrt 0.1
    rmse = compute_rmse(y_true, y_hat)
    assert np.isclose(rmse, 0.1)


def test_steady_state_error_norm() -> None:
    y_true = np.linspace(0.0, 1.0, 100)
    y_hat = np.copy(y_true)
    # Inject offset only in the last 10 samples
    y_hat[-10:] += 0.05

    mean_norm, final_norm = compute_steady_state_error_norm(y_true, y_hat, window_ratio=0.2)
    assert np.isclose(final_norm, 0.05)
    assert mean_norm > 0.0


def test_transient_peak_overshoot() -> None:
    y_true = np.zeros(100)
    y_hat = np.zeros(100)
    y_hat[15] = 2.5  # Transient spike

    max_err, pct = compute_transient_peak_overshoot(y_true, y_hat)
    assert np.isclose(max_err, 2.5)


def test_thd_computation() -> None:
    # Pure sinusoid should have near-zero THD
    dt = 0.001
    t = np.arange(0.0, 2.0, dt)
    pure_sin = np.sin(2.0 * np.pi * 5.0 * t)
    thd_pure = compute_thd(pure_sin, dt)
    assert thd_pure < 0.05

    # Distorted signal (fundamental + large 3rd harmonic)
    distorted = pure_sin + 0.5 * np.sin(2.0 * np.pi * 15.0 * t)
    thd_distorted = compute_thd(distorted, dt)
    assert thd_distorted > 0.3


def test_chattering_index() -> None:
    v_true = np.zeros(100)
    # High frequency chattering
    v_noisy = np.array([0.1 if i % 2 == 0 else -0.1 for i in range(100)])

    chat = compute_chattering_index(v_noisy, v_true, dt=0.001)
    assert chat["total_variation"] is not None and chat["total_variation"] > 15.0
    assert chat["jerk_energy"] is not None and chat["jerk_energy"] > 1e4


def test_evaluate_estimation_performance() -> None:
    n = 200
    t = np.linspace(0.0, 2.0, n)
    state_true = np.zeros((n, 4))
    state_hat = np.zeros((n, 4))

    state_true[:, 0] = np.sin(t)
    state_hat[:, 0] = np.sin(t) + 0.01 * np.random.randn(n)

    report = evaluate_estimation_performance(t, state_true, state_hat)
    assert "x" in report.metrics_by_state
    assert report.summary_df.shape[0] == 4
