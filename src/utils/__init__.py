"""
Scientific metrics and evaluation utilities for state and velocity estimation.
"""

from .metrics import (
    compute_rmse,
    compute_steady_state_error_norm,
    compute_transient_peak_overshoot,
    compute_thd,
    compute_chattering_index,
    evaluate_estimation_performance,
    EstimationReport,
)

__all__ = [
    "compute_rmse",
    "compute_steady_state_error_norm",
    "compute_transient_peak_overshoot",
    "compute_thd",
    "compute_chattering_index",
    "evaluate_estimation_performance",
    "EstimationReport",
]
