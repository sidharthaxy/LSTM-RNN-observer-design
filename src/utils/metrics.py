"""
Evaluation metrics for state estimation and dynamic model identification.

Implements:
1. Root Mean Square Error (RMSE)
2. Steady-state error norm ||x2 - x2_hat||
3. Transient peak overshoot (absolute peak error & relative percentage overshoot)
4. Total Harmonic Distortion (THD) via FFT spectrum
5. Chattering index / Total Variation on velocity estimates
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


@dataclass
class EstimationReport:
    """Structured report container for estimation benchmark metrics."""
    metrics_by_state: Dict[str, Dict[str, float]]
    summary_df: pd.DataFrame


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def compute_rmse(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    is_angle: bool = False,
) -> float:
    """
    Computes Root Mean Square Error (RMSE):
    RMSE = sqrt( (1 / N) * sum( (y_true - y_hat)^2 ) )
    """
    error = y_true - y_hat
    if is_angle:
        error = _wrap_angle(error)
    return float(np.sqrt(np.mean(error ** 2)))


def compute_steady_state_error_norm(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    window_ratio: float = 0.2,
    is_angle: bool = False,
) -> Tuple[float, float]:
    """
    Computes steady-state error norm on the final portion of trajectory:
    t in [t_end * (1 - window_ratio), t_end].
    Returns:
        (mean_norm, final_norm)
    """
    n_samples = len(y_true)
    start_idx = int(n_samples * (1.0 - window_ratio))
    start_idx = max(0, min(start_idx, n_samples - 1))

    error = y_true[start_idx:] - y_hat[start_idx:]
    if is_angle:
        error = _wrap_angle(error)

    if error.ndim == 1:
        mean_norm = float(np.mean(np.abs(error)))
        final_norm = float(np.abs(error[-1]))
    else:
        # Vector state: compute Euclidean 2-norm per time step
        norms = np.linalg.norm(error, axis=-1)
        mean_norm = float(np.mean(norms))
        final_norm = float(norms[-1])

    return mean_norm, final_norm


def compute_transient_peak_overshoot(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    is_angle: bool = False,
) -> Tuple[float, float]:
    """
    Computes:
    1. Maximum absolute estimation error: max_t |y_true(t) - y_hat(t)|
    2. Relative percentage peak overshoot relative to true range:
       (max_error / (max(y_true) - min(y_true) + 1e-6)) * 100%
    """
    error = y_true - y_hat
    if is_angle:
        error = _wrap_angle(error)

    if error.ndim == 1:
        max_abs_error = float(np.max(np.abs(error)))
        val_range = float(np.ptp(y_true))
    else:
        norms = np.linalg.norm(error, axis=-1)
        max_abs_error = float(np.max(norms))
        val_range = float(np.max(np.ptp(y_true, axis=0)))

    pct_overshoot = float((max_abs_error / (val_range + 1e-6)) * 100.0)
    return max_abs_error, pct_overshoot


def compute_thd(signal: np.ndarray, dt: float, max_harmonics: int = 10) -> float:
    """
    Computes standard Total Harmonic Distortion (THD) of a signal:
        THD = sqrt( sum_{k=2}^H V_k^2 ) / V_1
    Identifies dominant fundamental frequency (excluding DC) and measures
    spectral harmonic peaks at 2*f0, 3*f0, ..., up to Nyquist.
    """
    n = len(signal)
    if n < 16:
        return 0.0

    sig_detrend = signal - np.mean(signal)
    window = np.blackman(n)
    fft_vals = np.fft.rfft(sig_detrend * window)
    magnitudes = np.abs(fft_vals)

    if len(magnitudes) <= 4:
        return 0.0

    # Search for fundamental peak above DC (skip first 2 bins)
    fund_idx = int(np.argmax(magnitudes[2:])) + 2
    fund_mag = magnitudes[fund_idx]

    if fund_mag < 1e-10:
        return 0.0

    harmonic_sq_sum = 0.0
    n_bins = len(magnitudes)

    for h in range(2, max_harmonics + 1):
        target_idx = h * fund_idx
        if target_idx >= n_bins:
            break
        # Search within small neighborhood +/- 2 bins for peak
        low = max(0, target_idx - 2)
        high = min(n_bins, target_idx + 3)
        h_mag = float(np.max(magnitudes[low:high]))
        harmonic_sq_sum += h_mag ** 2

    thd = np.sqrt(harmonic_sq_sum) / fund_mag
    return float(thd)


def compute_chattering_index(
    v_hat: np.ndarray,
    v_true: np.ndarray | None = None,
    dt: float = 0.001,
) -> Dict[str, float]:
    """
    Computes chattering and high-frequency noise metrics on estimated velocity:
    - total_variation: sum |v_hat[k+1] - v_hat[k]|
    - normalized_chattering: TV(v_hat) / (TV(v_true) + 1e-8)
    - jerk_energy: mean( ((v_hat[k+1] - v_hat[k]) / dt)^2 )
    """
    diff_hat = np.diff(v_hat, axis=0)
    tv_hat = float(np.sum(np.abs(diff_hat)))
    jerk_power = float(np.mean((diff_hat / dt) ** 2))

    if v_true is not None:
        diff_true = np.diff(v_true, axis=0)
        tv_true = float(np.sum(np.abs(diff_true)))
        chattering_ratio = float(tv_hat / (tv_true + 1e-8))
    else:
        tv_true = None
        chattering_ratio = None

    return {
        "total_variation": tv_hat,
        "true_total_variation": tv_true,
        "chattering_ratio": chattering_ratio,
        "jerk_energy": jerk_power,
    }


def evaluate_estimation_performance(
    t: np.ndarray,
    state_true: np.ndarray,
    state_hat: np.ndarray,
    state_names: Tuple[str, ...] = ("x", "x_dot", "theta", "theta_dot"),
    dt: float | None = None,
) -> EstimationReport:
    """
    Comprehensive evaluation of state estimation performance across all state channels.
    """
    if dt is None:
        dt = float(np.mean(np.diff(t))) if len(t) > 1 else 0.001

    results: Dict[str, Dict[str, float]] = {}
    rows = []

    for idx, name in enumerate(state_names):
        y_t = state_true[:, idx]
        y_h = state_hat[:, idx]
        is_angle = ("theta" in name and "dot" not in name)

        rmse = compute_rmse(y_t, y_h, is_angle=is_angle)
        ss_mean, ss_final = compute_steady_state_error_norm(y_t, y_h, is_angle=is_angle)
        peak_err, pct_os = compute_transient_peak_overshoot(y_t, y_h, is_angle=is_angle)
        thd = compute_thd(y_h, dt)
        chat = compute_chattering_index(y_h, y_t, dt)

        channel_metrics = {
            "RMSE": rmse,
            "SS_Mean_Norm": ss_mean,
            "SS_Final_Norm": ss_final,
            "Peak_Error": peak_err,
            "Pct_Overshoot": pct_os,
            "THD": thd,
            "Total_Variation": chat["total_variation"],
            "Chattering_Ratio": chat["chattering_ratio"] or 1.0,
            "Jerk_Energy": chat["jerk_energy"],
        }
        results[name] = channel_metrics

        row = {"Channel": name, **channel_metrics}
        rows.append(row)

    summary_df = pd.DataFrame(rows).set_index("Channel")
    return EstimationReport(metrics_by_state=results, summary_df=summary_df)
