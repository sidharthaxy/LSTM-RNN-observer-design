"""
Unit tests for baseline state estimators: Dirty Derivative, Butterworth, EKF, Shallow RNN.
"""

from __future__ import annotations

import numpy as np

from src.plant.pendulum_plant import PendulumPlant, PlantState
from src.baselines.classical_estimators import (
    DirtyDerivativeFilter,
    ButterworthDifferentiator,
    ContinuousDiscreteEKF,
    ContinuousShallowRNNObserver,
    RNNConfig,
)


def test_dirty_derivative_sinusoid() -> None:
    """Verify Dirty Derivative accurately tracks analytical velocity of a clean sine wave."""
    filt = DirtyDerivativeFilter(tau_d=0.01)
    omega = 2.0 * np.pi * 1.0  # 1 Hz signal
    dt = 0.001
    t_arr = np.arange(0.0, 3.0, dt)

    u_arr = np.sin(omega * t_arr)
    du_true = omega * np.cos(omega * t_arr)

    du_est = np.zeros_like(u_arr)
    filt.reset(u_arr[0])
    for i in range(len(t_arr)):
        du_est[i] = filt.update(u_arr[i], dt)

    # After initial transient (t > 0.5s), estimate should track with small phase lag
    idx_eval = int(0.5 / dt)
    corr = np.corrcoef(du_true[idx_eval:], du_est[idx_eval:])[0, 1]
    assert corr > 0.98, f"Dirty derivative correlation too low: {corr}"


def test_butterworth_differentiator_sinusoid() -> None:
    """Verify 2nd-order Butterworth filter differentiator tracks derivative."""
    butter = ButterworthDifferentiator()
    omega = 2.0 * np.pi * 1.5  # 1.5 Hz signal
    dt = 0.001
    t_arr = np.arange(0.0, 3.0, dt)

    u_arr = np.sin(omega * t_arr)
    du_true = omega * np.cos(omega * t_arr)

    du_est = np.zeros_like(u_arr)
    butter.reset(u_arr[0])
    for i in range(len(t_arr)):
        du_est[i] = butter.update(u_arr[i], dt)

    idx_eval = int(0.5 / dt)
    corr = np.corrcoef(du_true[idx_eval:], du_est[idx_eval:])[0, 1]
    assert corr > 0.99, f"Butterworth differentiator correlation too low: {corr}"


def test_ekf_convergence() -> None:
    """Verify Continuous-Discrete EKF converges from an initial state estimation offset."""
    plant = PendulumPlant()
    ekf = ContinuousDiscreteEKF(plant)

    # True state
    true_state = np.array([0.1, 0.2, 0.05, -0.1], dtype=np.float64)
    # Observer initialized with offset
    ekf_init = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ekf.reset(ekf_init)

    dt = 0.001
    force = 0.0
    for _ in range(500):
        # Forward true state
        plant.state = PlantState.from_array(true_state)
        plant.step(force, dt)
        true_state = plant.state.to_array()

        # Measurement with small noise
        meas = np.array([true_state[0] + 0.0001, true_state[2] + 0.0002])
        x_hat = ekf.step(force, meas, dt)

    # Error after 500ms
    err = np.linalg.norm(true_state - x_hat)
    assert err < 0.05, f"EKF failed to converge within threshold: err={err}"


def test_shallow_dynamic_rnn_boundedness() -> None:
    """Verify Shallow Dynamic RNN weights remain strictly bounded under excitation."""
    plant = PendulumPlant()
    cfg = RNNConfig(hidden_dim=8, gamma1=2.0, gamma2=2.0, sigma_leak=0.05)
    rnn = ContinuousShallowRNNObserver(plant, cfg)

    dt = 0.001
    true_state = np.array([0.05, 0.1, 0.02, -0.05])
    rnn.reset(np.zeros(4))

    for step in range(1000):
        plant.state = PlantState.from_array(true_state)
        force = float(np.sin(0.01 * step))
        plant.step(force, dt)
        true_state = plant.state.to_array()

        meas = np.array([true_state[0], true_state[2]])
        x_hat = rnn.update(meas, force, dt)

    # Check finite numbers and bounded weights
    assert np.all(np.isfinite(x_hat))
    assert np.all(np.isfinite(rnn.W1))
    assert np.all(np.isfinite(rnn.W2))
    assert np.linalg.norm(rnn.W1) < 50.0
    assert np.linalg.norm(rnn.W2) < 50.0
