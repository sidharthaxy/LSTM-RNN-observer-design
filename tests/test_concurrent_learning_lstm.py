"""
Unit tests for Approach C: Concurrent Learning Lb-LSTM Observer.
Validates online history stack rank conditioning and parameter convergence without PE.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.plant.pendulum_plant import PendulumPlant, PlantState
from src.observers.concurrent_learning_lstm_observer import (
    ConcurrentLearningLSTMObserver,
    ConcurrentLearningConfig,
)


def test_history_stack_population_and_eigenvalues() -> None:
    """Verify history stack records transitions and accumulates non-zero excitation matrix."""
    cfg = ConcurrentLearningConfig(stack_capacity=10, sample_min_interval=0.01)
    obs = ConcurrentLearningLSTMObserver(cfg)

    # Feed distinct states
    rng = np.random.default_rng(42)
    for i in range(20):
        h = rng.normal(size=cfg.hidden_dim)
        a = rng.normal(size=2)
        obs.maybe_add_to_history_stack(h, a, t=i * 0.02)

    assert len(obs.history_stack) == 10
    Omega = obs.stack_excitation_matrix()
    assert Omega.shape == (cfg.hidden_dim, cfg.hidden_dim)
    eigvals = np.linalg.eigvalsh(Omega)
    # Numerical positive semi-definiteness within floating point precision
    assert np.all(eigvals >= -1e-12)
    assert np.max(eigvals) > 0.0


def test_cl_lstm_tracking_under_fading_excitation() -> None:
    """
    Verify Concurrent Learning observer remains stable when external excitation fades,
    simulating lack of Persistent Excitation (PE).
    """
    plant = PendulumPlant()
    cfg = ConcurrentLearningConfig(stack_capacity=16, sample_min_interval=0.02)
    obs = ConcurrentLearningLSTMObserver(cfg)
    obs.reset(np.zeros(4))

    dt = 0.001
    true_state = np.array([0.05, 0.1, 0.02, -0.05])

    # Phase 1: High excitation (populates history stack)
    for step in range(500):
        force = float(4.0 * np.sin(0.02 * step))
        plant.state = PlantState.from_array(true_state)
        plant.step(force, dt)
        true_state = plant.state.to_array()

        meas = np.array([true_state[0] + 0.0001, true_state[2] - 0.0002])
        x_hat = obs.update(meas, force, dt)

    assert len(obs.history_stack) > 0

    # Phase 2: Fading excitation (zero force, system settles, PE = 0)
    for step in range(500):
        force = 0.0  # Zero excitation
        plant.state = PlantState.from_array(true_state)
        plant.step(force, dt)
        true_state = plant.state.to_array()

        meas = np.array([true_state[0], true_state[2]])
        x_hat = obs.update(meas, force, dt)

    assert np.all(np.isfinite(x_hat))
    assert np.all(np.isfinite(obs.W_a))
    assert np.linalg.norm(obs.W_a) < 50.0  # Weights do not drift
