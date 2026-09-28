"""
Unit tests for Approach A: Black-Box Continuous-Time Lb-LSTM Observer.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.plant.pendulum_plant import PendulumPlant, PlantState
from src.observers.blackbox_lstm_observer import BlackBoxLSTMObserver, BlackBoxLSTMConfig


def test_blackbox_lstm_initialization_and_reset() -> None:
    obs = BlackBoxLSTMObserver()
    init_state = np.array([0.1, -0.2, 0.05, 0.1])
    obs.reset(init_state)

    assert np.allclose(obs.x_hat, init_state)
    assert np.all(obs.c == 0.0)
    assert np.all(obs.W_a == 0.0)


def test_blackbox_lstm_stability_and_finite_outputs() -> None:
    plant = PendulumPlant()
    obs = BlackBoxLSTMObserver()
    obs.reset(np.zeros(4))

    dt = 0.001
    true_state = np.array([0.05, 0.1, 0.02, -0.05])

    for step in range(1000):
        force = float(2.0 * np.sin(0.02 * step))
        plant.state = PlantState.from_array(true_state)
        plant.step(force, dt)
        true_state = plant.state.to_array()

        meas = np.array([true_state[0] + 0.0001, true_state[2] - 0.0002])
        x_hat = obs.update(meas, force, dt)

    assert np.all(np.isfinite(x_hat))
    assert np.all(np.isfinite(obs.c))
    assert np.all(np.isfinite(obs.W_a))
    assert np.linalg.norm(obs.W_a) < 100.0  # Leakage guarantees boundedness
