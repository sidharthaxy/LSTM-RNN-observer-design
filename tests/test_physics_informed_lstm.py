"""
Unit tests for Approach B: Physics-Informed PI-LSTM Observer.
Validates structural positive-definiteness, Coriolis skew-symmetry, and observer stability.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.plant.pendulum_plant import PendulumPlant, PendulumParameters, PlantState
from src.observers.physics_informed_lstm_observer import (
    PhysicsInformedLSTMObserver,
    PhysicsInformedLSTMConfig,
)


def test_coriolis_skew_symmetry_property() -> None:
    """
    Rigorously verify the mechanical passivity identity:
    z^T * [ dM/dt - 2*C(q, q_dot) ] * z == 0 for all z in R^2.
    """
    obs = PhysicsInformedLSTMObserver()
    thetas = np.linspace(-np.pi, np.pi, 20)
    theta_dots = np.linspace(-5.0, 5.0, 10)

    # Random test vectors in R^2
    rng = np.random.default_rng(101)
    test_vectors = [rng.normal(size=2) for _ in range(10)]

    for th in thetas:
        for th_dot in theta_dots:
            C = obs.coriolis_matrix(th, th_dot)

            # Analytical time derivative dM/dt:
            # d/dt [ml*cos(theta)] = -ml*sin(theta)*theta_dot
            p = obs.params
            dM_dt = np.array([
                [0.0, -p.m * p.l * th_dot * np.sin(th)],
                [-p.m * p.l * th_dot * np.sin(th), 0.0]
            ], dtype=np.float64)

            skew_matrix = dM_dt - 2.0 * C

            # Matrix must be strictly skew-symmetric: S^T = -S
            assert np.allclose(skew_matrix, -skew_matrix.T, atol=1e-12)

            # Quadratic form z^T * S * z must be zero
            for z in test_vectors:
                quad = float(z.T @ skew_matrix @ z)
                assert abs(quad) < 1e-12, f"Skew symmetry failed: quad={quad}"


def test_inertia_matrix_positive_definiteness() -> None:
    """Verify inertia matrix M(theta) is strictly symmetric positive definite."""
    obs = PhysicsInformedLSTMObserver()
    for th in np.linspace(-np.pi, np.pi, 50):
        M = obs.inertia_matrix(th)
        assert np.allclose(M, M.T)
        eigvals = np.linalg.eigvals(M)
        assert np.all(eigvals > 0.0)


def test_pi_lstm_tracking_and_finite_outputs() -> None:
    """Verify observer runs stably and outputs finite state estimates."""
    plant = PendulumPlant()
    obs = PhysicsInformedLSTMObserver()
    obs.reset(np.zeros(4))

    dt = 0.001
    true_state = np.array([0.05, 0.1, 0.02, -0.05])

    for step in range(1000):
        force = float(3.0 * np.sin(0.015 * step))
        plant.state = PlantState.from_array(true_state)
        plant.step(force, dt)
        true_state = plant.state.to_array()

        meas = np.array([true_state[0] + 0.0001, true_state[2] - 0.0002])
        x_hat = obs.update(meas, force, dt)

    assert np.all(np.isfinite(x_hat))
    assert np.all(np.isfinite(obs.c))
    assert np.all(np.isfinite(obs.W_delta))
    assert np.linalg.norm(obs.W_delta) < 100.0
