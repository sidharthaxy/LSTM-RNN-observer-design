"""
Unit tests for Feedback 33-936S Pendulum Plant dynamics and Jacobians.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.plant.pendulum_plant import PendulumPlant, PendulumParameters, PlantState


@pytest.fixture
def plant() -> PendulumPlant:
    return PendulumPlant()


def test_plant_parameters(plant: PendulumPlant) -> None:
    """Verify exact manufacturer specifications from Feedback 33-936S manual."""
    p = plant.params
    assert p.M == 2.4
    assert p.m == 0.23
    assert p.l == 0.36
    assert p.I == 0.099
    assert p.b == 0.05
    assert p.d == 0.005
    assert p.g == 9.81
    assert p.F_max == 20.0
    assert p.x_limit == 0.5


def test_mass_matrix_positive_definiteness(plant: PendulumPlant) -> None:
    """Verify inertia matrix M(theta) is strictly positive definite for all theta."""
    thetas = np.linspace(-2.0 * np.pi, 2.0 * np.pi, 100)
    for th in thetas:
        M = plant.mass_matrix(th)
        det_M = plant.mass_matrix_determinant(th)
        assert det_M > 0.3, f"Determinant too small at theta={th}: {det_M}"

        # Eigenvalues must be strictly positive
        eigvals = np.linalg.eigvals(M)
        assert np.all(eigvals > 0), f"Negative eigenvalue at theta={th}: {eigvals}"
        # Symmetry check
        assert np.allclose(M, M.T)


def test_analytical_jacobian_vs_finite_difference(plant: PendulumPlant) -> None:
    """
    Rigorously verify analytical state Jacobian A = df/dx and input Jacobian B = df/du
    against central finite-difference numerical approximations.
    """
    test_states = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.15, 0.4, 0.25, -0.6]),
        np.array([-0.2, -0.5, -0.3, 0.8]),
        np.array([0.05, 0.1, 0.05, 0.2]),
    ]
    test_forces = [0.0, 5.0, -10.0, 15.0]

    eps = 1e-6
    for state in test_states:
        for force in test_forces:
            A_ana, B_ana = plant.analytical_jacobian(state, force)

            # Central finite differences for A
            A_num = np.zeros((4, 4), dtype=np.float64)
            for i in range(4):
                dx = np.zeros(4)
                dx[i] = eps
                f_plus = plant.state_derivative(state + dx, force)
                f_minus = plant.state_derivative(state - dx, force)
                A_num[:, i] = (f_plus - f_minus) / (2.0 * eps)

            # Central finite differences for B
            f_plus_u = plant.state_derivative(state, force + eps)
            f_minus_u = plant.state_derivative(state, force - eps)
            B_num = ((f_plus_u - f_minus_u) / (2.0 * eps)).reshape(4, 1)

            assert np.allclose(A_ana, A_num, atol=1e-5, rtol=1e-4), (
                f"Jacobian A mismatch at state={state}, force={force}:\n"
                f"Analytical:\n{A_ana}\nNumerical:\n{A_num}\nDiff:\n{A_ana - A_num}"
            )
            assert np.allclose(B_ana, B_num, atol=1e-5, rtol=1e-4), (
                f"Jacobian B mismatch at state={state}, force={force}:\n"
                f"Analytical:\n{B_ana}\nNumerical:\n{B_num}\nDiff:\n{B_ana - B_num}"
            )


def test_energy_conservation_undamped() -> None:
    """Verify mechanical energy is conserved when damping and external forces are zero."""
    undamped_params = PendulumParameters(b=0.0, d=0.0)
    plant = PendulumPlant(undamped_params)
    init_state = PlantState(x=0.0, x_dot=0.0, theta=0.2, theta_dot=0.0)
    plant.reset(init_state)

    e_initial = plant.total_energy()
    dt = 0.0005
    for _ in range(2000):  # Simulate 1 second
        plant.step(force=0.0, dt=dt, integrator="rk4")

    e_final = plant.total_energy()
    rel_energy_drift = abs(e_final - e_initial) / abs(e_initial)
    assert rel_energy_drift < 1e-4, f"Energy conservation violated: drift={rel_energy_drift}"


def test_force_saturation(plant: PendulumPlant) -> None:
    """Ensure actuator forces are strictly saturated at +/-20 N."""
    state = np.array([0.0, 0.0, 0.1, 0.0])
    accel_pos_sat = plant.forward_dynamics(state, 50.0)
    accel_pos_limit = plant.forward_dynamics(state, 20.0)
    assert np.allclose(accel_pos_sat, accel_pos_limit)

    accel_neg_sat = plant.forward_dynamics(state, -100.0)
    accel_neg_limit = plant.forward_dynamics(state, -20.0)
    assert np.allclose(accel_neg_sat, accel_neg_limit)


def test_track_limits_and_boundary_clamping(plant: PendulumPlant) -> None:
    """Ensure cart cannot travel outside +/- 0.5m."""
    plant.reset(PlantState(x=0.49, x_dot=2.0, theta=0.0, theta_dot=0.0))
    # Move forward with high velocity
    for _ in range(50):
        plant.step(force=20.0, dt=0.01, integrator="rk4")
    assert plant.state.x <= 0.5 + 1e-6
