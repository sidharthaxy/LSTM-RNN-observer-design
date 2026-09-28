"""
Unit tests for Sensor Noise Model and Stiction / Dead-Zone.
"""

from __future__ import annotations

import numpy as np

from src.plant.sensor_noise import SensorNoiseModel, SensorNoiseConfig


def test_uniform_noise_bounds() -> None:
    """Verify uniform sensor noise strictly respects +/- 0.5 deg and +/- 0.2 mm."""
    config = SensorNoiseConfig(
        theta_noise_deg=0.5,
        x_noise_mm=0.2,
        enable_quantization=False,
        enable_stiction=False,
        seed=123,
    )
    noise_model = SensorNoiseModel(config)

    true_x = 0.1
    true_theta = 0.05
    max_theta_err = np.deg2rad(0.5)
    max_x_err = 0.0002

    for _ in range(1000):
        x_meas, theta_meas = noise_model.measure(true_x, true_theta)
        assert abs(x_meas - true_x) <= max_x_err + 1e-10
        assert abs(theta_meas - true_theta) <= max_theta_err + 1e-10


def test_quantization_resolution() -> None:
    """Verify optical encoder quantization discretizes measurements onto grid."""
    config = SensorNoiseConfig(
        theta_noise_deg=0.0,  # Zero continuous noise
        x_noise_mm=0.0,
        enable_quantization=True,
        cart_counts_per_meter=4096.0,
        pendulum_counts_per_rev=4096.0,
    )
    noise_model = SensorNoiseModel(config)

    # Test values not on the grid
    x_test = 0.12345678
    theta_test = 0.4567891
    x_meas, theta_meas = noise_model.measure(x_test, theta_test)

    x_step = 1.0 / 4096.0
    th_step = (2.0 * np.pi) / 4096.0

    # Must be integer multiple of step
    assert np.isclose(x_meas % x_step, 0.0) or np.isclose(x_meas % x_step, x_step)
    assert np.isclose(theta_meas % th_step, 0.0) or np.isclose(theta_meas % th_step, th_step)


def test_asymmetric_deadzone_and_stiction() -> None:
    """Verify asymmetric deadband and stiction thresholds."""
    config = SensorNoiseConfig(
        enable_stiction=True,
        v_deadzone_positive=0.15,
        v_deadzone_negative=-0.12,
        v_to_force_gain=2.0,
        f_coulomb=0.35,
        v_stiction_threshold=0.005,
    )
    noise_model = SensorNoiseModel(config)

    # 1. Zero velocity with small force: should remain stuck (0.0 N)
    small_cmd_force = 0.15 * 2.0 * 0.5  # Below deadband
    net_f = noise_model.apply_actuator_deadzone_and_stiction(small_cmd_force, current_cart_velocity=0.0)
    assert net_f == 0.0

    # 2. Velocity above threshold: dynamic friction active
    large_cmd_force = 10.0
    net_f_moving = noise_model.apply_actuator_deadzone_and_stiction(large_cmd_force, current_cart_velocity=0.1)
    assert net_f_moving > 0.0
    assert net_f_moving < large_cmd_force  # Drag reduced it
