"""
Unit tests for Approach E: the hybrid of Approach B (PI-LSTM) and Approach D (PI-ICL).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.observers.el_linear_model import LinearELModel
from src.observers.hybrid_observer import HybridObserver, HybridObserverConfig, damped_step
from src.observers.pi_icl_observer import PIICLObserver, PIICLObserverConfig
from src.plant.pendulum_plant import PendulumParameters, PendulumPlant
from src.plant.sensor_noise import SensorNoiseConfig, SensorNoiseModel

DT = 1e-3


def _hanging_run(obs, t_final: float, noise_deg: float = 0.0, seed: int = 0):
    """Hanging pendulum, two-tone input (Approach A's scenario). Returns (velocity error, plant parameters)."""
    p = PendulumParameters()
    plant = PendulumPlant(p)
    plant.reset(np.array([0.0, -2.0 / (1.5 * (p.M + p.m)), np.pi - 0.4, 0.0]))
    sensor = SensorNoiseModel(SensorNoiseConfig(theta_noise_deg=noise_deg, x_noise_mm=0.0 if noise_deg == 0.0 else 0.2))
    sensor.seed(42 + seed)
    obs.reset(np.array([0.0, 0.0, np.pi - 0.4, 0.0]))
    N = int(t_final / DT)
    err = np.zeros((N, 2))
    for k in range(N):
        s = plant.state.to_array()
        u = 2.0 * np.sin(1.5 * k * DT) + 1.2 * np.cos(3.0 * k * DT)
        est = obs.update(np.array(sensor.measure(s[0], s[2])), u, DT)
        err[k] = est[[1, 3]] - s[[1, 3]]
        plant.step(u, DT)
    return err, p


# ---------------------------------------------------------------------- blend algebra
def test_damped_step_weights_are_complementary_and_recover_newton() -> None:
    rng = np.random.default_rng(0)
    R = rng.normal(size=(40, 5))
    omega, phi_true, phi = R.T @ R, rng.normal(size=5), rng.normal(size=5)
    cross = omega @ phi_true
    lam = 0.3
    delta, W_w = damped_step(omega, cross, phi, lam)
    W_s = np.linalg.solve(omega + lam * np.eye(5), omega)
    np.testing.assert_allclose(W_s + W_w, np.eye(5), atol=1e-12)
    np.testing.assert_allclose(delta, W_s @ (phi_true - phi), atol=1e-10)      # = W_s (phi_LS - phi)
    np.testing.assert_allclose(damped_step(omega, cross, phi, 1e-12)[0], phi_true - phi, atol=1e-6)   # lam -> 0: Newton


def test_damped_step_leaves_unexcited_directions_to_the_instantaneous_law() -> None:
    rng = np.random.default_rng(1)
    R = rng.normal(size=(30, 5))
    R[:, 4] = R[:, 3]                                    # columns 3 and 4 collinear: one direction carries no information
    omega = R.T @ R
    null = np.array([0.0, 0.0, 0.0, 1.0, -1.0]) / np.sqrt(2.0)
    delta, W_w = damped_step(omega, omega @ rng.normal(size=5), rng.normal(size=5), 1e-3)
    assert abs(null @ delta) < 1e-8                     # integral CL does not move the unexcited direction
    np.testing.assert_allclose(W_w @ null, null, atol=1e-8)    # the instantaneous gradient passes through unchanged
    strong = np.linalg.eigh(omega)[1][:, -1]
    assert np.linalg.norm(W_w @ strong) < 1e-3          # and is removed along a well-excited direction


# ---------------------------------------------------------------------- structure
def test_own_coordinate_inertia_is_frozen() -> None:
    obs = HybridObserver(HybridObserverConfig())
    lay, P = obs.model.layout, obs.model.layout.n_features
    start = lay.block_slice("M").start
    e22 = obs.model.entries.index((1, 1))
    expected = np.zeros(lay.n_params, dtype=bool)
    expected[start + e22 * P + 1: start + e22 * P + P] = True       # cos / sin coefficients of M_22 only
    np.testing.assert_array_equal(obs.frozen, expected)
    for st_cols in (obs.cols_A, obs.unact[0].cols):
        assert not np.isin(np.flatnonzero(expected), st_cols).any()   # frozen parameters enter no stack
    # the true rig is still inside the model class
    assert np.all(obs.model.true_parameters(PendulumParameters())[expected] == 0.0)
    assert not HybridObserver(HybridObserverConfig(freeze_own_inertia=False)).frozen.any()


def test_frozen_parameters_stay_zero_and_residual_dissipates() -> None:
    obs = HybridObserver(HybridObserverConfig(seed=3, residual=True))
    rng = np.random.default_rng(3)
    obs.theta_res0[obs.res_layout.n_gate_params:] = rng.normal(size=obs.res_layout.n_out_params)   # non-zero readout
    obs.reset(np.array([0.0, 0.0, np.pi - 0.4, 0.0]))
    p = PendulumParameters()
    plant = PendulumPlant(p)
    plant.reset(np.array([0.0, 0.0, np.pi - 0.4, 0.0]))
    for k in range(3000):
        s = plant.state.to_array()
        u = 2.0 * np.sin(3.0 * k * DT)
        obs.update(s[[0, 2]], u, DT)
        plant.step(u, DT)
        assert obs.residual_power() >= 0.0
        assert np.all(obs.theta[obs.frozen] == 0.0)
    assert np.linalg.eigvalsh(obs.inertia_estimate())[0] >= obs.config.eps_M - 1e-9


def test_frozen_model_matches_the_plant_at_the_true_parameters() -> None:
    p = PendulumParameters()
    plant = PendulumPlant(p)
    obs = HybridObserver(HybridObserverConfig())
    obs.theta = obs.model.true_parameters(p)
    twin = obs.frozen_model()
    assert twin.n_mem == 0
    rng = np.random.default_rng(4)
    for _ in range(50):
        s = rng.normal(size=4) * np.array([0.3, 1.0, 3.0, 4.0])
        u = float(rng.normal() * 3.0)
        a, _ = twin.evaluate(s[[0, 2]], s[[1, 3]], np.array([u]), np.zeros(0))
        np.testing.assert_allclose(a, plant.forward_dynamics(s, u), atol=1e-12)
    # with the optional residual on, the frozen model carries the LSTM memory
    twin_r = HybridObserver(HybridObserverConfig(residual=True)).frozen_model()
    assert twin_r.n_mem == 2 * HybridObserverConfig().residual_hidden


def test_no_blend_no_freeze_reduces_to_approach_d() -> None:
    cfg = dict(gamma_inst=2.0)
    e = HybridObserver(HybridObserverConfig(blend=False, freeze_own_inertia=False, **cfg))
    d = PIICLObserver(PIICLObserverConfig(**cfg))
    err_e, _ = _hanging_run(e, 4.0, noise_deg=0.5)
    err_d, _ = _hanging_run(d, 4.0, noise_deg=0.5)
    np.testing.assert_allclose(err_e, err_d, atol=1e-10)
    np.testing.assert_allclose(e.theta, d.theta, atol=1e-10)


# ---------------------------------------------------------------------- closed loop
def test_identifies_the_pendulum_from_small_swings_where_approach_d_does_not() -> None:
    """Noisy hanging run: the hybrid recovers the pendulum inertia and gravity; Approach D's scale runs away."""
    truth = LinearELModel.physical_truth(PendulumParameters())
    hybrid, d = HybridObserver(HybridObserverConfig()), PIICLObserver(PIICLObserverConfig())
    err_h, _ = _hanging_run(hybrid, 20.0, noise_deg=0.5)
    _hanging_run(d, 20.0, noise_deg=0.5)
    est_h, est_d = hybrid.physical_parameters(), d.physical_parameters()
    assert est_h["I+ml^2"] == pytest.approx(truth["I+ml^2"], rel=0.15)
    assert est_h["mgl"] == pytest.approx(truth["mgl"], rel=0.15)
    assert abs(est_d["I+ml^2"] - truth["I+ml^2"]) > 5.0 * abs(est_h["I+ml^2"] - truth["I+ml^2"])
    assert np.sqrt(np.mean(err_h[-5000:, 1] ** 2)) < 0.03
    w = hybrid.information_weights()
    assert set(w) == {"actuated", "row 1 (normalized)"} and all(np.all((v >= 0.0) & (v <= 1.0)) for v in w.values())
