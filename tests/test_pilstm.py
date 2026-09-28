"""
Unit tests for Approach B: physics-informed PI-LSTM (structured Euler-Lagrange sub-networks,
Jacobian engine, observer). Every structural invariant is checked for *random* parameters,
since it must hold along the whole adaptation transient, not only at convergence.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.adaptation.pilstm_jacobian_engine import (
    PILSTMAdaptationLaw,
    pilstm_jacobian,
    pilstm_jacobian_transpose_vec,
)
from src.observers.pilstm_network import ConfigurationFeatures, PILSTMNetwork
from src.observers.pilstm_observer import PILSTMObserver, PILSTMObserverConfig
from src.plant.pendulum_plant import PendulumPlant


def _net(**kw) -> PILSTMNetwork:
    defaults = dict(n_features=5, harmonics=3, friction_hidden=3, inertia_scale=(2.5, 0.15), seed=3)
    defaults.update(kw)
    return PILSTMNetwork(**defaults)  # type: ignore[arg-type]


def _random_theta(net: PILSTMNetwork, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    th = net.initial_parameters((1.0, 0.2), seed=seed)
    return th + scale * rng.normal(size=th.size)


def _random_point(rng: np.random.Generator, L: int = 3):
    q = rng.uniform([-0.4, -np.pi], [0.4, np.pi])
    qd = rng.normal(0.0, 2.0, 2)
    return q, qd, rng.normal(size=1), 0.3 * rng.normal(size=L), 0.3 * rng.normal(size=L)


# ---------------------------------------------------------------------------- features
@pytest.mark.parametrize("cyclic", [(False, False), (True, False)])
def test_feature_jacobian_matches_finite_differences(cyclic) -> None:
    feats = ConfigurationFeatures((False, True), 6, position_scale=(3.0, 1.0), harmonics=3,
                                  active=~np.asarray(cyclic), seed=1)
    q = np.array([0.13, 0.9])
    _, drho = feats(q)
    h = 1e-6
    for k in range(2):
        dq = np.zeros(2)
        dq[k] = h
        fd = (feats(q + dq)[0] - feats(q - dq)[0]) / (2 * h)
        assert np.allclose(drho[:, k], fd, atol=1e-8)
    if cyclic[0]:
        assert np.all(drho[:, 0] == 0.0)


def test_initial_parameters_give_configuration_independent_prior() -> None:
    net = _net()
    th = net.initial_parameters((2.0, 0.2))
    for q_theta in np.linspace(-np.pi, np.pi, 7):
        M = net.inertia(th, np.array([0.1, q_theta])).M
        assert np.allclose(M, np.diag([2.0, 0.2]), atol=1e-12)


# ---------------------------------------------------------------------------- inertia
def test_inertia_symmetric_positive_definite_for_random_parameters() -> None:
    net = _net(eps_M=0.05)
    rng = np.random.default_rng(0)
    for trial in range(30):
        th = _random_theta(net, 3.0, trial)       # far from any physical inertia
        q, qd, *_ = _random_point(rng)
        M = net.inertia(th, q).M
        assert np.allclose(M, M.T, atol=1e-14)
        assert np.linalg.eigvalsh(M)[0] >= 0.05 - 1e-12
        assert net.kinetic_energy(th, q, qd) >= 0.0


def test_inertia_configuration_derivative_matches_finite_differences() -> None:
    net = _net()
    th = _random_theta(net, 0.5, 1)
    q = np.array([0.05, 1.1])
    dM = net.inertia(th, q).dM
    h = 1e-6
    for k in range(2):
        dq = np.zeros(2)
        dq[k] = h
        fd = (net.inertia(th, q + dq).M - net.inertia(th, q - dq).M) / (2 * h)
        assert np.allclose(dM[k], fd, atol=1e-7)


def test_cyclic_coordinate_is_absent_from_the_lagrangian() -> None:
    net = _net(cyclic=(True, False))
    th = _random_theta(net, 0.5, 2)
    q1, q2 = np.array([-0.3, 0.7]), np.array([0.4, 0.7])
    assert np.allclose(net.inertia(th, q1).M, net.inertia(th, q2).M)
    assert net.potential_energy(th, q1) == pytest.approx(net.potential_energy(th, q2))
    c = net.forward(th, q1, np.array([0.3, -1.0]), np.zeros(1), np.zeros(3), np.zeros(3))
    assert c.gravity[0] == 0.0


# ---------------------------------------------------------------------------- Coriolis / skew symmetry
@pytest.mark.parametrize("gyroscopic", [False, True])
def test_skew_symmetry_holds_for_random_parameters(gyroscopic) -> None:
    net = _net(gyroscopic=gyroscopic)
    rng = np.random.default_rng(5)
    for trial in range(20):
        th = _random_theta(net, 1.0, trial)
        q, qd, *_ = _random_point(rng)
        N = net.skew_residual(th, q, qd)
        assert np.allclose(N, -N.T, atol=1e-12)
        for _ in range(5):
            z = rng.normal(size=2)
            assert abs(z @ N @ z) < 1e-12


def test_christoffel_vector_equals_matrix_times_velocity() -> None:
    net = _net()
    rng = np.random.default_rng(7)
    th = _random_theta(net, 1.0, 4)
    q, qd, u, c_hat, h_hat = _random_point(rng)
    cache = net.forward(th, q, qd, u, c_hat, h_hat)
    assert np.allclose(net.coriolis_matrix(th, q, qd) @ qd, cache.coriolis + cache.gyro, atol=1e-13)


def test_christoffel_matches_brute_force_symbols() -> None:
    """C_ij = sum_k 1/2 (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i) q_dot_k with finite-difference dM/dq."""
    net = _net(gyroscopic=False, cyclic=(False, False), position_scale=(2.0, 1.0))
    th = _random_theta(net, 0.7, 6)
    q, qd = np.array([0.2, 0.8]), np.array([0.5, -1.7])
    h = 1e-6
    dM = np.zeros((2, 2, 2))
    for k in range(2):
        dq = np.zeros(2)
        dq[k] = h
        dM[k] = (net.inertia(th, q + dq).M - net.inertia(th, q - dq).M) / (2 * h)
    C = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            C[i, j] = sum(0.5 * (dM[k][i, j] + dM[j][i, k] - dM[i][j, k]) * qd[k] for k in range(2))
    assert np.allclose(net.coriolis_matrix(th, q, qd), C, atol=1e-7)


# ---------------------------------------------------------------------------- energy
def test_learned_rigid_body_conserves_its_energy() -> None:
    """Free response of the learned conservative model (u = 0, F = 0) conserves T_hat + P_hat."""
    net = _net(gyroscopic=True)
    th = _random_theta(net, 0.4, 11)
    u0 = np.zeros(1)

    def f(z: np.ndarray) -> np.ndarray:
        return np.concatenate((z[2:], net.conservative_acceleration(th, z[:2], z[2:], u0)))

    def H(z: np.ndarray) -> float:
        return net.kinetic_energy(th, z[:2], z[2:]) + net.potential_energy(th, z[:2])

    z = np.array([0.0, 2.0, 0.3, 1.5])
    H0, dt = H(z), 1e-3
    for _ in range(2000):
        k1 = f(z); k2 = f(z + 0.5 * dt * k1); k3 = f(z + 0.5 * dt * k2); k4 = f(z + dt * k3)
        z = z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    assert abs(H(z) - H0) < 1e-8 * max(1.0, abs(H0))


def test_friction_is_dissipative_for_random_parameters() -> None:
    net = _net()
    rng = np.random.default_rng(9)
    for trial in range(30):
        th = _random_theta(net, 3.0, trial)
        q, qd, u, c_hat, h_hat = _random_point(rng)
        cache = net.forward(th, q, qd, u, c_hat, h_hat)
        assert qd @ cache.friction_force >= 0.0


# ---------------------------------------------------------------------------- Jacobians
@pytest.mark.parametrize("dissipative", [True, False])
def test_block_jacobians_match_finite_differences(dissipative) -> None:
    net = _net(dissipative_friction=dissipative, cyclic=(True, False))
    rng = np.random.default_rng(0)
    th = _random_theta(net, 0.3, 0)
    q, qd, u, c_hat, h_hat = _random_point(rng)
    J = pilstm_jacobian(net, th, net.forward(th, q, qd, u, c_hat, h_hat))
    h = 1e-6
    J_fd = np.zeros_like(J)
    for i in range(th.size):
        tp, tm = th.copy(), th.copy()
        tp[i] += h
        tm[i] -= h
        J_fd[:, i] = (net.forward(tp, q, qd, u, c_hat, h_hat).phi - net.forward(tm, q, qd, u, c_hat, h_hat).phi) / (2 * h)
    for b in net.layout.BLOCKS:
        sl = net.layout.block_slice(b)
        assert np.abs(J_fd[:, sl]).max() > 1e-4, f"block {b} degenerate"
        assert np.allclose(J[:, sl], J_fd[:, sl], atol=1e-7, rtol=1e-6), f"block {b}"


def test_kinetic_metric_gradient_is_mass_weighted() -> None:
    net = _net()
    rng = np.random.default_rng(3)
    th = _random_theta(net, 0.3, 1)
    q, qd, u, c_hat, h_hat = _random_point(rng)
    cache = net.forward(th, q, qd, u, c_hat, h_hat)
    J = pilstm_jacobian(net, th, cache)
    e = rng.normal(size=2)
    assert np.allclose(pilstm_jacobian_transpose_vec(net, th, cache, e), J.T @ e, atol=1e-12)
    law = PILSTMAdaptationLaw(net, {"M": 1, "V": 1, "G": 1, "F_gates": 1, "F_out": 1},
                              {"M": 1e3, "V": 1e3, "G": 1e3, "F": 1e3}, metric="kinetic")
    assert np.allclose(law.theta_dot(th, cache, e), J.T @ cache.inertia.M @ e, atol=1e-10)


def test_projection_keeps_every_block_in_its_ball() -> None:
    cfg = PILSTMObserverConfig(w_bar_G=0.3, gamma_G=500.0, w_bar_V=0.02, gamma_V=50.0)
    obs = PILSTMObserver(cfg)
    plant = PendulumPlant()
    plant.reset(np.array([0.0, 0.0, 0.5, 0.0]))
    obs.reset(np.array([0.0, 0.0, 0.5, 0.0]))
    for k in range(3000):
        s = plant.state.to_array()
        u = 2.0 * np.sin(3.0 * k * 1e-3)
        obs.update(s[[0, 2]], u, 1e-3)
        plant.step(u, 1e-3)
    norms = obs.block_norms()
    for b, w in (("M", cfg.w_bar_M), ("V", cfg.w_bar_V), ("G", cfg.w_bar_G), ("F", cfg.w_bar_F)):
        assert norms[b] <= w + 1e-9


# ---------------------------------------------------------------------------- observer
def test_observer_tracks_swinging_pendulum_and_keeps_invariants() -> None:
    dt = 1e-3
    plant = PendulumPlant()
    x0 = np.array([0.0, -2.0 / (3.0 * 2.63), 0.5, 0.0])
    plant.reset(x0)
    obs = PILSTMObserver(PILSTMObserverConfig(velocity_scale=(2.0, 0.2)))
    obs.reset(np.array([0.0, 0.0, 0.5, 0.0]))
    rng = np.random.default_rng(1)
    errs = []
    for k in range(12000):
        s = plant.state.to_array()
        u = 2.0 * np.sin(3.0 * k * dt) + 1.2 * np.cos(6.0 * k * dt)
        y = s[[0, 2]] + np.array([0.0, np.deg2rad(rng.uniform(-1.0, 1.0))])
        est = obs.update(y, u, dt)
        plant.step(u, dt)
        errs.append(np.abs(est[[1, 3]] - s[[1, 3]]))
        assert obs.last_cache is not None
        assert np.linalg.eigvalsh(obs.last_cache.inertia.M)[0] >= obs.config.eps_M - 1e-12
        assert obs.kinetic_energy_estimate() >= 0.0
        assert obs.friction_power() >= 0.0
    errs_arr = np.array(errs)
    assert np.all(np.isfinite(errs_arr))
    # theta_dot swings up to ~5 rad/s; after the transient the error is a few percent of that
    assert np.sqrt(np.mean(errs_arr[8000:, 1] ** 2)) < 0.1
    assert np.sqrt(np.mean(errs_arr[8000:, 0] ** 2)) < 0.02


def test_observer_interface_matches_blackbox() -> None:
    obs = PILSTMObserver()
    obs.reset(np.array([0.1, 0.2, 3.0, -0.1]))
    assert np.allclose(obs.state_estimate(), [0.1, 0.2, 3.0, -0.1])
    out = obs.update(np.array([0.1, 3.0]), 0.5, 1e-3)
    assert out.shape == (4,)
    obs.reset(reset_weights=True)
    assert np.allclose(obs.theta, obs.theta0)


def test_invalid_configurations_are_rejected() -> None:
    with pytest.raises(ValueError):
        PILSTMObserver(PILSTMObserverConfig(eps_M=0.0))
    with pytest.raises(ValueError):
        PILSTMObserver(PILSTMObserverConfig(inertia_init=(0.01, 0.2)))     # below eps_M
    with pytest.raises(ValueError):
        PILSTMObserver(PILSTMObserverConfig(cyclic=(True, True)))
    with pytest.raises(ValueError):
        PILSTMObserver(PILSTMObserverConfig(w_bar_M=0.1))                  # theta_M(0) outside ball
