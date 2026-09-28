"""
Unit tests for Approach A: black-box continuous-time Lb-LSTM observer.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines.classical_estimators import DirtyDerivativeFilter
from src.observers.blackbox_lstm import LbLSTMObserver, LbLSTMObserverConfig
from src.simulation.open_loop import simulate_open_loop


def _lyapunov_residual(chi_x1_coeff: float | None) -> float:
    """
    Plugs random signals into the implemented vector field and returns
        V_dot - [ -alpha(|x~1|^2 + |eta|^2 + |nu|^2) - k_r |r|^2 + r^T (g - Phi_hat - k_s sgn(e)) ]
    for V = 1/2 (|x~1|^2 + |eta|^2 + |nu|^2 + |r|^2), r = x~2 + alpha x~1 + eta.
    """
    rng = np.random.default_rng(11)
    cfg = LbLSTMObserverConfig(alpha=3.0, k_r=7.0, k_s=0.4, chi_x1_coeff=chi_x1_coeff, adapt=False)
    obs = LbLSTMObserver(cfg)
    a, k = cfg.alpha, cfg.k_r

    obs.x1_hat, obs.x2_hat = rng.normal(size=2), rng.normal(size=2)
    obs.p, obs.nu = rng.normal(size=2), rng.normal(size=2)
    obs.c_hat, obs.h_hat = rng.normal(size=cfg.hidden_dim), rng.normal(size=cfg.hidden_dim)
    x1_tilde, x2_tilde, g = rng.normal(size=2), rng.normal(size=2), rng.normal(size=2)
    y = obs.x1_hat + x1_tilde
    d = obs.derivatives(y, np.array([0.7]))

    eta, nu = d.eta, obs.nu
    r = x2_tilde + a * x1_tilde + eta
    x2_tilde_dot = g - d.x2_hat
    eta_dot = d.p - (a + k) * x2_tilde
    r_dot = x2_tilde_dot + a * x2_tilde + eta_dot

    # Structural identities of the filter
    assert np.allclose(d.nu, eta - a * nu)
    assert np.allclose(eta_dot, -(k + a) * r - a * eta + x1_tilde - nu)
    assert np.allclose(x2_tilde + d.nu, r - a * d.e)  # e_dot = r - alpha e

    V_dot = x1_tilde @ x2_tilde + eta @ eta_dot + nu @ d.nu + r @ r_dot
    target = (-a * (x1_tilde @ x1_tilde + eta @ eta + nu @ nu) - k * (r @ r)
              + r @ (g - d.cache.phi - cfg.k_s * np.sign(d.e)))
    return float(V_dot - target)


def test_auxiliary_filter_cross_terms_cancel_exactly() -> None:
    assert abs(_lyapunov_residual(None)) < 1e-10


def test_alpha_squared_plus_two_leaves_cross_term() -> None:
    """chi with (alpha^2 + 2) x~1 leaves a -2 alpha^2 x~1^T r term in V_dot."""
    assert abs(_lyapunov_residual(3.0 ** 2 + 2.0)) > 1e-3


def test_reset_uses_interleaved_ordering() -> None:
    obs = LbLSTMObserver()
    obs.theta += 1.0
    obs.reset(np.array([0.1, -0.2, 3.0, 0.4]))
    assert np.allclose(obs.x1_hat, [0.1, 3.0])
    assert np.allclose(obs.x2_hat, [-0.2, 0.4])
    assert np.allclose(obs.state_estimate(), [0.1, -0.2, 3.0, 0.4])
    assert np.array_equal(obs.theta, obs.theta0)
    assert np.all(obs.theta0[obs.layout.n_gate_params:] == 0.0)  # Phi_hat(0) = 0: no prior model


def test_frozen_weights_do_not_change() -> None:
    obs = LbLSTMObserver(LbLSTMObserverConfig(adapt=False))
    obs.reset(np.zeros(4))
    theta_before = obs.theta.copy()
    for k in range(200):
        obs.update(np.array([0.01 * k, 0.02]), 1.0, 1e-3)
    assert np.array_equal(obs.theta, theta_before)


def test_observer_beats_dirty_derivative_on_pendulum() -> None:
    """5 s of the validation scenario: bounded weights and lower angular-velocity error."""
    dt = 1e-3
    traj = simulate_open_loop(
        lambda t: 2.0 * np.sin(1.5 * t) + 1.2 * np.cos(3.0 * t), 5.0, dt,
        np.array([0.0, -0.507, np.pi - 0.4, 0.0]), apply_actuator_effects=False, seed=0,
    )
    offset = (traj.measurements[0, 0], traj.measurements[0, 1], 0.0, 0.0, 0.0)
    cfg = LbLSTMObserverConfig(input_scale=(3.0, 3.0, 2.0, 1.0, 0.5), input_offset=offset)
    obs = LbLSTMObserver(cfg)
    obs.reset(np.array([traj.measurements[0, 0], 0.0, traj.measurements[0, 1], 0.0]))
    dirty = DirtyDerivativeFilter(0.02)
    dirty.reset(traj.measurements[0, 1])

    th_dot_obs = np.zeros(len(traj.t))
    th_dot_dirty = np.zeros(len(traj.t))
    for k in range(len(traj.t)):
        th_dot_obs[k] = obs.update(traj.measurements[k], traj.u_cmd[k], dt)[3]
        th_dot_dirty[k] = dirty.update(traj.measurements[k, 1], dt)

    tail = traj.t >= 3.0
    rmse_obs = np.sqrt(np.mean((th_dot_obs[tail] - traj.state[tail, 3]) ** 2))
    rmse_dirty = np.sqrt(np.mean((th_dot_dirty[tail] - traj.state[tail, 3]) ** 2))
    assert np.all(np.isfinite(obs.theta))
    assert obs.theta_norm <= cfg.w_bar
    assert rmse_obs < 0.5 * rmse_dirty, f"Lb-LSTM {rmse_obs:.4f} vs dirty {rmse_dirty:.4f}"


def test_input_preconditioning_validation() -> None:
    with pytest.raises(ValueError):
        LbLSTMObserver(LbLSTMObserverConfig(input_scale=(1.0, 2.0)))
    with pytest.raises(ValueError):
        LbLSTMObserver(LbLSTMObserverConfig(input_offset=(0.0,)))
