"""
Unit tests for Approach C: concurrent-learning Lb-LSTM observer, acceleration proxy and
predictor benchmark plumbing.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.adaptation.jacobian_engine import gate_sensitivities, lstm_forward, phi_jacobian
from src.concurrent_learning.point_history_stack import HistoryStackEntry
from src.concurrent_learning.scenarios import (
    DT,
    ZETA_INPUT_SCALE,
    drift_cancelling_state,
    measurement_offset,
    simulate_weak_excitation,
    single_sine,
)
from src.identification.benchmark_predictor import (
    MODEL_C,
    evaluate_model,
    short_horizon_rollout,
    train_models,
)
from src.identification.extract_model import LbLSTMDigitalTwin
from src.observers.blackbox_lstm import LbLSTMObserver, LbLSTMObserverConfig
from src.observers.cl_lstm_observer import (
    CLLbLSTMObserver,
    CLLbLSTMObserverConfig,
    SavitzkyGolayAccelerationProxy,
    batch_lstm,
)
from src.simulation.open_loop import simulate_open_loop


# ---------------------------------------------------------------------- acceleration proxy
def test_sg_proxy_is_exact_on_cubics_and_centered() -> None:
    proxy = SavitzkyGolayAccelerationProxy.from_duration(0.1, 3, DT)
    assert proxy.window % 2 == 1 and proxy.delay == (proxy.window - 1) // 2
    t = np.arange(proxy.window) * DT
    tc = t[proxy.delay]
    q = np.column_stack((1.0 + 2.0 * t - 3.0 * t ** 2 + 0.5 * t ** 3, np.sin(0.0 * t) + 4.0 * t ** 2))
    pos, vel, acc = proxy.estimate(q)
    np.testing.assert_allclose(pos, [1.0 + 2.0 * tc - 3.0 * tc ** 2 + 0.5 * tc ** 3, 4.0 * tc ** 2], atol=1e-9)
    np.testing.assert_allclose(vel, [2.0 - 6.0 * tc + 1.5 * tc ** 2, 8.0 * tc], atol=1e-7)
    np.testing.assert_allclose(acc, [-6.0 + 3.0 * tc, 8.0], atol=1e-5)


def test_sg_proxy_noise_gain_matches_theory() -> None:
    proxy = SavitzkyGolayAccelerationProxy.from_duration(0.25, 3, DT)
    T_w = proxy.window * DT
    assert proxy.noise_gain == pytest.approx(np.sqrt(720.0 * DT / T_w ** 5), rel=0.05)
    rng = np.random.default_rng(0)
    est = [proxy.estimate(rng.normal(0.0, 1e-3, size=(proxy.window, 1)))[2][0] for _ in range(2000)]
    assert np.std(est) == pytest.approx(1e-3 * proxy.noise_gain, rel=0.1)


# ---------------------------------------------------------------------- batched LSTM / Jacobians
def _observer_with_stack(n_points: int = 20, seed: int = 1, **cfg: object) -> CLLbLSTMObserver:
    rng = np.random.default_rng(seed)
    obs = CLLbLSTMObserver(CLLbLSTMObserverConfig(stack_capacity=max(n_points, 16), novelty_tol=0.0, **cfg))  # type: ignore[arg-type]
    obs.theta[obs.h_slice] = rng.normal(size=obs.h_slice.stop - obs.h_slice.start)
    for j in range(n_points):
        z = rng.normal(size=obs.zeta_dim)
        c = rng.normal(size=obs.config.hidden_dim)
        a = rng.normal(size=2) * np.array([0.3, 2.0])
        h = lstm_forward(obs.theta, z, c, obs.layout).h
        obs.stack.consider(HistoryStackEntry(z, c, np.zeros(1), np.zeros(2), a, float(j), h))
    return obs


def test_batch_lstm_matches_single_point_engine() -> None:
    obs = _observer_with_stack(5)
    Z = np.vstack([e.zeta for e in obs.stack.entries])
    C = np.vstack([e.c_hat for e in obs.stack.entries])
    H, delta = batch_lstm(obs.theta, Z, C, obs.layout, sensitivities=True)
    assert delta is not None
    for j in range(len(Z)):
        cache = lstm_forward(obs.theta, Z[j], C[j], obs.layout)
        np.testing.assert_allclose(H[j], cache.h, atol=1e-14)
        np.testing.assert_allclose(delta[j], gate_sensitivities(cache), atol=1e-14)


@pytest.mark.parametrize("normalize", [False, True])
def test_stack_gradient_equals_explicit_jacobian_sum(normalize: bool) -> None:
    """Gate and readout blocks of sum_j Phi'(t_j)^T Lambda (a_j - Phi_hat_j) (full Jacobian)."""
    obs = _observer_with_stack(20, cl_normalize_outputs=normalize)
    lam = obs.output_weights()
    if normalize:
        assert not np.allclose(lam, 1.0)
    ref = np.zeros(obs.layout.n_params)
    for e in obs.stack.entries:
        cache = lstm_forward(obs.theta, e.zeta, e.c_hat, obs.layout)
        ref += phi_jacobian(cache, obs.theta, obs.layout).T @ (lam * (e.accel_target - cache.phi))
    n_g = obs.layout.n_gate_params
    np.testing.assert_allclose(obs.stack_gate_gradient(), ref[:n_g], rtol=1e-10, atol=1e-12)
    readout = (obs.cl_residuals() * lam).T @ obs.stack_features()      # (n, L) = vec(H^T E Lambda)
    np.testing.assert_allclose(readout.ravel(), ref[n_g:], rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------- CL loop
def _cl_only_observer(W_star: np.ndarray, gamma_cl: float, seed: int = 4) -> CLLbLSTMObserver:
    """Frozen gates, instantaneous law off, stack filled with realizable targets a_j = W*^T h_j."""
    cfg = CLLbLSTMObserverConfig(
        gamma_out=0.0, gamma_gates=0.0, gamma_cl_gates=0.0, gamma_cl=gamma_cl,
        stack_capacity=24, novelty_tol=0.0, record_start=1e9, w_bar=1e3, seed=seed,
    )
    obs = CLLbLSTMObserver(cfg)
    rng = np.random.default_rng(seed)
    for j in range(24):
        z = rng.normal(size=obs.zeta_dim)
        c = rng.normal(size=cfg.hidden_dim)
        h = lstm_forward(obs.theta, z, c, obs.layout).h
        obs.stack.consider(HistoryStackEntry(z, c, np.zeros(1), np.zeros(2), W_star.T @ h, float(j), h))
    return obs


def test_cl_converges_exponentially_to_true_parameters_without_excitation() -> None:
    """
    Realizable case of the CL theorem: with the rank condition met and no instantaneous
    excitation at all (y = x_hat, u = 0), W_h -> W* at rate >= gamma_cl lambda_min(Omega).
    """
    rng = np.random.default_rng(5)
    W_star = rng.normal(size=(16, 2))
    gamma_cl = 20.0
    obs = _cl_only_observer(W_star, gamma_cl)
    lam = obs.lambda_min()
    assert lam > 0.0
    err0 = np.linalg.norm(obs.W_h - W_star)
    T = 2.0
    for _ in range(int(T / DT)):
        obs.update(np.zeros(2), 0.0, DT)
    err = np.linalg.norm(obs.W_h - W_star)
    assert err < err0 * np.exp(-gamma_cl * lam * T) * 1.05 + 1e-9
    assert err < 1e-3 * err0


def test_standard_law_does_not_identify_without_excitation() -> None:
    """Same data, CL loop off: the weights do not move (the stack is recorded but unused)."""
    W_star = np.random.default_rng(6).normal(size=(16, 2))
    obs = _cl_only_observer(W_star, gamma_cl=20.0)
    obs.cl_config.cl_enabled = False
    W0 = obs.W_h
    for _ in range(500):
        obs.update(np.zeros(2), 0.0, DT)
    np.testing.assert_allclose(obs.W_h, W0)


def test_implicit_readout_step_solves_the_implicit_equation() -> None:
    """W+ = W + dt Gamma_CL (B - Omega W+) with a stiff gain (dt Gamma lambda_max >> 2)."""
    W_star = np.random.default_rng(7).normal(size=(16, 2))
    obs = _cl_only_observer(W_star, gamma_cl=1e5)
    assert DT * 1e5 * obs.stack.lambda_max() > 2.0
    W = obs.W_h
    obs.update(np.zeros(2), 0.0, DT)
    H, A = obs.stack.data_matrix(), obs.stack.targets()
    W_plus = obs.W_h
    np.testing.assert_allclose(W_plus, W + DT * 1e5 * (H.T @ A - H.T @ H @ W_plus), atol=1e-8)
    assert np.all(np.isfinite(W_plus))


def test_cl_disabled_with_frozen_gates_reproduces_approach_a_law() -> None:
    """With the CL loop off, the CL observer is exactly Approach A's observer (same gains)."""
    traj, _ = simulate_weak_excitation("decaying", 3.0)
    off = measurement_offset(traj)
    common = dict(input_scale=ZETA_INPUT_SCALE, input_offset=off, gamma_gates=0.0, w_bar=80.0, seed=3)
    a = LbLSTMObserver(LbLSTMObserverConfig(**common))  # type: ignore[arg-type]
    c = CLLbLSTMObserver(CLLbLSTMObserverConfig(cl_enabled=False, **common))  # type: ignore[arg-type]
    y0 = traj.measurements[0]
    for obs in (a, c):
        obs.reset(np.array([y0[0], 0.0, y0[1], 0.0]))
    ea = ec = np.zeros(4)
    for k in range(len(traj.t)):
        ea = a.update(traj.measurements[k], traj.u_cmd[k], DT)
        ec = c.update(traj.measurements[k], traj.u_cmd[k], DT)
    np.testing.assert_allclose(ec, ea, atol=1e-12)
    np.testing.assert_allclose(c.theta, a.theta, atol=1e-12)


def test_online_run_meets_rank_condition_and_stays_bounded() -> None:
    traj, _ = simulate_weak_excitation("decaying", 8.0)
    cfg = CLLbLSTMObserverConfig(input_scale=ZETA_INPUT_SCALE, input_offset=measurement_offset(traj), seed=0)
    assert cfg.gamma_cl_gates == 5.0 and not cfg.cl_normalize_outputs   # locked defaults
    obs = CLLbLSTMObserver(cfg)
    y0 = traj.measurements[0]
    obs.reset(np.array([y0[0], 0.0, y0[1], 0.0]))
    est = np.zeros(4)
    for k in range(len(traj.t)):
        est = obs.update(traj.measurements[k], traj.u_cmd[k], DT)
    assert len(obs.stack) == cfg.stack_capacity
    assert obs.lambda_min() > 0.0 and obs.stack.rank() == cfg.hidden_dim
    assert all(e.t >= cfg.record_start for e in obs.stack.entries)
    assert np.all(np.isfinite(est)) and np.linalg.norm(obs.theta) <= cfg.w_bar + 1e-9
    # The proxy targets track the true acceleration of the recorded instants.
    idx = np.array([round(e.t / DT) for e in obs.stack.entries])
    err = obs.stack.targets() - traj.accel[idx]
    assert np.sqrt(np.mean(err[:, 1] ** 2)) < 0.25 * np.sqrt(np.mean(traj.accel[idx, 1] ** 2))
    # ~5 s after the rank condition the CL loop has removed most of the initial distance
    # ||W_h(0) - W_H|| = ||W_H|| to its fixed point (the 60 s behaviour is in run_cl_validation).
    W_H = obs.stack_least_squares()
    assert W_H is not None and np.linalg.norm(obs.W_h - W_H) < 0.5 * np.linalg.norm(W_H)


def test_freeze_produces_a_digital_twin() -> None:
    obs = _observer_with_stack(16)
    frozen = obs.freeze()
    assert type(frozen.config) is LbLSTMObserverConfig
    twin = LbLSTMDigitalTwin(frozen)
    out = twin.simulate(np.zeros(50), np.array([0.0, 0.0, np.pi, 0.0]), DT, t_settle=0.1)
    assert out.state.shape == (50, 4) and np.all(np.isfinite(out.state))


# ---------------------------------------------------------------------- scenarios / benchmark
def test_drift_cancelling_state_keeps_the_cart_on_the_track() -> None:
    u = single_sine(amplitude=1.0, omega=1.2)
    x0, v0 = drift_cancelling_state(u, 12.0)
    tr = simulate_open_loop(u, 12.0, DT, np.array([x0, v0, np.pi, 0.0]), apply_actuator_effects=False)
    assert np.max(np.abs(tr.state[:, 0])) < 0.5


def test_short_horizon_rollout_restarts_from_truth_and_benchmark_runs() -> None:
    traj, _ = simulate_weak_excitation("decaying", 4.0)
    model = train_models(traj, seed=0, include=(MODEL_C,))[MODEL_C]
    test = simulate_open_loop(single_sine(1.0, 1.2), 1.0, DT, np.array([0.0, 0.0, np.pi, 0.0]),
                              apply_actuator_effects=False)
    twin = LbLSTMDigitalTwin(model.frozen)
    pred = short_horizon_rollout(twin, test, horizon=0.25)
    for k0 in range(0, len(test.t), 250):
        np.testing.assert_allclose(pred[k0], test.state[k0], atol=1e-12)
    res = evaluate_model(MODEL_C, model.frozen, "sine", test)
    for key in ("one-step NMSE x_ddot", "one-step NMSE theta_ddot", "short-horizon NMSE x", "short-horizon NMSE theta"):
        assert np.isfinite(res.metrics[key])
