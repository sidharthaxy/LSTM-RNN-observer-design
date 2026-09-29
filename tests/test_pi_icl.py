"""
Unit tests for Approach D: linear-in-parameters Euler-Lagrange model, Savitzky-Golay smoother,
integral-CL regressors and the PI-ICL observer.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.concurrent_learning.savitzky_golay import SavitzkyGolay
from src.observers.el_linear_model import LinearELModel
from src.observers.pi_icl_observer import PIICLObserver, PIICLObserverConfig
from src.plant.pendulum_plant import PendulumParameters, PendulumPlant
from src.simulation.open_loop import simulate_open_loop

DT = 1e-3


@pytest.fixture(params=[1, 2])
def model(request: pytest.FixtureRequest) -> LinearELModel:
    return LinearELModel(harmonics=request.param)


# ---------------------------------------------------------------------- model
def test_true_parameters_reproduce_the_plant(model: LinearELModel) -> None:
    p = PendulumParameters()
    plant = PendulumPlant(p)
    th = model.true_parameters(p)
    rng = np.random.default_rng(0)
    for _ in range(100):
        s = rng.normal(size=4) * np.array([0.3, 1.0, 3.0, 4.0])
        u = float(rng.normal() * 3.0)
        q, qd = s[[0, 2]], s[[1, 3]]
        a_true = np.array(plant.forward_dynamics(s, u))
        a_hat, M = model.acceleration(th, q, qd, np.array([u]))
        np.testing.assert_allclose(a_hat, a_true, atol=1e-12)
        np.testing.assert_allclose(M, plant.mass_matrix(s[2]), atol=1e-14)
        np.testing.assert_allclose(model.torque_regressor(q, qd, a_true) @ th, model.B[:, 0] * u, atol=1e-10)


def test_structural_invariants_hold_for_random_parameters(model: LinearELModel) -> None:
    rng = np.random.default_rng(1)
    th = rng.normal(size=model.layout.n_params)
    for _ in range(20):
        q, qd = rng.normal(size=2) * 3.0, rng.normal(size=2)
        M = model.inertia(th, q)
        np.testing.assert_allclose(M, M.T)
        N = model.skew_residual(th, q, qd)
        np.testing.assert_allclose(N + N.T, 0.0, atol=1e-12)              # M_dot - 2C skew-symmetric
        # G = dP/dq (conservative), checked by central differences of the learned potential
        h = 1e-6
        grad = [(model.potential_energy(th, q + h * e) - model.potential_energy(th, q - h * e)) / (2 * h) for e in np.eye(2)]
        G = model.regressors(q, qd)["G"] @ th[model.layout.block_slice("G")]
        np.testing.assert_allclose(G, grad, atol=1e-7)
        # the cart position is cyclic
        np.testing.assert_allclose(model.inertia(th, q + np.array([0.7, 0.0])), M)


def test_regressors_are_consistent(model: LinearELModel) -> None:
    """Y(q, q_dot, a) = d/dt Y_mom + Y_int, and observer_terms returns Y at a = Phi_hat."""
    rng = np.random.default_rng(2)
    th = model.prior_parameters((2.0, 0.2)) + 0.01 * rng.normal(size=model.layout.n_params)
    q, qd, a = rng.normal(size=2), rng.normal(size=2), rng.normal(size=2)
    h = 1e-6
    dYmom = (model.momentum_regressor(q + h * qd, qd + h * a) - model.momentum_regressor(q - h * qd, qd - h * a)) / (2 * h)
    np.testing.assert_allclose(dYmom + model.integrand_regressor(q, qd), model.torque_regressor(q, qd, a), atol=1e-6)
    phi, M, Y = model.observer_terms(th, q, qd, np.array([0.3]))
    np.testing.assert_allclose(Y, model.torque_regressor(q, qd, phi), atol=1e-12)
    np.testing.assert_allclose(M @ phi + model.rest_regressor(q, qd) @ th, model.B[:, 0] * 0.3, atol=1e-12)


def test_icl_identity_holds_on_a_trajectory(model: LinearELModel) -> None:
    th = model.true_parameters(PendulumParameters())
    tr = simulate_open_loop(lambda t: 2.0 * np.sin(3.0 * t), 2.0, DT, np.array([0.0, 0.0, 0.5, 0.0]),
                            apply_actuator_effects=False)
    k0, k1 = 500, 800
    Ym0, _ = model.icl_regressors(tr.positions[k0], tr.velocities[k0])
    Ym1, _ = model.icl_regressors(tr.positions[k1], tr.velocities[k1])
    Yi = np.array([model.integrand_regressor(tr.positions[k], tr.velocities[k]) for k in range(k0, k1 + 1)])
    integral = DT * (0.5 * Yi[0] + Yi[1:-1].sum(axis=0) + 0.5 * Yi[-1])
    lhs = (Ym1 - Ym0 + integral) @ th
    rhs = model.B[:, 0] * tr.u_net[k0:k1].sum() * DT
    np.testing.assert_allclose(lhs, rhs, atol=5e-5)


def test_pd_projection_restores_positive_inertia() -> None:
    m = LinearELModel(harmonics=2)
    rng = np.random.default_rng(3)
    th = m.prior_parameters((2.0, 0.2))
    th[m.layout.block_slice("M")] += 0.4 * rng.normal(size=m.layout.sizes["M"])
    grid = m.grid_features(m.angle_grid(72))
    assert m.min_inertia_eigenvalue(th, grid)[0] < 0.05
    m.project_positive_inertia(th, grid, 0.05, np.ones(m.layout.n_params), max_iter=200)
    assert m.min_inertia_eigenvalue(th, grid)[0] >= 0.05 - 1e-9
    fine = m.grid_features(m.angle_grid(2000))
    assert m.min_inertia_eigenvalue(th, fine)[0] > 0.0


def test_physical_parameter_reading() -> None:
    m = LinearELModel()
    p = PendulumParameters()
    phys = m.physical_parameters(m.true_parameters(p))
    for k, v in LinearELModel.physical_truth(p).items():
        assert phys[k] == pytest.approx(v)


# ---------------------------------------------------------------------- smoother
def test_savitzky_golay_exact_on_cubics() -> None:
    for where in ("centre", "end"):
        sg = SavitzkyGolay.from_duration(0.05, 3, DT, where)  # type: ignore[arg-type]
        c = sg.window - 1 - sg.delay
        t = (np.arange(sg.window) - c) * DT
        y = (1.0 + 2.0 * t - 3.0 * t ** 2 + 0.5 * t ** 3)[:, None]
        assert sg.apply(y, 0)[0] == pytest.approx(1.0, abs=1e-9)
        assert sg.apply(y, 1)[0] == pytest.approx(2.0, abs=1e-6)
        assert sg.apply(y, 2)[0] == pytest.approx(-6.0, abs=1e-3)
    sg_c, sg_e = SavitzkyGolay.from_duration(0.1, 3, DT), SavitzkyGolay.from_duration(0.1, 3, DT, "end")
    assert sg_c.noise_gain(1) < sg_e.noise_gain(1)


# ---------------------------------------------------------------------- observer
def _run(obs: PIICLObserver, u_fn, T: float, x0: np.ndarray, p: PendulumParameters | None = None, seed: int = 1):
    tr = simulate_open_loop(u_fn, T, DT, x0, plant_params=p, apply_actuator_effects=False, seed=seed)
    obs.reset(np.array([tr.measurements[0, 0], 0.0, tr.measurements[0, 1], 0.0]))
    est = np.zeros((len(tr.t), 4))
    for k in range(len(tr.t)):
        est[k] = obs.update(tr.measurements[k], tr.u_cmd[k], DT)
    return tr, est


def test_stack_structure_for_the_rig() -> None:
    obs = PIICLObserver()
    lay = obs.model.layout
    assert obs.act_rows.tolist() == [0]
    assert len(obs.unact) == 1 and obs.unact[0].row == 1
    u = obs.unact[0]
    assert u.col_J == lay.block_slice("M").start + 2 * lay.n_features     # constant feature of M_22
    assert set(u.cols[u.own]) >= set(range(*lay.block_slice("G").indices(lay.n_params)))
    assert set(obs.cols_A) | set(u.cols) | {u.col_J} == set(range(lay.n_params))


def test_identifies_physical_parameters_without_noise_and_detects_a_change() -> None:
    """Noise-free encoder (quantization only), swing-up start, +50 % pendulum mass at 12 s."""
    from dataclasses import replace
    from src.plant.sensor_noise import SensorNoiseConfig
    p0 = PendulumParameters()
    p1 = replace(p0, m=1.5 * p0.m, I=1.5 * p0.I)
    cfg = PIICLObserverConfig()
    obs = PIICLObserver(cfg)
    u_fn = lambda t: 2.0 * np.sin(3.0 * t) + 1.2 * np.cos(6.0 * t)
    # two-phase simulation with a parameter step
    from src.plant.sensor_noise import SensorNoiseModel
    plant = PendulumPlant(p0)
    plant.reset(np.array([0.0, -2.0 / (3.0 * (p0.M + p0.m)), 0.5, 0.0]))
    noise = SensorNoiseModel(SensorNoiseConfig(theta_noise_deg=0.0, x_noise_mm=0.0))
    obs.reset(np.array([0.0, 0.0, 0.5, 0.0]))
    truth = LinearELModel.physical_truth
    T, T_step = 21.0, 12.0          # the cart reaches a bumper at ~23 s in this variant (impacts are not EL)
    est_pre = None
    x_max = 0.0
    for k in range(int(T / DT)):
        t = k * DT
        if abs(t - T_step) < DT / 2:
            est_pre = obs.physical_parameters()
            plant.params = p1
        s = plant.state.to_array()
        x_max = max(x_max, abs(s[0]))
        y = np.array(noise.measure(s[0], s[2]))
        u = u_fn(t)
        obs.update(y, u, DT)
        plant.step(u, DT)
    assert est_pre is not None and x_max < 0.5
    for name, ref, est in (("pre", truth(p0), est_pre), ("post", truth(p1), obs.physical_parameters())):
        for key in ("M+m", "ml", "I+ml^2", "mgl"):
            assert est[key] == pytest.approx(ref[key], rel=0.05), (name, key)
    assert len(obs.purge_times) >= 1 and T_step < obs.purge_times[0] < T_step + 2.0
    assert obs.lambda_min() > 0.0


def test_cl_off_leaves_parameters_to_the_instantaneous_law() -> None:
    obs = PIICLObserver(PIICLObserverConfig(cl_enabled=False, detect_changes=False, gamma_inst=0.0))
    _run(obs, lambda t: 2.0 * np.sin(3.0 * t), 1.0, np.array([0.0, 0.0, 0.5, 0.0]))
    np.testing.assert_allclose(obs.theta, obs.theta0)
    assert len(obs.stack_A) == 0


def test_invariants_hold_along_an_online_run() -> None:
    obs = PIICLObserver()
    tr, est = _run(obs, lambda t: 2.0 * np.sin(3.0 * t) + 1.2 * np.cos(6.0 * t), 6.0, np.array([0.0, -0.25, 0.5, 0.0]))
    assert np.all(np.isfinite(est))
    grid = obs.model.grid_features(obs.model.angle_grid(360))
    assert obs.model.min_inertia_eigenvalue(obs.theta, grid)[0] >= obs.config.eps_M - 1e-6
    lay = obs.model.layout
    assert np.all(obs.theta[lay.block_slice("Fv")] >= 0.0) and np.all(obs.theta[lay.block_slice("Fc")] >= 0.0)
    assert np.linalg.norm(obs.phi - obs.phi0) <= obs.config.w_bar + 1e-9
