"""
Unit tests for the shared benchmark harness: scenarios, model registry, twin adapters, warm start.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.benchmark.models import (
    MODELS,
    ELTwin,
    LSTMTwin,
    PILSTMTwin,
    make_observer,
    make_twin,
    nmse,
    short_horizon,
    teacher_forced,
    twin_metrics,
)
from src.benchmark.scenarios import DT, HELD_OUT, TRAINING, Dataset, held_out_tests, simulate
from src.benchmark.warm_start import WarmStartObserver
from src.plant.pendulum_plant import PendulumParameters


def _crop(d: Dataset, n: int) -> Dataset:
    return replace(d, t=d.t[:n], state=d.state[:n], accel=d.accel[:n], meas=d.meas[:n], u=d.u[:n])


@pytest.fixture(scope="module")
def short_s1() -> Dataset:
    return _crop(TRAINING["S1"](0), 1500)


def test_training_scenarios_reproduce_their_branch_settings() -> None:
    s2 = TRAINING["S2"](0)
    assert s2.shift_time == 25.0 and s2.params_final.m == pytest.approx(1.5 * PendulumParameters().m)
    assert s2.state[0, 2] == pytest.approx(0.5)
    for key in TRAINING:
        d = TRAINING[key](0) if key != "S2" else s2
        assert np.max(np.abs(d.state[:, 0])) < 0.5          # no bumper contact


def test_held_out_tests_stay_on_the_track_and_are_noise_free() -> None:
    tests = held_out_tests(PendulumParameters())
    assert [t.name for t in tests] == list(HELD_OUT)
    for d in tests:
        assert np.max(np.abs(d.state[:, 0])) < 0.5
        np.testing.assert_allclose(d.meas[:, 1], d.state[:, 2], atol=2e-3)   # quantization only


def test_every_model_builds_and_runs(short_s1: Dataset) -> None:
    for name in MODELS:
        obs = make_observer(name, short_s1, seed=0)
        obs.reset(np.array([short_s1.meas[0, 0], 0.0, short_s1.meas[0, 1], 0.0]))
        for k in range(200):
            est = obs.update(short_s1.meas[k], short_s1.u[k], DT)
        assert np.all(np.isfinite(est))


def test_x_free_variants_ignore_the_cart_position(short_s1: Dataset) -> None:
    for name in ("A-x", "C-x"):
        obs = make_observer(name, short_s1, 0)
        z1 = obs.build_zeta(np.array([0.0, np.pi]), np.zeros(2), np.zeros(1), np.zeros(16))  # type: ignore[attr-defined]
        z2 = obs.build_zeta(np.array([0.4, np.pi]), np.zeros(2), np.zeros(1), np.zeros(16))  # type: ignore[attr-defined]
        np.testing.assert_allclose(z1, z2)
    obs = make_observer("A", short_s1, 0)
    z1 = obs.build_zeta(np.array([0.0, np.pi]), np.zeros(2), np.zeros(1), np.zeros(16))  # type: ignore[attr-defined]
    z2 = obs.build_zeta(np.array([0.4, np.pi]), np.zeros(2), np.zeros(1), np.zeros(16))  # type: ignore[attr-defined]
    assert not np.allclose(z1, z2)


def test_twin_adapters_match_their_observers(short_s1: Dataset) -> None:
    kinds = {"A": LSTMTwin, "C": LSTMTwin, "B": PILSTMTwin, "D": ELTwin}
    for name, kind in kinds.items():
        obs = make_observer(name, short_s1, 0)
        assert isinstance(make_twin(obs), kind)


def test_el_twin_with_true_parameters_is_exact() -> None:
    """D's twin carrying theta* reproduces the plant: zero one-step error, near-zero short horizon."""
    d = held_out_tests(PendulumParameters())[1]
    d = _crop(d, 3000)
    obs = make_observer("D", d, 0)
    obs.theta = obs.model.true_parameters(PendulumParameters())   # type: ignore[attr-defined]
    twin = make_twin(obs)
    acc, mems = teacher_forced(twin, d)
    np.testing.assert_allclose(acc, d.accel, atol=1e-10)
    pred = short_horizon(twin, d, mems)
    assert nmse(d.state[:, 2], pred[:, 2]) < 1e-8
    m = twin_metrics(twin, d)
    assert m["one-step NMSE theta_ddot"] < 1e-12


def test_warm_start_removes_the_initial_velocity_error() -> None:
    d = simulate("drift", lambda t: 0.0, 1.0, np.array([0.0, -0.5, np.pi, 0.0]), 0.5, 1)
    cold = make_observer("D", d, 0)
    warm = WarmStartObserver(make_observer("D", d, 0), DT)
    for o in (cold, warm):
        o.reset(np.array([d.meas[0, 0], 0.0, d.meas[0, 1], 0.0]))
    e_cold, e_warm = [], []
    for k in range(len(d.t)):
        e_cold.append(cold.update(d.meas[k], d.u[k], DT)[1] - d.state[k, 1])
        e_warm.append(warm.update(d.meas[k], d.u[k], DT)[1] - d.state[k, 1])
    k0 = warm.sg.window
    assert warm.started and warm.initial_estimate is not None and warm.injected is not None
    assert warm.injected[0] and abs(warm.initial_estimate[1] - d.state[k0 - 1, 1]) < 0.01
    assert np.max(np.abs(e_warm[k0:])) < 0.1 * np.max(np.abs(e_cold[k0:]))


def test_warm_start_leaves_a_resting_coordinate_untouched() -> None:
    """The pendulum is at rest: its fitted velocity is noise, is not injected, and its estimate
    evolves exactly as without the wrapper."""
    d = simulate("rest", lambda t: 0.0, 0.3, np.array([0.0, -0.5, np.pi, 0.0]), 1.0, 3)
    cold = make_observer("D", d, 0)
    warm = WarmStartObserver(make_observer("D", d, 0), DT)
    for o in (cold, warm):
        o.reset(np.array([d.meas[0, 0], 0.0, d.meas[0, 1], 0.0]))
    for k in range(warm.sg.window):
        ec = cold.update(d.meas[k], 0.0, DT)
        ew = warm.update(d.meas[k], 0.0, DT)
    assert warm.injected is not None and warm.injected[0] and not warm.injected[1]
    assert ew[3] == ec[3]                                    # theta_dot untouched
    assert abs(ew[1] + 0.5) < 0.01                           # x_dot injected
