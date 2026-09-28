"""
Unit tests for the Lb-LSTM system identification stage (frozen digital twin).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.identification.extract_model import (
    FrozenLbLSTM,
    LbLSTMDigitalTwin,
    chirp,
    freeze_observer,
    step_doublet,
)
from src.observers.blackbox_lstm import LbLSTMObserver, LbLSTMObserverConfig


def _frozen(seed: int = 0) -> FrozenLbLSTM:
    cfg = LbLSTMObserverConfig(input_scale=(3.0, 3.0, 2.0, 1.0, 0.5), input_offset=(0.0, 3.1, 0.0, 0.0, 0.0), seed=seed)
    obs = LbLSTMObserver(cfg)
    rng = np.random.default_rng(seed)
    obs.theta[obs.layout.n_gate_params:] = rng.normal(0.0, 0.5, obs.layout.n_out_params)
    return freeze_observer(obs, t_now=25.0)


def test_freeze_refuses_before_settling() -> None:
    with pytest.raises(ValueError):
        freeze_observer(LbLSTMObserver(), t_now=5.0)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    frozen = _frozen()
    path = tmp_path / "w.npz"
    frozen.save(path)
    loaded = FrozenLbLSTM.load(path)
    assert np.array_equal(loaded.theta, frozen.theta)
    assert loaded.config == frozen.config
    assert loaded.t_freeze == frozen.t_freeze


def test_frozen_copy_is_decoupled_from_observer() -> None:
    obs = LbLSTMObserver()
    frozen = freeze_observer(obs, t_now=20.0)
    obs.theta += 1.0
    assert not np.array_equal(frozen.theta, obs.theta)


def test_twin_matches_observer_lstm_at_same_state() -> None:
    frozen = _frozen()
    twin = LbLSTMDigitalTwin(frozen)
    obs = LbLSTMObserver(frozen.config)
    obs.theta = frozen.theta.copy()
    rng = np.random.default_rng(4)
    obs.x1_hat, obs.x2_hat = rng.normal(size=2), rng.normal(size=2)
    obs.c_hat, obs.h_hat = rng.normal(size=16), rng.normal(size=16)
    u = np.array([0.8])
    phi_obs = obs.lstm(u).phi
    phi_twin = twin._lstm(obs.x1_hat, obs.x2_hat, u, obs.c_hat, obs.h_hat).phi
    assert np.allclose(phi_obs, phi_twin)


def test_zero_readout_twin_is_a_pure_double_integrator() -> None:
    frozen = _frozen()
    frozen.theta[LbLSTMObserver(frozen.config).layout.n_gate_params:] = 0.0
    twin = LbLSTMDigitalTwin(frozen)
    dt = 1e-3
    roll = twin.simulate(np.ones(1000), np.array([0.0, 0.2, np.pi, -0.1]), dt, t_settle=0.1)
    assert np.allclose(roll.accel, 0.0)
    assert np.allclose(roll.state[-1], [0.2 * 999 * dt, 0.2, np.pi - 0.1 * 999 * dt, -0.1])


def test_unseen_input_signals() -> None:
    u = step_doublet(amplitude=1.5, t_on=1.0, width=0.5)
    t = np.arange(0.0, 3.0, 1e-3)
    samples = np.array([u(tk) for tk in t])
    assert samples[0] == 0.0 and samples.max() == 1.5 and samples.min() == -1.5
    assert abs(np.sum(samples) * 1e-3) < 1e-2  # zero net impulse
    c = chirp(amplitude=1.0, f0=0.3, f1=1.2, t_sweep=10.0, t_ramp=2.0)
    assert c(0.0) == 0.0 and c(10.5) == 0.0
    assert max(abs(c(tk)) for tk in t) <= 1.0
