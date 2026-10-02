"""
Tests for the rig-recording loader and the rig evaluation (src/rig/rig_data.py,
experiments/run_rig_evaluation.py), on simulated runs written in the rig's file format.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.run_rig_evaluation import check_signs, deadzone_thresholds, job, rest_statistics
from src.benchmark.scenarios import simulate
from src.rig.rig_data import (
    METADATA_TEMPLATE, as_dataset, is_mass_step, load_folder, load_metadata, load_run, metadata_text, reference_derivatives,
    role_of, write_run,
)


def _write(folder: Path, name: str, u_fn, t_final: float, x0, meta: dict, noise: bool = True):
    d = simulate(name, u_fn, t_final, np.asarray(x0, dtype=float), 0.5, 42, noise=noise)
    write_run(folder / f"{name}.csv", d.t, d.meas[:, 0], d.meas[:, 1], d.u, meta)
    return d


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    (tmp_path / "metadata.txt").write_text(metadata_text(), encoding="utf-8")
    return tmp_path


def test_units_and_conventions_round_trip(folder: Path) -> None:
    """Counts and volts with the angle zero at hanging come back as metres, radians (0 = upright) and newtons."""
    meta = dict(METADATA_TEMPLATE)
    u = lambda t: 2.0 * np.sin(1.5 * t)
    d = _write(folder, "run4_twotone_trial1", u, 4.0, [0.0, -2.0 / (1.5 * 2.63), np.pi - 0.4, 0.0], meta)   # drift-free start
    run = load_run(folder / "run4_twotone_trial1.csv", meta)
    assert run.role == "train" and not run.near_stop and run.max_dt_error < 1e-6
    n = len(run.t)
    np.testing.assert_allclose(run.meas[:, 0], d.meas[:n, 0], atol=1.0 / 4096.0)             # one count
    np.testing.assert_allclose(run.meas[:, 1], d.meas[:n, 1], atol=2.0 * np.pi / 4096.0)
    np.testing.assert_allclose(run.u, d.u[:n], atol=1e-5)
    # other declared units give the same result
    for alt in ({"angle_units": "deg", "angle_zero": "upright"}, {"cart_units": "m", "command_units": "N"}, {"angle_sign": -1, "cart_sign": -1}):
        m2 = {**meta, **alt}
        write_run(folder / "run4_twotone_trial2.csv", d.t, d.meas[:, 0], d.meas[:, 1], d.u, m2)
        r2 = load_run(folder / "run4_twotone_trial2.csv", m2)
        np.testing.assert_allclose(r2.meas, d.meas[:n], atol=2e-3)
        np.testing.assert_allclose(r2.u, d.u[:n], atol=1e-5)


def test_file_checks(folder: Path) -> None:
    assert role_of("run6_heldout_chirp") == "test" and role_of("run7_mass_step_twotone") == "train"
    assert role_of("run1_rest_trial2") == "rest" and role_of("run2_freeswing_trial1") == "train" and role_of("run3_deadzone_trial1") == "deadzone"
    assert is_mass_step("run7_mass_step_twotone") and not is_mass_step("run4_twotone_trial1")
    meta = load_metadata(folder)                       # metadata.txt: key = value lines with comments
    assert meta == METADATA_TEMPLATE and isinstance(meta["cart_sign"], int) and isinstance(meta["newtons_per_volt"], float)
    (folder / "metadata.txt").write_text("angle_units = deg\nnewtons_per_volt = 1.8\n# a comment\n", encoding="utf-8")
    assert load_metadata(folder)["angle_units"] == "deg" and load_metadata(folder)["newtons_per_volt"] == 1.8
    (folder / "metadata.txt").write_text("gain = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown field"):
        load_metadata(folder)
    with pytest.raises(ValueError, match="unknown run type"):
        role_of("my_run")
    (folder / "run4_twotone_trial1.csv").write_text("time_s,cart_pos,pend_angle\n0,0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing column"):
        load_run(folder / "run4_twotone_trial1.csv", dict(METADATA_TEMPLATE))
    with pytest.raises(FileNotFoundError):
        load_folder(folder / "nowhere")


def test_reference_velocity_is_close_to_the_truth(folder: Path) -> None:
    meta = dict(METADATA_TEMPLATE)
    d = _write(folder, "run4_twotone_trial1", lambda t: 2.0 * np.sin(1.5 * t) + 1.2 * np.cos(3.0 * t), 8.0,
               [0.0, -0.5, np.pi - 0.4, 0.0], meta)
    run = load_run(folder / "run4_twotone_trial1.csv", meta)
    _, vel, _ = reference_derivatives(run.meas)
    n = len(run.t)
    err = vel[200:-200] - d.state[200:n - 200][:, [1, 3]]
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    assert rmse[0] < 0.01 and rmse[1] < 0.1          # [m/s], [rad/s]: limited by the +-0.5 deg encoder noise
    ds = as_dataset(run)
    assert ds.state.shape == (n, 4) and ds.accel.shape == (n, 2)


def test_sign_check_detects_a_flipped_convention(folder: Path) -> None:
    meta = dict(METADATA_TEMPLATE)
    _write(folder, "run4_twotone_trial1", lambda t: 2.0 * np.sin(1.5 * t) + 1.2 * np.cos(3.0 * t), 8.0, [0.0, -0.5, np.pi - 0.4, 0.0], meta)
    good = check_signs(load_run(folder / "run4_twotone_trial1.csv", meta))
    assert good["g_u"] > 0.0 and good["a"] > 0.0
    bad_angle = check_signs(load_run(folder / "run4_twotone_trial1.csv", {**meta, "angle_sign": -1}))
    assert bad_angle["a"] < 0.0
    bad_cart = check_signs(load_run(folder / "run4_twotone_trial1.csv", {**meta, "cart_sign": -1}))
    assert bad_cart["g_u"] < 0.0


def test_rest_and_deadzone_statistics(folder: Path) -> None:
    meta = dict(METADATA_TEMPLATE)
    _write(folder, "run1_rest_trial1", lambda t: 0.0, 3.0, [0.0, 0.0, np.pi, 0.0], meta)
    stats = rest_statistics(load_run(folder / "run1_rest_trial1.csv", meta))
    assert stats["theta resolution [rad]"] == pytest.approx(2.0 * np.pi / 4096.0, rel=1e-6)
    assert 0.0 < stats["theta std [rad]"] < np.deg2rad(0.5)
    _write(folder, "run3_deadzone_trial1", lambda t: 3.0 * np.sin(2.0 * np.pi * t / 6.0), 6.0, [0.0, 0.0, np.pi, 0.0], meta)
    thr = deadzone_thresholds(load_run(folder / "run3_deadzone_trial1.csv", meta))
    assert set(thr) == {"breakaway positive [N]", "breakaway negative [N]"}


def test_replay_job_scores_velocity_parameters_and_held_out_prediction(folder: Path) -> None:
    meta = dict(METADATA_TEMPLATE)
    _write(folder, "run4_twotone_trial1", lambda t: 2.0 * np.sin(1.5 * t) + 1.2 * np.cos(3.0 * t), 6.0, [0.0, -0.5, np.pi - 0.4, 0.0], meta)
    _write(folder, "run6_heldout_pulse", lambda t: 1.0 if 0.2 <= t % 2.0 < 0.5 else 0.0, 3.0, [0.0, 0.0, np.pi, 0.0], meta)
    rows = job((str(folder), "run4_twotone_trial1", "E", 0))
    kinds = [r["kind"] for r in rows]
    assert kinds == ["velocity", "parameters", "prediction"]
    assert rows[0]["diverged"] == 0.0 and np.isfinite(rows[0]["th_dot RMSE vs ref, second half"])
    assert rows[1]["M+m datasheet"] == pytest.approx(2.63)
    assert rows[2]["test"] == "run6_heldout_pulse" and np.isfinite(rows[2]["0.5 s NMSE theta"])
