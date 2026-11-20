"""Real compiled-scene checks for the optional analytical wind backend."""

import copy
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from wind_field_simulation import simulation
from wind_field_simulation.training.reference_aerodynamics import (
    paired_wind_labels,
)
from wind_field_simulation.training.xml_parameters import (
    DEFAULT_ASSUMPTIONS,
    DEFAULT_SCENE,
    content_sha256,
    export_reference_bundle,
)
from wind_field_simulation.wind_backends import AnalyticalWindResidualModel
from test_simulation import RecordingBridge, RecordingPinnPredictor, parse_nominal_wind_config


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    """Build source-derived artifacts without relying on ignored outputs."""
    export_reference_bundle(DEFAULT_SCENE, DEFAULT_ASSUMPTIONS, tmp_path)
    return tmp_path / "reference_config.json"


def make_predictor(path: Path) -> AnalyticalWindResidualModel:
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    return AnalyticalWindResidualModel(path, DEFAULT_SCENE, model)


def test_predictor_matches_independent_paired_force(bundle: Path) -> None:
    predictor = make_predictor(bundle)
    state = dict(velocity_world=[-3, 0, 0.2],
                 quaternion_wxyz=[1, 0, 0, 0],
                 angular_velocity_body=[0.1, -0.2, 0.3],
                 wind_world=[3, 0, 0])
    reference = json.loads(bundle.read_text())
    independent = paired_wind_labels(**state, reference=reference)
    # Raw full difference retained; the app performs the shared XY projection.
    expected = independent["delta_force_world_n"] / 8.0004
    np.testing.assert_allclose(predictor.predict(**state), expected)
    state["wind_world"] = [0, 0, 0]
    np.testing.assert_array_equal(predictor.predict(**state), np.zeros(3))


def test_analytic_drag_randomization_preserves_nominal_calibration() -> None:
    """Randomized coefficients are fixed for a run and reuse the nominal gain."""
    config = simulation.parse_config([
        "--wind-backend", "analytic", "--drag-variation", "0.2", "--wind-seed", "42",
    ])
    app = simulation.WindSimulationApp(config, bridge=RecordingBridge())
    replay = simulation.WindSimulationApp(config, bridge=RecordingBridge())
    try:
        assert app.wind_cda_m2 == 0.25
        assert app.wind_max_horizontal_m_s == 10
        wind = np.array([4., 4., 0.])
        direction = wind / np.linalg.norm(wind)
        predictor = app.pinn_predictor
        assert app.wind_drag_scale * simulation.reference_horizontal_drag_area_m2(
            predictor.reference,
        ) == pytest.approx(0.25)
        assert predictor.drag_group_scales == replay.pinn_predictor.drag_group_scales
        assert set(predictor.drag_group_scales) == {"frame", "booms_and_motors", "arm"}
        assert all(0.8 <= scale <= 1.2 for scale in predictor.drag_group_scales.values())
        state = ([0, 0, 0], [1, 0, 0, 0], [0, 0, 0])
        force = predictor.predict(*state, wind) * app.bindings.uav_mass_kg * app.wind_drag_scale
        assert force @ direction != pytest.approx(4.9)
        np.testing.assert_array_equal(
            predictor.predict(*state, wind), replay.pinn_predictor.predict(*state, wind),
        )
        np.testing.assert_array_equal(predictor.predict(*state, [0, 0, 0]), [0, 0, 0])
    finally:
        app._close_runtime()
        replay._close_runtime()


def test_analytic_gust_changes_applied_force_and_respects_wind_gate(tmp_path: Path) -> None:
    """Exercise the normal Apply/On/Off path with a seeded pulse."""
    log = tmp_path / "gust.jsonl"
    config = simulation.parse_config([
        "--wind-backend", "analytic", "--gust-amplitude", "1.5", "--gust-period", "4",
        "--wind-seed", "42", "--run-log", str(log),
    ])
    app = simulation.WindSimulationApp(config, bridge=RecordingBridge())
    try:
        base = np.array([4., 4., 0.])
        app._process_ui_command(
            simulation.UiCommand(simulation.UiCommandType.SET_WIND, base),
        )
        np.testing.assert_array_equal(app._apply_wind_force()[2], [0, 0, 0])
        app._process_ui_command(simulation.UiCommand(simulation.UiCommandType.TOGGLE_WIND))
        app.data.time = 8.
        at_start, _, nominal_force = app._apply_wind_force()
        np.testing.assert_array_equal(at_start, base)
        app.residual_filter.reset()
        app.data.time = 10.
        peak, _, gust_force = app._apply_wind_force()
        assert np.linalg.norm(peak - base) == pytest.approx(1.5)
        replay = simulation.HorizontalGust(1.5, 4, 42)
        np.testing.assert_array_equal(peak, replay.velocity(base, 2.0, 10.0))
        different_seed = simulation.HorizontalGust(1.5, 4, 43)
        assert not np.allclose(peak, different_seed.velocity(base, 2.0, 10.0))
        assert peak[2] == gust_force[2] == 0
        assert not np.allclose(gust_force, nominal_force)
        app.data.time = 12.
        np.testing.assert_array_equal(app._apply_wind_force()[0], base)
        next_gust = replay.velocity(np.zeros(3), 6.0, 10.0)
        near_limit = 9.9 * next_gust / np.linalg.norm(next_gust)
        app._process_ui_command(
            simulation.UiCommand(simulation.UiCommandType.SET_WIND, near_limit),
        )
        app.data.time = 14.
        bounded_wind = app._apply_wind_force()[0]
        assert np.linalg.norm(bounded_wind) <= 10.0
        assert np.linalg.norm(bounded_wind) == pytest.approx(10.0)
        app._process_ui_command(simulation.UiCommand(simulation.UiCommandType.TOGGLE_WIND))
        active, _, force = app._apply_wind_force()
        np.testing.assert_array_equal(active, [0, 0, 0])
        np.testing.assert_array_equal(force, [0, 0, 0])
    finally:
        app._close_runtime()
    metadata = json.loads(log.read_text().splitlines()[0])
    assert metadata["wind_seed"] == 42
    assert metadata["gust_amplitude_m_s"] == 1.5
    assert metadata["gust_period_s"] == 4.0


@pytest.mark.parametrize("flags", [
    ["--wind-backend", "pinn", "--gust-amplitude", "1"],
    ["--wind-backend", "analytic", "--drag-variation", "1"],
    ["--wind-backend", "analytic", "--gust-amplitude", "nan"],
    ["--wind-backend", "analytic", "--gust-amplitude", "1", "--gust-period", "0"],
    ["--wind-backend", "analytic", "--drag-variation", "0.2", "--wind-seed", "-1"],
])
def test_invalid_analytic_perturbation_settings_are_rejected(flags: list[str]) -> None:
    with pytest.raises(ValueError):
        simulation.parse_config(flags)


@pytest.mark.parametrize("wind", [[3, 3, 0], [0, 0, 0.1], [3.001, 0, 0]])
def test_predictor_rejects_invalid_wind(bundle: Path, wind: list[float]) -> None:
    with pytest.raises(ValueError):
        make_predictor(bundle).predict([0, 0, 0], [1, 0, 0, 0],
                                       [0, 0, 0], wind)


@pytest.mark.parametrize("artifact", ["reference_config", "vehicle_parameters",
                                      "geometry_manifest"])
def test_corrupted_artifacts_fail_closed(bundle: Path, artifact: str) -> None:
    path = bundle.parent / f"{artifact}.json"
    payload = json.loads(path.read_text())
    payload["unexpected_change"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fingerprint"):
        make_predictor(bundle)


def test_self_consistent_reference_cannot_disagree_with_vehicle(bundle: Path) -> None:
    reference = json.loads(bundle.read_text())
    reference["mass_total_kg"] = 6.5
    reference.pop("reference_sha256")
    reference["reference_sha256"] = content_sha256(reference)
    bundle.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="mass|vehicle"):
        make_predictor(bundle)


def test_different_scene_and_modified_runtime_model_are_rejected(
    bundle: Path, tmp_path: Path,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    with pytest.raises(ValueError, match="XML|scene"):
        AnalyticalWindResidualModel(bundle, tmp_path / "other.xml", model)
    model.body_mass[model.body("quadrotor_950").id] += 1
    with pytest.raises(ValueError, match="mass|model"):
        AnalyticalWindResidualModel(bundle, DEFAULT_SCENE, model)


def test_invalid_reference_geometry_or_schema_is_rejected(bundle: Path) -> None:
    reference = json.loads(bundle.read_text())
    reference["proxies"] = copy.deepcopy(reference["proxies"])
    reference["proxies"][0]["areas_m2"][0] *= 2
    reference.pop("reference_sha256")
    reference["reference_sha256"] = content_sha256(reference)
    bundle.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="geometry"):
        make_predictor(bundle)


def test_cli_defaults_and_explicit_selection(bundle: Path) -> None:
    assert simulation.parse_config([]).wind_backend == "pinn-x4"
    config = simulation.parse_config([
        "--wind-backend", "analytic", "--reference-config", str(bundle),
    ])
    assert config.wind_backend == "analytic"
    assert config.reference_config == bundle
    default_analytic = simulation.parse_config(["--wind-backend", "analytic"])
    assert default_analytic.reference_config == simulation.default_wind_reference_config()
    with pytest.raises(ValueError, match="analytic"):
        simulation.parse_config(["--wind-backend", "pinn", "--reference-config", str(bundle)])


@pytest.mark.parametrize("wind", [[3, 3, 0], [3.01, 0, 0]])
def test_shared_application_wind_domain(wind: list[float]) -> None:
    with pytest.raises(ValueError, match="3 m/s"):
        simulation.ConstantWindField(wind)
    with pytest.raises(ValueError, match="3 m/s"):
        simulation.parse_config([
            "--wind-backend", "analytic", "--wind-x", str(wind[0]), "--wind-y", str(wind[1]),
        ])


def analytic_app(bundle: Path, *extra: str) -> simulation.WindSimulationApp:
    """Construct the real app without sockets, viewers or old weights."""
    return simulation.WindSimulationApp(
        simulation.parse_config([
            "--no-viewer", "--wind-backend", "analytic",
            "--reference-config", str(bundle), *extra,
        ]), bridge=RecordingBridge(),
    )


def test_analytic_selection_does_not_construct_old_pinn(
    bundle: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden() -> None:
        raise AssertionError("Analytic selection loaded old weights")

    monkeypatch.setattr(simulation, "PinnWindResidualModel", forbidden)
    app = analytic_app(bundle)
    assert isinstance(app.pinn_predictor, AnalyticalWindResidualModel)
    assert app._panel_snapshot(app._current_state()[0]).wind_backend == "analytic"
    app._process_ui_command(simulation.UiCommand(
        simulation.UiCommandType.TOGGLE_WIND,
    ))
    assert not app.flight_control.wind_enabled
    np.testing.assert_array_equal(app._apply_wind_force()[2], [0, 0, 0])


def test_explicit_pinn_selection_uses_old_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = RecordingPinnPredictor()
    monkeypatch.setattr(simulation, "PinnWindResidualModel", lambda: predictor)
    app = simulation.WindSimulationApp(
        simulation.parse_config(["--wind-backend", "pinn", "--no-viewer"]),
        bridge=RecordingBridge(),
    )
    assert app.pinn_predictor is predictor


def test_actual_analytic_force_pipeline_and_frozen_geometry(bundle: Path) -> None:
    app = analytic_app(bundle)
    initial_pose = app.data.qpos.copy()
    initial_control = app.data.ctrl.copy()
    app._process_ui_command(simulation.UiCommand(
        simulation.UiCommandType.SET_WIND, (3, 0, 0),
    ))
    assert not app.flight_control.wind_enabled
    np.testing.assert_array_equal(app.data.qpos, initial_pose)
    np.testing.assert_array_equal(app.data.ctrl, initial_control)
    app.flight_control.wind_enabled = True
    app.residual_filter.filtered[:] = [0.2, -0.1, 9]
    app.data.xfrc_applied[app.bindings.quadrotor_body_id] = 99
    raw = app.pinn_predictor.predict([0, 0, 0], [1, 0, 0, 0],
                                      [0, 0, 0], [3, 0, 0])
    _, _, force = app._apply_wind_force()
    expected = 0.95 * np.array([0.2, -0.1, 0]) + 0.05 * raw
    expected[2] = 0
    np.testing.assert_allclose(force, 8.0004 * expected)
    assert app.last_raw_residual_acceleration[2] != 0
    np.testing.assert_array_equal(
        app.data.xfrc_applied[app.bindings.quadrotor_body_id, 2:], [0, 0, 0, 0],
    )
    filter_before = app.residual_filter.filtered.copy()
    target_before = app.flight_control.target_position_world.copy()
    app._process_ui_command(simulation.UiCommand(
        simulation.UiCommandType.SET_WIND, (3, 3, 0),
    ))
    np.testing.assert_array_equal(app.wind_field.velocity(True), [3, 0, 0])
    np.testing.assert_array_equal(app.residual_filter.filtered, filter_before)
    np.testing.assert_array_equal(app.flight_control.target_position_world,
                                  target_before)
    app._process_ui_command(simulation.UiCommand(
        simulation.UiCommandType.SET_WIND, (0, 3, 0),
    ))
    np.testing.assert_array_equal(app.residual_filter.filtered, filter_before)
    # Changing actual arm state does not change the frozen reference geometry.
    before = app.pinn_predictor.predict([0, 0, 0], [1, 0, 0, 0],
                                       [0, 0, 0], [0, 3, 0])
    app.data.qpos[app.model.jnt_qposadr[app.model.joint("joint2").id]] = 0.1
    mujoco.mj_forward(app.model, app.data)
    np.testing.assert_array_equal(app.pinn_predictor.predict(
        [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], [0, 3, 0]), before)
    assert app._arm_joint_deviations()["joint2"] == pytest.approx(0.1)
    app.flight_control.wind_enabled = False
    np.testing.assert_array_equal(app._apply_wind_force()[2], [0, 0, 0])
    np.testing.assert_array_equal(app.residual_filter.filtered, [0, 0, 0])


def test_analytic_backend_uses_existing_acceleration_clip(bundle: Path) -> None:
    app = analytic_app(bundle)
    app._process_ui_command(simulation.UiCommand(
        simulation.UiCommandType.SET_WIND, (3, 0, 0),
    ))
    app.flight_control.wind_enabled = True
    app.residual_filter.ema_alpha = 1
    # Out-of-inspection-domain state deliberately exercises the unchanged clip.
    app.data.qvel[app.bindings.main_dof_adr] = -1000
    mujoco.mj_forward(app.model, app.data)
    force = app._apply_wind_force()[2]
    assert app.last_raw_residual_acceleration[0] > 6
    assert force[0] == pytest.approx(8.0004 * 6)
    assert force[2] == 0


def test_panel_rejects_resultant_wind_above_three() -> None:
    import queue

    commands: queue.SimpleQueue = queue.SimpleQueue()
    panel = simulation.ControlPanel(commands)
    try:
        for box in panel.wind_inputs:
            box.set_val("3")
        panel._on_apply_wind(None)
        assert commands.empty()
        assert "3 m/s" in panel.wind_feedback_text.get_text()
    finally:
        panel.close()


def test_panel_displays_selected_backend_and_simulation_clock(bundle: Path) -> None:
    app = analytic_app(bundle)
    panel = simulation.ControlPanel(app.command_queue)
    try:
        app.data.time = 12.5
        panel.update(app._panel_snapshot(app._current_state()[0]))
        assert "analytic" in panel.status_text.get_text()
        assert "12.5 s" in panel.status_text.get_text()
    finally:
        panel.close()


def test_optional_run_log_records_synchronized_force_and_joint_deviation(
    bundle: Path, tmp_path: Path,
) -> None:
    log = tmp_path / "analytic.jsonl"
    app = analytic_app(bundle, "--run-log", str(log))
    app._process_ui_command(simulation.UiCommand(
        simulation.UiCommandType.SET_WIND, (3, 0, 0),
    ))
    app.flight_control.wind_enabled = True
    address = app.model.jnt_qposadr[app.model.joint("endleft").id]
    app.data.qpos[address] += 0.001
    mujoco.mj_forward(app.model, app.data)
    for _ in range(5):
        assert app._run_iteration()
    app._close_runtime()
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows[0]["type"] == "metadata"
    assert rows[0]["wind_backend"] == "analytic"
    assert rows[0]["reference_sha256"]
    samples = [row for row in rows if row["type"] == "sample"]
    assert [row["sim_time_s"] for row in samples] == pytest.approx([0, 0.02])
    assert samples[0]["arm_joint_deviation"]["endleft"] == pytest.approx(0.001)
    assert samples[0]["force_world_n"][2] == 0
    np.testing.assert_allclose(samples[0]["force_world_n"],
        np.array(samples[0]["filtered_acceleration_world_m_s2"]) * 8.0004)
    assert samples[0]["wind_compute_wall_s"] > 0
    assert samples[0]["iteration_wall_s"] >= samples[0]["wind_compute_wall_s"]
    assert rows[-1]["type"] == "closed"
    assert rows[-1]["sample_count"] == 2
    with pytest.raises(FileExistsError):
        analytic_app(bundle, "--run-log", str(log))


def test_run_log_marks_cleanup_transport_failure(
    bundle: Path, tmp_path: Path,
) -> None:
    log = tmp_path / "failed.jsonl"
    app = analytic_app(bundle, "--run-log", str(log))

    def broken_disarm() -> None:
        raise OSError("cleanup transport failed")

    app.bridge.request_disarm = broken_disarm
    with pytest.raises(OSError, match="cleanup transport"):
        app._close_runtime()
    footer = json.loads(log.read_text().splitlines()[-1])
    assert "cleanup transport" in footer["exception"]


def test_missing_reference_never_falls_back_or_constructs_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Invalid reference touched another backend/bridge")

    monkeypatch.setattr(simulation, "PinnWindResidualModel", forbidden)
    monkeypatch.setattr(simulation, "Px4Bridge", forbidden)
    with pytest.raises(FileNotFoundError):
        simulation.WindSimulationApp(simulation.parse_config([
            "--wind-backend", "analytic", "--reference-config",
            str(tmp_path / "missing.json"),
        ]))


def test_log_reads_current_free_joint_state_after_mujoco_step(
    bundle: Path, tmp_path: Path,
) -> None:
    """MuJoCo sensor/kinematic caches can lag integrated qpos/qvel by a step."""
    app = analytic_app(bundle, "--run-log", str(tmp_path / "sync.jsonl"))
    address = app.bindings.main_qpos_adr
    dof = app.bindings.main_dof_adr
    app.data.qpos[address + 2] = 5
    app.data.qvel[dof:dof + 3] = [1, 2, 0]
    app.data.qvel[dof + 3:dof + 6] = [0.2, -0.3, 0.4]
    mujoco.mj_forward(app.model, app.data)
    mujoco.mj_step(app.model, app.data)
    assert not np.allclose(app.data.xpos[app.bindings.quadrotor_body_id],
                            app.data.qpos[address:address + 3])
    sample = app._run_log_sample()
    np.testing.assert_array_equal(sample["position_world_m"],
                                   app.data.qpos[address:address + 3])
    np.testing.assert_array_equal(sample["angular_velocity_body_rad_s"],
                                   app.data.qvel[dof + 3:dof + 6])
    app._close_runtime()


def test_both_wind_predictors_receive_current_root_angular_rate() -> None:
    predictor = RecordingPinnPredictor()
    app = simulation.WindSimulationApp(
        parse_nominal_wind_config(["--no-viewer"]),
        bridge=RecordingBridge(), pinn_predictor=predictor,
    )
    # Simulate integrated state advancing while sensor caches remain stale.
    dof = app.bindings.main_dof_adr
    app.data.qvel[dof + 3:dof + 6] = [0.1, 0.2, 0.3]
    app._process_ui_command(simulation.UiCommand(
        simulation.UiCommandType.SET_WIND, (1, 0, 0),
    ))
    app.flight_control.wind_enabled = True
    app._apply_wind_force()
    np.testing.assert_array_equal(predictor.calls[-1]["angular_velocity_body"],
                                   [0.1, 0.2, 0.3])
