"""Runtime contracts for the separately selected X4 pilot, no PX4 sockets."""

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest
import torch

from test_simulation import RecordingBridge
from wind_field_simulation import simulation
from wind_field_simulation.training.model_v1 import KIND, X4WindAeroV1
from wind_field_simulation.training.xml_parameters import (
    DEFAULT_ASSUMPTIONS,
    DEFAULT_SCENE,
    content_sha256,
    export_reference_bundle,
)
from wind_field_simulation.wind_backends import X4PinnWindResidualModel


@pytest.fixture()
def candidate(tmp_path: Path) -> tuple[Path, Path]:
    """Use a real 6D prior network and source-derived reference, no ignored data."""
    reference = export_reference_bundle(DEFAULT_SCENE, DEFAULT_ASSUMPTIONS, tmp_path)
    vehicle = json.loads((tmp_path / "vehicle_parameters.json").read_text())
    torch.manual_seed(42)
    network = X4WindAeroV1(np.zeros(6), np.ones(6), np.full(3, .05),
                          np.array([.01, .02, .03]))
    checkpoint = tmp_path / "candidate.pt"
    torch.save({
        "kind": KIND, "feature_version": "body_air_velocity_omega_6d_v1",
        "state_dict": network.state_dict(), "normalization": {
            "input_mean": [0.] * 6, "input_std": [1.] * 6,
            "full_scale": [.05] * 3, "delta_scale": [.05] * 3,
        }, "prior_coefficients": [.01, .02, .03],
        "reference_sha256": reference["reference_sha256"],
        "reference_config": reference, "xml_mesh_dependencies": vehicle["dependencies"],
        "mass_total_kg": reference["mass_total_kg"], "arm_pose": reference["initial_joints"],
        "coordinates_units": "B FLU at root O; W ENU-like; WXYZ B→W; m/s, rad/s, m/s²",
        "wind_domain": "horizontal resultant 0–3 m/s; fixed initial arm geometry",
        "dataset_manifest_sha256": "a" * 64, "dataset_archive_sha256": "b" * 64,
    }, checkpoint)
    return checkpoint, tmp_path / "reference_config.json"


def make_app(candidate: tuple[Path, Path], *extra: str) -> simulation.WindSimulationApp:
    """Construct selected backend through the production config/app path."""
    checkpoint, reference = candidate
    return simulation.WindSimulationApp(simulation.parse_config([
        "--wind-backend", "pinn-x4", "--pinn-checkpoint", str(checkpoint),
        "--reference-config", str(reference), "--no-viewer", *extra,
    ]), bridge=RecordingBridge())


def test_real_core_pair_full_z_and_zero_wind(candidate: tuple[Path, Path]) -> None:
    checkpoint, reference = candidate
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    predictor = X4PinnWindResidualModel(checkpoint, reference, DEFAULT_SCENE, model)
    quaternion = np.array([np.cos(.1), 0, np.sin(.1), 0])
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quaternion)
    rotation = rotation.reshape(3, 3)
    velocity, wind = np.array([-.5, .2, .1]), np.array([3., 0, 0])
    pair = np.stack(((velocity-wind) @ rotation, velocity @ rotation))
    acceleration = -np.array([.01, .02, .03]) * pair * np.abs(pair)
    expected = rotation @ (acceleration[0] - acceleration[1])
    actual = predictor.predict(velocity, quaternion, [0, 0, 0], wind)
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    assert actual[2] != 0
    np.testing.assert_array_equal(predictor.predict(
        velocity, quaternion, [0, 0, 0], [0, 0, 0]), np.zeros(3))
    for bad_wind in ([3, 3, 0], [0, 0, 1], [float("nan"), 0, 0]):
        with pytest.raises(ValueError):
            predictor.predict(velocity, quaternion, [0, 0, 0], bad_wind)
    with pytest.raises(ValueError, match="shape"):
        predictor.predict([velocity], quaternion, [0, 0, 0], wind)
    with pytest.raises(ValueError, match="unit"):
        predictor.predict(velocity, [1.000001, 0, 0, 0], [0, 0, 0], wind)


@pytest.mark.parametrize("field,value", [
    ("mass_total_kg", 6.5), ("reference_sha256", "f" * 64),
    ("coordinates_units", "NED FRD"), ("arm_pose", {}),
    ("xml_mesh_dependencies", {}), ("wind_domain", "unbounded"),
])
def test_inconsistent_metadata_fails_before_bridge(
    candidate: tuple[Path, Path], field: str, value: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, reference = candidate
    saved = torch.load(checkpoint, weights_only=True)
    saved[field] = value
    torch.save(saved, checkpoint)
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Invalid candidate connected bridge or loaded legacy")
    monkeypatch.setattr(simulation, "Px4Bridge", forbidden)
    monkeypatch.setattr(simulation, "PinnWindResidualModel", forbidden)
    with pytest.raises(ValueError):
        simulation.WindSimulationApp(simulation.parse_config([
            "--wind-backend", "pinn-x4", "--pinn-checkpoint", str(checkpoint),
            "--reference-config", str(reference), "--no-viewer",
        ]))


def test_cli_default_x4_and_backend_specific_paths(
    candidate: tuple[Path, Path],
) -> None:
    checkpoint, reference = candidate
    default = simulation.parse_config([])
    assert default.wind_backend == "pinn-x4"
    assert default.pinn_checkpoint == simulation.default_x4_pinn_checkpoint()
    assert default.reference_config == default.pinn_checkpoint.with_name("reference_config.json")
    assert default.pinn_checkpoint.is_relative_to(simulation.default_pinn_checkpoint_dir())
    assert default.pinn_checkpoint.is_file()
    legacy = simulation.parse_config(["--wind-backend", "pinn"])
    assert legacy.pinn_checkpoint is None and legacy.reference_config is None
    analytic = simulation.parse_config(["--wind-backend", "analytic"])
    assert analytic.pinn_checkpoint is None
    assert analytic.reference_config == simulation.default_wind_reference_config()
    assert analytic.reference_config != default.reference_config
    selected = simulation.parse_config(["--pinn-checkpoint", str(checkpoint),
                                         "--reference-config", str(reference)])
    assert selected.pinn_checkpoint == checkpoint
    assert selected.reference_config == reference
    with pytest.raises(ValueError, match="pinn-x4"):
        simulation.parse_config(["--wind-backend", "pinn", "--pinn-checkpoint", str(checkpoint)])
    with pytest.raises(ValueError, match="pinn-x4"):
        simulation.parse_config(["--wind-backend", "analytic", "--reference-config",
                                 str(reference), "--pinn-checkpoint", str(checkpoint)])


def test_no_argument_app_loads_bundled_x4_and_preserves_apply_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden() -> None:
        raise AssertionError("Default startup loaded legacy model")
    monkeypatch.setattr(simulation, "PinnWindResidualModel", forbidden)
    app = simulation.WindSimulationApp(simulation.parse_config([]),
                                       bridge=RecordingBridge())
    try:
        assert isinstance(app.pinn_predictor, X4PinnWindResidualModel)
        assert app.pinn_predictor.checkpoint_sha256 == (
            "43c44e6b7ea509d0e54206aad8678cd8b5384aa69171ca77f9b89e278b8f89ba")
        assert app.wind_max_horizontal_m_s == 10.0
        assert app.wind_field is None
        app._process_ui_command(simulation.UiCommand(simulation.UiCommandType.TOGGLE_WIND))
        assert not app.flight_control.wind_enabled
        np.testing.assert_array_equal(app._apply_wind_force()[2], [0, 0, 0])
        app._process_ui_command(simulation.UiCommand(simulation.UiCommandType.SET_WIND, (6, 8, 0)))
        assert app.wind_field is not None
        np.testing.assert_array_equal(app.wind_field.configured_velocity_world, [6, 8, 0])
        assert not app.flight_control.wind_enabled
        app._process_ui_command(simulation.UiCommand(simulation.UiCommandType.SET_WIND, (10, 10, 0)))
        np.testing.assert_array_equal(app.wind_field.configured_velocity_world, [6, 8, 0])
    finally:
        app._close_runtime()


def test_same_app_xy_ema_clip_and_log_metadata(
    candidate: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden() -> None:
        raise AssertionError("New selection loaded legacy weights")
    monkeypatch.setattr(simulation, "PinnWindResidualModel", forbidden)
    app = make_app(candidate, "--run-log", str(tmp_path / "run.jsonl"))
    try:
        assert isinstance(app.pinn_predictor, X4PinnWindResidualModel)
        app._process_ui_command(simulation.UiCommand(simulation.UiCommandType.TOGGLE_WIND))
        assert not app.flight_control.wind_enabled
        app._process_ui_command(simulation.UiCommand(simulation.UiCommandType.SET_WIND,
                                                     (3, 0, 0)))
        app.flight_control.wind_enabled = True
        address = app.bindings.main_qpos_adr
        app.data.qpos[address+3:address+7] = [np.cos(.1), 0, np.sin(.1), 0]
        mujoco.mj_forward(app.model, app.data)
        app.residual_filter.filtered[:] = [.2, -.1, 9]
        app.data.xfrc_applied[:] = 99
        app._apply_wind_force()
        raw = app.last_raw_residual_acceleration.copy()
        assert raw[2] != 0
        expected = .95 * np.array([.2, -.1, 0]) + .05 * raw
        expected[2] = 0
        np.testing.assert_allclose(app.last_pinn_force, app.bindings.uav_mass_kg * expected)
        np.testing.assert_array_equal(app.data.xfrc_applied[app.bindings.quadrotor_body_id, 2:], 0)
        before = app.residual_filter.filtered.copy()
        app._process_ui_command(simulation.UiCommand(simulation.UiCommandType.SET_WIND,
                                                     (3, 3, 0)))
        np.testing.assert_array_equal(app.residual_filter.filtered, before)
        app.residual_filter.filtered[:] = [100, -100, 9]
        app._apply_wind_force()
        np.testing.assert_allclose(app.last_pinn_force[:2], app.bindings.uav_mass_kg * np.array([6, -6]))
        assert app.last_pinn_force[2] == 0
        app.flight_control.wind_enabled = False
        np.testing.assert_array_equal(app._apply_wind_force()[2], np.zeros(3))
    finally:
        app._close_runtime()
    metadata = json.loads((tmp_path / "run.jsonl").read_text().splitlines()[0])
    assert metadata["wind_backend"] == "pinn-x4"
    assert metadata["checkpoint_sha256"] == app.pinn_predictor.checkpoint_sha256
    assert metadata["dataset_manifest_sha256"] == "a" * 64
    assert metadata["inference_device"] == "cpu"
    assert metadata["torch_threads"] == 1


@pytest.fixture()
def wide_candidate(
    candidate: tuple[Path, Path], tmp_path: Path,
) -> tuple[Path, Path]:
    """Export a real wide reference and matching test-only checkpoint."""
    assumptions = json.loads(DEFAULT_ASSUMPTIONS.read_text())
    assumptions["assumptions"]["wind_max_horizontal_m_s"] = 10.0
    assumptions_path = tmp_path / "wide_assumptions.json"
    assumptions_path.write_text(json.dumps(assumptions))
    wide_dir = tmp_path / "wide"
    reference = export_reference_bundle(DEFAULT_SCENE, assumptions_path, wide_dir)
    saved = torch.load(candidate[0], weights_only=True)
    saved.update(kind="x4_950_wind_increment_xy_v2", reference_config=reference,
                 reference_sha256=reference["reference_sha256"],
                 state_domain=reference["inspection_state_domain"],
                 wind_max_horizontal_m_s=10.0,
                 wind_domain="horizontal resultant 0–10 m/s; fixed initial arm geometry")
    checkpoint = wide_dir / "candidate.pt"
    torch.save(saved, checkpoint)
    return checkpoint, wide_dir / "reference_config.json"


@pytest.mark.parametrize("backend", ["pinn-x4", "analytic"])
def test_wide_cli_panel_and_force_share_selected_round_domain(
    wide_candidate: tuple[Path, Path], backend: str, tmp_path: Path,
) -> None:
    """Both real paths accept 6-8-10, reject invalid Apply and apply XY force."""
    checkpoint, reference = wide_candidate
    log = tmp_path / "wide.jsonl"
    flags = ["--wind-backend", backend, "--reference-config", str(reference),
             "--wind-x", "6", "--wind-y", "8", "--no-viewer", "--run-log", str(log)]
    if backend == "pinn-x4":
        flags += ["--pinn-checkpoint", str(checkpoint)]
    config = simulation.parse_config(flags)
    app = simulation.WindSimulationApp(config, bridge=RecordingBridge())
    panel = simulation.ControlPanel(
        app.command_queue, config.wind_velocity_world,
        wind_max_horizontal_m_s=app.wind_max_horizontal_m_s,
    )
    try:
        assert app.wind_max_horizontal_m_s == 10
        assert app.wind_field is None
        panel._on_apply_wind(None)
        app._process_ui_command(app.command_queue.get_nowait())
        np.testing.assert_array_equal(app.wind_field.configured_velocity_world, [6, 8, 0])
        assert not app.flight_control.wind_enabled
        app._process_ui_command(simulation.UiCommand(simulation.UiCommandType.TOGGLE_WIND))
        app._apply_wind_force()
        assert np.linalg.norm(app.last_pinn_force[:2]) > 0
        assert app.last_pinn_force[2] == 0
        assert app.data.xfrc_applied[app.bindings.quadrotor_body_id, 2] == 0
        for invalid in ((10, 10, 0), (10.01, 0, 0), (1, 2, 0.1)):
            app._process_ui_command(simulation.UiCommand(simulation.UiCommandType.SET_WIND, invalid))
            np.testing.assert_array_equal(app.wind_field.configured_velocity_world, [6, 8, 0])
            with pytest.raises(ValueError):
                app.pinn_predictor.predict([0, 0, 0], [1, 0, 0, 0], [0, 0, 0], invalid)
        panel.wind_inputs[0].set_val("10")
        panel.wind_inputs[1].set_val("10")
        panel._on_apply_wind(None)
        assert app.command_queue.empty()
        assert "10 m/s" in panel.wind_feedback_text.get_text()
    finally:
        panel.close()
        app._close_runtime()
    metadata = json.loads(log.read_text().splitlines()[0])
    assert metadata["wind_max_m_s"] == 10
    assert metadata["inspection_state_domain"]["root_horizontal_speed_max_m_s"] == 3


@pytest.mark.parametrize("backend", ["pinn-x4", "analytic"])
def test_bundled_wide_drag_area_sets_applied_force(
    backend: str, tmp_path: Path,
) -> None:
    """Check the force applied by both bundled wide backends at the anchor."""
    log = tmp_path / "run.jsonl"
    flags = ["--no-viewer", "--run-log", str(log)]
    if backend == "analytic":
        reference = simulation.default_x4_pinn_checkpoint().with_name(
            "reference_config.json"
        )
        flags += ["--wind-backend", backend, "--reference-config", str(reference)]
    app = simulation.WindSimulationApp(
        simulation.parse_config(flags), bridge=RecordingBridge(),
    )
    wind = np.array([4.0, 4.0, 0.0])
    try:
        address = app.bindings.main_qpos_adr
        app.data.qpos[address + 3:address + 7] = [1, 0, 0, 0]
        app.data.qvel[:] = 0
        mujoco.mj_forward(app.model, app.data)
        app._process_ui_command(simulation.UiCommand(
            simulation.UiCommandType.SET_WIND, tuple(wind),
        ))
        app._process_ui_command(simulation.UiCommand(
            simulation.UiCommandType.TOGGLE_WIND,
        ))
        _, _, force = app._apply_wind_force()
        # Initial EMA step from zero, with no acceleration clipping.
        along_wind = force @ (wind / np.linalg.norm(wind))
        assert along_wind / 0.05 == pytest.approx(
            0.5 * 1.225 * (wind @ wind) * 0.25,
            rel=0.01 if backend == "pinn-x4" else 1e-12,
        )
        assert force[2] == 0.0
    finally:
        app._close_runtime()
    metadata = json.loads(log.read_text().splitlines()[0])
    assert metadata["wind_cda_m2"] == 0.25
    assert metadata["wind_drag_scale"] > 1.0


def test_old_candidate_cannot_advertise_wide_domain(
    candidate: tuple[Path, Path], wide_candidate: tuple[Path, Path],
) -> None:
    """A wider reference alone cannot extend old PINN or legacy runtime."""
    with pytest.raises(ValueError, match="wind|reference"):
        make_app((candidate[0], wide_candidate[1]), "--wind-x", "10")
    with pytest.raises(ValueError, match="wind|reference"):
        make_app((wide_candidate[0], candidate[1]))
    for backend in ("pinn", "analytic"):
        with pytest.raises(ValueError, match="3 m/s"):
            simulation.parse_config(["--wind-backend", backend, "--wind-x", "10"])
    config = simulation.parse_config(["--wind-x", "10"])
    np.testing.assert_array_equal(config.wind_velocity_world, [10, 0, 0])


@pytest.mark.parametrize("bad_cap", [None, "10", 3, 100, float("nan")])
def test_wide_checkpoint_requires_explicit_numeric_domain(
    wide_candidate: tuple[Path, Path], bad_cap: Any,
) -> None:
    """Malformed or incompatible embedded capabilities fail at loading."""
    checkpoint, _ = wide_candidate
    saved = torch.load(checkpoint, weights_only=True)
    if bad_cap is None:
        saved.pop("wind_max_horizontal_m_s")
    else:
        saved["wind_max_horizontal_m_s"] = bad_cap
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="wind_max_horizontal_m_s"):
        make_app(wide_candidate)


def test_wide_reference_rejects_rehashed_invalid_state_domain(
    wide_candidate: tuple[Path, Path],
) -> None:
    """A matching hash does not establish compatible state-domain metadata."""
    reference_path = wide_candidate[1]
    reference = json.loads(reference_path.read_text())
    reference.pop("reference_sha256")
    reference["inspection_state_domain"]["root_horizontal_speed_max_m_s"] = 10
    reference["reference_sha256"] = content_sha256(reference)
    reference_path.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="root_horizontal_speed_max_m_s"):
        simulation.WindSimulationApp(simulation.parse_config([
            "--wind-backend", "analytic", "--reference-config", str(reference_path),
        ]), bridge=RecordingBridge())
