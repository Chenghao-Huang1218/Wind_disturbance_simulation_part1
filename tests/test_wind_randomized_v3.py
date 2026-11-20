"""Regression tests for shared grouped drag and versioned conditional inference."""

import numpy as np
import pytest
import torch

from wind_field_simulation import simulation
from wind_field_simulation.training.model_v3 import X4WindAeroV3
from wind_field_simulation.training.randomized_v3 import DEFAULT_REFERENCE, sample_partition
from wind_field_simulation.wind_backends import HorizontalGust
from wind_field_simulation.wind_randomization import DRAW_GROUPS, draw_drag_scales, with_drag_scales
from wind_field_simulation.training.reference_aerodynamics import paired_wind_labels

def test_pair_generation_replays_the_production_experiment() -> None:
    """Verify grouped seeds, labels, true zero and radial boundary coverage."""
    import json

    reference = json.loads(DEFAULT_REFERENCE.read_text())
    data = sample_partition(reference, seed=19663, count=2880, split=2)
    for index in np.linspace(0, len(data["group_id"]) - 1, 37, dtype=int):
        seed = int(data["wind_seed"][index])
        scales = draw_drag_scales(.2, seed)
        np.testing.assert_array_equal(list(scales.values()), data["drag_scales"][index])
        gust = HorizontalGust(data["gust_amplitude"][index], data["gust_period"][index], seed)
        wind = gust.velocity(data["base_wind_world"][index], data["gust_elapsed"][index], 10.)
        np.testing.assert_array_equal(wind, data["wind_world"][index])
        label = paired_wind_labels(
            data["velocity_world"][index], data["quaternion_wxyz"][index],
            data["omega_body"][index], wind, with_drag_scales(reference, scales),
        )
        np.testing.assert_allclose(
            data["delta_force_world"][index], label["delta_force_world_n"], atol=1e-12,
        )
    speed = np.linalg.norm(data["wind_world"], axis=-1)
    assert np.count_nonzero(speed == 0) and np.count_nonzero(np.isclose(speed, 10))
    np.testing.assert_array_equal(data["delta_force_world"][speed == 0], 0.0)
    np.testing.assert_array_equal(data["drag_scales"], data["drag_scales"][::10].repeat(10, axis=0))
    assert len(np.unique(data["wind_seed"])) == len(data["wind_seed"]) // 10


def test_model_uses_each_scale_in_group_order() -> None:
    """Isolate each physical input from the two other multiplier channels."""
    model = X4WindAeroV3(np.zeros(6), np.ones(6), np.ones((3, 3)), np.eye(3))
    velocity = torch.tensor([[2.0, 2.0, 2.0]]).repeat(4, 1)
    rates = torch.zeros_like(velocity)
    scales = torch.cat((torch.ones((1, 3)), 1.0 + .2 * torch.eye(3)))
    force = model(velocity, rates, scales)
    assert len(DRAW_GROUPS) == 3
    torch.testing.assert_close(force[1:] - force[0], -.8 * torch.eye(3))


def test_six_dimensional_baseline_can_replay_the_same_wind_schedule() -> None:
    """Keep the explicit six-dimensional fallback available for matched runs."""
    config = simulation.parse_config([
        "--wind-backend", "pinn-x4", "--pinn-checkpoint",
        str(DEFAULT_REFERENCE.with_name("candidate.pt")), "--drag-variation", ".2",
        "--gust-amplitude", "1.5",
    ])
    assert config.drag_variation == pytest.approx(.2)


def test_bundled_v3_runs_in_the_normal_force_path() -> None:
    """Load trained weights and verify the fixed seeded draw and CdA application."""
    app = simulation.WindSimulationApp(simulation.parse_config([
        "--drag-variation", ".2", "--wind-seed", "42", "--gust-amplitude", "1.5",
    ]))
    try:
        assert app.pinn_predictor.drag_group_scales == draw_drag_scales(.2, 42)
        assert app.wind_cda_m2 == .25
        app.wind_field = simulation.ConstantWindField([4., 4., 0.], 10.)
        app.flight_control.wind_enabled = True
        _, _, force = app._apply_wind_force()
        assert np.linalg.norm(force[:2]) > 0.1
        assert force[2] == 0.
        np.testing.assert_array_equal(
            app.pinn_predictor.predict([1, 2, 3], [1, 0, 0, 0], [.2, .5, .1], [0, 0, 0]),
            np.zeros(3),
        )
    finally:
        app._close_runtime()

    with pytest.raises(ValueError, match="supports --drag-variation up to 0.20"):
        simulation.WindSimulationApp(simulation.parse_config(["--drag-variation", ".21"]))
