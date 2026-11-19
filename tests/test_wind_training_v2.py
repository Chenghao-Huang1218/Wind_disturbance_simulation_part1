"""Regression checks for the independent 0–10 m/s experiments."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wind_field_simulation.training import sampling, study_v2, train_v1
from wind_field_simulation.training.xml_parameters import PROJECT_ROOT


def test_wide_partitions_have_nested_train_budget_and_direction_coverage() -> None:
    full = sampling.sample_wide_partition(9251, 12000, 0)
    short = sampling.sample_wide_partition(9251, 6000, 0)
    for key, values in short.items():
        np.testing.assert_array_equal(values, full[key][:6000])
    valid = sampling.sample_wide_partition(9252, 3000, 1)
    test = sampling.sample_wide_partition(9253, 3000, 2)
    for data in (short, valid, test):
        wind = data['wind_world']
        speed = np.hypot(wind[:, 0], wind[:, 1])
        assert np.all(speed <= np.nextafter(10., np.inf))
        assert np.all(wind[:, 2] == 0.)
        assert np.all(wind.reshape(-1, 10, 3) == wind[::10, None, :])
        assert set(zip(data['speed_stratum'][::10], data['direction_sector'][::10])) == {
            (band, sector) for band in range(11) for sector in range(12)
        }
        np.testing.assert_allclose(speed[data['speed_stratum'] == 5], 3., atol=1e-14)
        np.testing.assert_allclose(speed[data['speed_stratum'] == 10], 10., atol=1e-14)
        assert np.max(np.linalg.norm(data['velocity_world'] - wind, axis=1)) > 12.95
    assert not np.intersect1d(short['group_id'], test['group_id']).size
    assert np.any(test['challenge_id'] > 0)


def test_pair_loss_accepts_training_calibrated_delta_weights() -> None:
    model = train_v1.X4WindAeroV1(np.zeros(6), np.ones(6), np.ones(3), np.zeros(3))
    batch = {'air': torch.zeros((2, 2, 3)), 'omega': torch.ones((2, 3)),
             'rotation': torch.eye(3).expand(2, 3, 3), 'full': torch.zeros((2, 2, 3)),
             'delta': torch.tensor([[1., 1., 0.], [0., 0., 0.]])}
    scale = torch.ones(3)
    assert train_v1.pair_loss(model, batch, scale, scale, 0.).item() == .5
    assert train_v1.pair_loss(model, batch, scale, scale, 0.,
                              delta_weights=torch.tensor([3., 1.])).item() == 1.5


def test_wide_development_uses_production_labels_without_opening_formal_test(tmp_path) -> None:
    from wind_field_simulation.training import study_v2
    from wind_field_simulation.training.xml_parameters import DEFAULT_SCENE, PROJECT_ROOT

    reference = tmp_path / "reference"
    study_v2.export_wide_reference(
        DEFAULT_SCENE, PROJECT_ROOT / 'configs/pinn/x4_950_reference_wind_10.json', reference,
    )
    config = PROJECT_ROOT / 'configs/pinn/x4_950_wind_10_study.json'
    manifest_path = study_v2.prepare_development(
        config, reference / 'reference_config.json', tmp_path / 'dev',
    )
    data, manifest = study_v2.load_dataset(manifest_path)
    np.testing.assert_array_equal(np.bincount(data['split']), [12000, 3000])
    assert set(data['sampling_seed']) == {9251, 9252}
    assert manifest['leakage_audit']['cross_split_core_duplicates'] == 0
    assert np.max(np.abs(data['delta_force_world_n'][:, 2])) > .01
    assert np.all(data['delta_acceleration_world'][:, 2] == 0.)
    assert np.max(np.linalg.norm(data['delta_acceleration_world'], axis=1)) > 1.8
    with pytest.raises(FileExistsError):
        study_v2.prepare_development(config, reference / 'reference_config.json', tmp_path / 'dev')


def test_formal_test_rejects_modified_freeze_before_creating_test_data(tmp_path) -> None:
    from wind_field_simulation.training import formal_v2

    freeze = tmp_path / "eval_protocol.json"
    freeze.write_text(json.dumps({
        "kind": "x4_950_wind_10_frozen_evaluation_v2",
        "candidate": {"path": "absent.pt", "sha256": "0" * 64},
        "protocol_sha256": "0" * 64,
    }))
    with pytest.raises(ValueError, match="protocol fingerprint"):
        formal_v2.evaluate_frozen(freeze)
    assert not (tmp_path / "formal_test").exists()


@pytest.mark.parametrize("invalid_limit", [-.001, 0., "0.01", float("inf")])
def test_wide_development_rejects_invalid_gate_limits(
    tmp_path: Path, invalid_limit: float | str,
) -> None:
    """Malformed thresholds cannot bypass validation or formal gates."""
    config = json.loads((PROJECT_ROOT / "configs/pinn/x4_950_wind_10_study.json").read_text())
    config["acceptance"]["full_axis_rmse"] = invalid_limit
    config_path = tmp_path / "bad_config.json"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="full_axis_rmse"):
        study_v2.prepare_development(config_path, tmp_path / "reference.json", tmp_path / "dev")
    assert not (tmp_path / "dev").exists()
