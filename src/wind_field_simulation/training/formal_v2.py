"""Freeze and evaluate one independently sampled 10 m/s formal test batch."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from .dataset_v1 import check_split_leakage, file_sha256, label_states, load_dataset
from .evaluate_v1 import error_metrics
from .model_v1 import KIND, WIDE_KIND, X4WindAeroV1, load_checkpoint, predict_increment
from .sampling import WIDE_SPEED_STRATA, sample_wide_partition
from .study_v2 import (
    DEV_KIND, TEST_KIND, concatenate, gated_errors, source_digests,
    validate_gate_limits, write_dataset, write_json,
)
from .xml_parameters import PROJECT_ROOT, content_sha256, load_reference_bundle

PROTOCOL_KIND = "x4_950_wind_10_frozen_evaluation_v2"
REPORT_KIND = "x4_950_wind_10_formal_evaluation_v2"
CHALLENGES = ("fast_headwind_high_tilt", "nearly_matched_airflow", "high_wind_high_rate_or_tilt")


def evaluator_source_digests() -> dict[str, str]:
    """Fingerprint the sampling, evaluation and shared model sources."""
    return {
        **source_digests(),
        **{name: file_sha256(PROJECT_ROOT / name) for name in (
            "src/wind_field_simulation/training/formal_v2.py", "scripts/evaluate_x4_wind_10_study.py",
        )},
    }


def comparable_reference_payload(reference: dict[str, Any]) -> dict[str, Any]:
    """Return frozen reference contents except the expected cap/version digests."""
    payload = {key: value for key, value in reference.items()
               if key not in ("reference_sha256", "assumptions_file_sha256")}
    payload["assumptions"] = {key: value for key, value in reference["assumptions"].items()
                              if key != "wind_max_horizontal_m_s"}
    return payload


def freeze_evaluation(
    manifest_path: Path, checkpoint_path: Path, baseline_path: Path, protocol_path: Path,
) -> Path:
    """Bind candidate SHA256, gate limits and independent sampler before test generation.

    Args:
        manifest_path: Development dataset manifest and declared test seed.
        checkpoint_path: Selected wide-domain weight with its development record.
        baseline_path: Original 3 m/s weight for comparison on overlap cases.
        protocol_path: New JSON path. Evaluation writes beside it in formal_test.

    Returns:
        The immutable protocol path; no formal states or test labels are opened.
    """
    data, manifest = load_dataset(manifest_path)
    if manifest["kind"] != DEV_KIND or set(data["split"]) != {0, 1}:
        raise ValueError("Formal test requires a development-only selection manifest")
    config = manifest["config"]
    validate_gate_limits(config.get("acceptance"))
    model, candidate = load_checkpoint(checkpoint_path)
    _, baseline = load_checkpoint(baseline_path)
    reference_path = manifest_path.parent / manifest["reference_bundle"]
    reference = load_reference_bundle(reference_path)
    if (candidate["kind"] != WIDE_KIND or candidate["reference_config"] != reference
            or candidate["dataset_manifest_sha256"] != manifest["manifest_sha256"]
            or candidate["dataset_archive_sha256"] != manifest["archive_sha256"]):
        raise ValueError("Candidate and development provenance differ")
    if (baseline["kind"] != KIND
            or comparable_reference_payload(baseline["reference_config"])
            != comparable_reference_payload(reference)):
        raise ValueError("Expected a narrow model with the same frozen reference physics")
    selected = candidate["selected_trial"]["snapshots"][-1]
    torch.set_num_threads(1)
    validation = data["split"] == 1
    prediction = predict_increment(model, *(data[key][validation] for key in (
        "velocity_world", "rotation_body_to_world", "omega_body", "wind_world",
    )))
    _, actual_fractions = gated_errors(
        prediction, data["delta_acceleration_world"][validation],
        data["speed_stratum"][validation], config["acceptance"],
    )
    actual_worst = max(actual_fractions.values())
    if selected["max_gate_fraction"] > 1. or actual_worst > 1.:
        raise ValueError("Candidate has an unmet development gate; formal test stays unopened")
    artifacts = {}
    for name, path in (("development_manifest", manifest_path),
                       ("reference", reference_path), ("candidate", checkpoint_path),
                       ("baseline", baseline_path)):
        artifacts[name] = {"path": os.path.relpath(path, protocol_path.parent), "sha256": file_sha256(path)}
    protocol = {
        "kind": PROTOCOL_KIND, "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts, "reference_sha256": reference["reference_sha256"],
        "test_sampling": config["formal_test"], "speed_strata_m_s": WIDE_SPEED_STRATA,
        "acceptance": config["acceptance"], "selected": {
            "step": selected["step"], "seed": candidate["selected_trial"]["seed"],
            "train_pairs": candidate["selected_trial"]["train_pairs"],
            "objective": candidate["selected_trial"]["objective"],
            "development_max_gate_fraction": selected["max_gate_fraction"],
            "verified_checkpoint_development_max_gate_fraction": actual_worst,
        },
        "selection": "lowest maximum fixed gate fraction on development validation across trials and steps",
        "source_sha256": evaluator_source_digests(),
    }
    protocol["protocol_sha256"] = content_sha256(protocol)
    write_json(protocol_path, protocol)
    print(f"Frozen formal protocol: {protocol_path} ({protocol['protocol_sha256']})", flush=True)
    return protocol_path


def coverage_masks(data: dict[str, NDArray]) -> dict[str, dict[str, NDArray]]:
    """Group state coverage by speed, relative heading, body inflow and joint edges."""
    wind = data["wind_world"]
    velocity = data["velocity_world"]
    euler = data["euler_rpy_rad"]
    angular_rate = np.linalg.norm(data["omega_body"], axis=1)
    body_air = data["air_velocity_body_pair"][:, 0]
    air_speed = np.linalg.norm(body_air, axis=1)
    tilt = np.rad2deg(np.max(np.abs(euler[:, :2]), axis=1))
    flight_speed = np.hypot(velocity[:, 0], velocity[:, 1])
    wind_angle = np.arctan2(wind[:, 1], wind[:, 0])
    relative_yaw = (np.rad2deg(wind_angle - euler[:, 2]) + 180.) % 360. - 180.
    nonzero = data["speed_stratum"] != 0
    bins = {
        "relative_yaw_deg": (relative_yaw, np.arange(-180, 181, 30)),
        "tilt_deg": (tilt, [0., 5., 10., 15., 19., 20.01]),
        "angular_speed_rad_s": (angular_rate, [0., .00000001, .5, 1., 1.4, 1.51]),
        "horizontal_flight_speed_m_s": (flight_speed, [0., .25, 1., 2., 2.9, 3.01]),
        "absolute_vertical_flight_speed_m_s": (np.abs(velocity[:, 2]), [0., .25, .5, .9, 1.01]),
        "body_air_speed_m_s": (air_speed, [0., .05, .25, 1., 3., 5., 7., 9., 11., 12.5, 13.1]),
    }
    for index, axis in enumerate("xyz"):
        bins[f"body_air_{axis}_m_s"] = (body_air[:, index], [-13.1, -10, -7, -3, 0, 3, 7, 10, 13.1])
    groups = {}
    for name, (values, edges) in bins.items():
        groups[name] = {f"[{low:g},{high:g})":
                        (values >= low) & (values < high) & (nonzero if name == "relative_yaw_deg" else True)
                        for low, high in zip(edges[:-1], edges[1:], strict=True)}
    groups["world_direction_sector_deg"] = {
        f"[{index * 30},{(index + 1) * 30})": nonzero & (data["direction_sector"] == index)
        for index in range(12)
    }
    groups["joint_edges"] = {
        "high_inflow_tilt_and_rate": (air_speed >= 12.5) & (tilt >= 19.) & (angular_rate >= 1.4),
        "near_zero_inflow_with_wind": nonzero & (air_speed < .05),
        "high_tilt_high_vertical_velocity": (tilt >= 19.) & (np.abs(velocity[:, 2]) >= .9),
        "high_positive_body_x_and_rate": (body_air[:, 0] >= 10.) & (angular_rate >= 1.4),
        "high_negative_body_x_and_rate": (body_air[:, 0] <= -10.) & (angular_rate >= 1.4),
        "high_positive_body_y_and_rate": (body_air[:, 1] >= 10.) & (angular_rate >= 1.4),
        "high_negative_body_y_and_rate": (body_air[:, 1] <= -10.) & (angular_rate >= 1.4),
    }
    return groups


def grouped_report(
    data: dict[str, NDArray], groups: dict[str, NDArray], prediction: NDArray | None = None,
) -> dict[str, dict[str, Any]]:
    """Count pairs and independent wind groups, optionally reporting XY errors."""
    report = {}
    nonzero = data["speed_stratum"] != 0
    label = data["delta_acceleration_world"]
    for name, selected in groups.items():
        entry = {"pair_count": int(selected.sum()),
                 "wind_group_count": len(np.unique(data["group_id"][selected]))}
        if prediction is not None and np.any(selected):
            entry.update(error_metrics(prediction[selected], label[selected], nonzero[selected]))
            magnitude = np.linalg.norm(prediction[selected, :2] - label[selected, :2], axis=1)
            entry["vector_error_p99_m_s2"] = float(np.percentile(magnitude, 99))
        report[name] = entry
    return report


def zero_and_full_diagnostics(
    model: X4WindAeroV1, data: dict[str, NDArray], raw: NDArray, prediction: NDArray,
) -> dict[str, Any]:
    """Measure exact zero differences and full-output rest/positive-work residuals."""
    air = torch.as_tensor(data["air_velocity_body_pair"], dtype=torch.float32)
    omega = torch.as_tensor(data["omega_body"], dtype=torch.float32)[:, None].expand(-1, 2, -1)
    identical_air = air[:, 1:2].expand(-1, 2, -1)
    with torch.inference_mode():
        full = model(air.reshape(-1, 3), omega.reshape(-1, 3)).numpy().reshape(-1, 2, 3)
        identical = model(identical_air.reshape(-1, 3), omega.reshape(-1, 3)).numpy().reshape(-1, 2, 3)
        rest = model(torch.zeros(1, 3), torch.zeros(1, 3)).numpy()[0]
    no_wind = data["speed_stratum"] == 0
    no_rate = np.all(data["omega_body"] == 0, axis=1)
    work = np.sum(full[no_rate] * data["air_velocity_body_pair"][no_rate], axis=-1)
    zero = {
        "actual_zero_wind_pairs": int(no_wind.sum()),
        "actual_zero_wind_raw_max_abs_m_s2": float(np.max(np.abs(raw[no_wind]))),
        "identical_pair_states_checked": len(identical),
        "raw_identical_pair_difference_max_abs_m_s2": float(np.max(np.abs(identical[:, 0] - identical[:, 1]))),
        "all_projected_z_exact_zero": bool(np.all(prediction[:, 2] == 0.)),
    }
    return {
        "zero_invariants": zero,
        "exact_zero_pass": zero["actual_zero_wind_raw_max_abs_m_s2"] == 0.
        and zero["raw_identical_pair_difference_max_abs_m_s2"] == 0. and zero["all_projected_z_exact_zero"],
        "full_body_axis_rmse_m_s2": np.sqrt(np.mean(
            (full - data["full_acceleration_body"]) ** 2, axis=(0, 1),
        )).tolist(),
        "full_acceleration_at_rest_body_m_s2": rest.tolist(),
        "zero_omega_pair_arms": int(work.size),
        "zero_omega_positive_work_count_above_1e_minus_8": int((work > 1e-8).sum()),
        "zero_omega_max_work_m2_s3": float(work.max()),
    }


def evaluate_frozen(protocol_path: Path) -> Path:
    """Verify the frozen protocol, generate the isolated test, and save raw results once.

    Args:
        protocol_path: An immutable protocol bound to exact candidate, prior
            baseline, source implementation, reference, test seed and gates.

    Returns:
        Full formal evaluation report beside the frozen protocol.
    """
    protocol = json.loads(protocol_path.read_text())
    payload = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if (protocol.get("kind") != PROTOCOL_KIND
            or protocol.get("protocol_sha256") != content_sha256(payload)):
        raise ValueError("Formal protocol fingerprint mismatch")
    validate_gate_limits(protocol.get("acceptance"))
    if protocol["source_sha256"] != evaluator_source_digests():
        raise ValueError("Frozen sampler/evaluator source fingerprint mismatch")
    paths = {}
    for name, artifact in protocol["artifacts"].items():
        path = protocol_path.parent / artifact["path"]
        if file_sha256(path) != artifact["sha256"]:
            raise ValueError(f"Frozen {name} fingerprint mismatch")
        paths[name] = path
    dev, dev_manifest = load_dataset(paths["development_manifest"])
    reference = load_reference_bundle(paths["reference"])
    candidate, _ = load_checkpoint(paths["candidate"])
    baseline, _ = load_checkpoint(paths["baseline"])
    torch.set_num_threads(1)
    output_dir = protocol_path.parent / "formal_test"
    output_dir.mkdir(exist_ok=False)
    sample = protocol["test_sampling"]
    data = label_states(sample_wide_partition(sample["seed"], sample["count"], 2), reference)
    leakage = check_split_leakage(concatenate([dev, data]))
    manifest_path = write_dataset(
        output_dir, paths["reference"], data, TEST_KIND,
        {"test_sampling": sample, "frozen_protocol_sha256": protocol["protocol_sha256"]},
        leakage, reference,
    )
    strata = data["speed_stratum"]
    overlapping = (strata > 0) & (strata <= 5)
    input_names = ("velocity_world", "rotation_body_to_world", "omega_body", "wind_world")
    inputs = [data[name] for name in input_names]
    raw = predict_increment(candidate, *inputs, project_xy=False)
    prediction = predict_increment(candidate, *inputs)
    baseline_prediction = predict_increment(baseline, *(values[overlapping] for values in inputs))
    scores, fractions = gated_errors(prediction, data["delta_acceleration_world"], strata, protocol["acceptance"])
    groups = {"full": np.ones(len(strata), dtype=bool), "overlap": overlapping,
              **{name: strata == index for index, (name, _, _) in enumerate(WIDE_SPEED_STRATA)}}
    scored = grouped_report(data, groups, prediction)
    baseline_full = np.zeros_like(prediction)
    baseline_full[overlapping] = baseline_prediction
    baseline_scores = grouped_report(data, {
        "overlap": overlapping, **{name: overlapping & mask for name, mask in groups.items()
                                   if name not in ("full", "overlap", "0") and np.any(overlapping & mask)},
    }, baseline_full)
    delta_metrics = ("normalized_nonzero_vector_rmse", "vector_error_p95_m_s2",
                     "vector_error_p99_m_s2", "vector_error_max_m_s2")
    overlap_differences = {
        name: {key: scored[name][key] - baseline_scores[name][key] for key in delta_metrics}
        for name in baseline_scores
    }
    diagnostics = zero_and_full_diagnostics(candidate, data, raw, prediction)
    challenge_groups = {
        name: data["challenge_id"] == i for i, name in enumerate(CHALLENGES, 1)
    }
    coverage = {}
    for name, arrays in (("train", {key: value[dev["split"] == 0] for key, value in dev.items()}),
                         ("validation", {key: value[dev["split"] == 1] for key, value in dev.items()}),
                         ("formal_test", data)):
        coverage[name] = {field: grouped_report(arrays, masks, prediction if name == "formal_test" else None)
                          for field, masks in coverage_masks(arrays).items()}
        root_relative_speed = np.linalg.norm(arrays["air_velocity_body_pair"][:, 0], axis=1)
        coverage[name]["max_root_inflow_m_s"] = float(root_relative_speed.max())
        coverage[name]["speed_strata"] = grouped_report(arrays, {
            label: arrays["speed_stratum"] == index for index, (label, _, _) in enumerate(WIDE_SPEED_STRATA)
        })
    predictions_path = output_dir / "predictions.npz"
    with predictions_path.open("xb") as stream:
        np.savez_compressed(
            stream, raw_world=raw, projected_world=prediction,
            reference_raw_world=data["delta_force_world_n"] / reference["mass_total_kg"],
            reference_projected_world=data["delta_acceleration_world"],
            baseline_projected_world_overlap=baseline_prediction,
            sample_id=data["sample_id"], baseline_sample_id=data["sample_id"][overlapping],
        )
    path = output_dir / "evaluation.json"
    write_json(path, {
        "kind": REPORT_KIND, "protocol_sha256": protocol["protocol_sha256"],
        "reference_sha256": protocol["reference_sha256"], "selected": protocol["selected"],
        "candidate_sha256": protocol["artifacts"]["candidate"]["sha256"],
        "baseline_sha256": protocol["artifacts"]["baseline"]["sha256"],
        "development_dataset_manifest_sha256": dev_manifest["manifest_sha256"],
        "test_dataset_manifest_sha256": json.loads(manifest_path.read_text())["manifest_sha256"],
        "predictions_archive_sha256": file_sha256(predictions_path),
        "fixed_limits": protocol["acceptance"], "gate_fractions": fractions,
        "overall_pass": max(fractions.values()) <= 1. and diagnostics["exact_zero_pass"],
        "candidate": scored, "narrow_baseline": baseline_scores, "candidate_minus_baseline": overlap_differences,
        "state_coverage": coverage, "formal_challenges": grouped_report(data, challenge_groups, prediction),
        "full_and_zero_diagnostics": diagnostics,
    })
    print(f"Formal held-out test: {path}; worst gate fraction={max(fractions.values()):.4f}; "
          f"overlap NRMSE={scores['overlap']['normalized_nonzero_vector_rmse']:.3%}; "
          f"P95={scores['overlap']['vector_error_p95_m_s2']:.6f} m/s²", flush=True)
    return path
