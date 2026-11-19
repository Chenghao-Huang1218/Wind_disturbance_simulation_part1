"""Reproduce development, budget comparison and frozen test for the 10 m/s model."""

from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .dataset_v1 import check_split_leakage, file_sha256, label_states, load_dataset
from .evaluate_v1 import error_metrics
from .model_v1 import WIDE_KIND, X4WindAeroV1, fit_normalization, fit_quadratic_prior
from .reference_aerodynamics import reference_wind_limit
from .sampling import WIDE_SPEED_STRATA, sample_wide_partition
from .train_v1 import pair_loss
from .xml_parameters import (
    PROJECT_ROOT, content_sha256, export_reference_bundle, load_reference_bundle,
)

STUDY_KIND = "x4_950_wind_10_study_v2"
DEV_KIND = "x4_950_wind_10_development_pairs_v2"
TEST_KIND = "x4_950_wind_10_formal_test_pairs_v2"
COORDINATES = "B FLU at root O; W ENU-like; WXYZ B→W; m/s, rad/s, m/s²"
BATCH_ARRAYS = {
    "air": "air_velocity_body_pair", "omega": "omega_body",
    "rotation": "rotation_body_to_world", "full": "full_acceleration_body",
    "delta": "delta_acceleration_world",
}
REFERENCE_FILES = ("reference_config.json", "vehicle_parameters.json", "geometry_manifest.json")
GATE_LIMIT_NAMES = (
    "full_axis_rmse", "full_vector_p95", "full_normalized_rmse",
    "overlap_vector_p95", "overlap_normalized_rmse", "weak_vector_p95",
    "low_band_normalized_rmse",
)


def validate_gate_limits(limits: object) -> None:
    """Require finite positive numeric thresholds for every formal gate."""
    if not isinstance(limits, dict) or set(limits) != set(GATE_LIMIT_NAMES):
        raise ValueError("Acceptance must contain every named wide evaluation gate")
    for name in GATE_LIMIT_NAMES:
        value = limits[name]
        if type(value) not in (int, float) or not 0 < value <= sys.float_info.max:
            raise ValueError(f"Acceptance {name} must be finite and positive")


def validate_sample_budgets(budgets: object, train_pairs: int) -> None:
    """Keep each nested budget within the available complete training groups."""
    if not isinstance(budgets, list) or not budgets:
        raise ValueError("Provide at least one wide training sample budget")
    if any(type(count) is not int or not 2880 <= count <= train_pairs or count % 10
           for count in budgets):
        raise ValueError("Wide sample budgets must be complete groups of ten within the training partition")


def write_json(path: Path, value: Any) -> None:
    """Create an immutable study artifact with finite JSON values."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_digests() -> dict[str, str]:
    """Fingerprint the study implementation and reused training sources."""
    filenames = (
        "dataset_v1.py", "evaluate_v1.py", "model_v1.py", "reference_aerodynamics.py",
        "sampling.py", "study_v2.py", "train_v1.py", "xml_parameters.py",
    )
    paths = [Path(__file__).parent / name for name in filenames]
    paths.append(PROJECT_ROOT / "scripts/run_x4_wind_10_study.py")
    return {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in sorted(paths)}


def export_wide_reference(xml_path: Path, config_path: Path, destination: Path) -> Path:
    """Export the explicit wide assumption bundle in a fresh destination."""
    assumptions = json.loads(config_path.read_text())
    if reference_wind_limit(assumptions) != 10.:
        raise ValueError("Expected a 10 m/s reference assumptions file")
    destination.mkdir(parents=True, exist_ok=False)
    export_reference_bundle(xml_path, config_path, destination)
    return destination / "reference_config.json"


def concatenate(parts: list[dict[str, NDArray]]) -> dict[str, NDArray]:
    """Combine same-schema partition arrays in their original order."""
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def write_dataset(
    output_dir: Path, reference_path: Path, data: dict[str, NDArray],
    kind: str, config: dict[str, Any], leakage: dict[str, int],
    reference: dict[str, Any],
) -> Path:
    """Archive numeric arrays and the verified frozen geometry alongside a manifest."""
    bundle = output_dir / "reference"
    bundle.mkdir()
    for name in REFERENCE_FILES:
        shutil.copyfile(reference_path.parent / name, bundle / name)
    archive = output_dir / "pairs.npz"
    with archive.open("xb") as stream:
        np.savez_compressed(stream, **data)
    counts = {name: int(np.count_nonzero(data["split"] == index))
              for index, name in enumerate(("train", "validation", "formal_test"))}
    wind_conditions = {}
    for split_name, split_index in zip(counts, range(3), strict=True):
        selected = (data["split"] == split_index)[::10]
        if np.any(selected):
            ids = 12 * data["speed_stratum"][::10][selected] + data["direction_sector"][::10][selected]
            wind_conditions[split_name] = np.bincount(
                ids, minlength=12 * len(WIDE_SPEED_STRATA),
            ).reshape(-1, 12).tolist()
    manifest = {
        "kind": kind, "config": config, "reference_sha256": reference["reference_sha256"],
        "reference_bundle": "reference/reference_config.json", "archive": archive.name,
        "archive_sha256": file_sha256(archive), "pair_count_by_split": counts, "group_size": 10,
        "wind_max_horizontal_m_s": 10., "speed_strata_m_s": WIDE_SPEED_STRATA,
        "mass_total_kg": reference["mass_total_kg"], "arm_pose": reference["initial_joints"],
        "state_domain": reference["inspection_state_domain"], "coordinates_units": COORDINATES,
        "group_counts_by_partition_speed_band_and_nominal_direction_sector": wind_conditions,
        "leakage_audit": leakage, "source_sha256": source_digests(),
    }
    manifest["manifest_sha256"] = content_sha256(manifest)
    path = output_dir / "manifest.json"
    write_json(path, manifest)
    return path


def prepare_development(config_path: Path, reference_path: Path, output_dir: Path) -> Path:
    """Sample and label development partitions using disjoint declared seeds."""
    config = json.loads(config_path.read_text())
    seeds = [config[part]["seed"] for part in ("train", "validation", "formal_test")]
    if config["kind"] != STUDY_KIND or len(set(seeds)) != 3:
        raise ValueError("Require a wide study configuration and independent partition seeds")
    validate_gate_limits(config.get("acceptance"))
    validate_sample_budgets(config["training"].get("sample_budgets"), config["train"]["count"])
    reference = load_reference_bundle(reference_path)
    if reference_wind_limit(reference) != 10.:
        raise ValueError("Development data requires the frozen 10 m/s reference")
    output_dir.mkdir(parents=True, exist_ok=False)
    data = label_states(concatenate([
        sample_wide_partition(config[part]["seed"], config[part]["count"], split)
        for split, part in enumerate(("train", "validation"))
    ]), reference)
    path = write_dataset(output_dir, reference_path, data, DEV_KIND, config,
                         check_split_leakage(data), reference)
    print(f"Development manifest: {path}", flush=True)
    return path


def gated_errors(
    prediction: NDArray, label: NDArray, strata: NDArray, limits: dict[str, float],
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    """Compute fixed absolute and relative gate scores as fractions of their limits."""
    masks = {"full": np.ones(len(label), dtype=bool), "overlap": (strata > 0) & (strata <= 5)}
    masks.update({name: strata == index for index, (name, _, _) in enumerate(WIDE_SPEED_STRATA)})
    groups = {name: error_metrics(prediction[mask], label[mask], strata[mask] > 0)
              for name, mask in masks.items()}
    fields = {
        "full_axis_rmse": max(groups["full"]["axis_rmse_m_s2"]),
        "full_vector_p95": groups["full"]["vector_error_p95_m_s2"],
        "full_normalized_rmse": groups["full"]["normalized_nonzero_vector_rmse"],
        "overlap_vector_p95": groups["overlap"]["vector_error_p95_m_s2"],
        "overlap_normalized_rmse": groups["overlap"]["normalized_nonzero_vector_rmse"],
        "weak_vector_p95": groups["(0,0.25)"]["vector_error_p95_m_s2"],
    }
    for name, _, _ in WIDE_SPEED_STRATA[2:6]:
        fields[f"low_band_normalized_rmse:{name}"] = groups[name]["normalized_nonzero_vector_rmse"]
    fractions = {key: value / limits[key.split(":")[0]] for key, value in fields.items()}
    return groups, fractions


def fit_delta_weights(
    label: NDArray, strata: NDArray, train: NDArray, config: dict[str, Any],
) -> tuple[NDArray, NDArray]:
    """Balance XY error using RMS levels estimated only on this training budget."""
    scale = np.array([np.sqrt(np.mean(label[train & (strata == index), :2] ** 2))
                      for index in range(len(WIDE_SPEED_STRATA))])
    scale = np.maximum(scale, config["low_wind_scale_floor_m_s2"])
    inverse_energy = scale ** -2
    # Zero-wind difference is exact by construction and uses no loss weight.
    inverse_energy[0] = 0.
    inverse_energy /= inverse_energy[strata[train & (strata > 0)]].mean()
    mix = config["low_wind_weight_mix"]
    return (1. - mix + mix * inverse_energy), scale


def validation_prediction(model: X4WindAeroV1, batch: dict[str, Tensor]) -> NDArray:
    """Compute raw paired world increments with a single model batch."""
    air = batch["air"]
    omega = batch["omega"][:, None, :].expand(-1, 2, -1)
    with torch.inference_mode():
        full = model(air.reshape(-1, 3), omega.reshape(-1, 3)).reshape(-1, 2, 3)
        return torch.einsum("nij,nj->ni", batch["rotation"], full[:, 0] - full[:, 1]).numpy()


def train_trial(
    data: dict[str, NDArray], reference: dict[str, Any], config: dict[str, Any],
    train_count: int, seed: int, objective: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Tensor]]:
    """Run one fixed three-seed budget cell, selecting from development validation."""
    settings = config["training"]
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train = np.zeros(len(data["split"]), dtype=bool)
    train[np.flatnonzero(data["split"] == 0)[:train_count]] = True
    valid = data["split"] == 1
    strata = data["speed_stratum"]
    core = np.concatenate((data["air_velocity_body_pair"],
                           np.repeat(data["omega_body"][:, None], 2, axis=1)), axis=2)
    target = data["delta_acceleration_world"]
    stats = fit_normalization(core, data["full_acceleration_body"], target, train)
    prior = fit_quadratic_prior(reference)
    model = X4WindAeroV1(stats["input_mean"], stats["input_std"], stats["full_scale"], prior,
                         wind_max_horizontal_m_s=10.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.5, patience=settings["lr_patience_checks"],
        threshold_mode="abs", threshold=settings["minimum_improvement"],
        min_lr=settings["minimum_learning_rate"],
    )
    tensors = {name: torch.as_tensor(data[key], dtype=torch.float32)
               for name, key in BATCH_ARRAYS.items()}
    validation = {name: value[valid] for name, value in tensors.items()}
    full_scale = torch.as_tensor(stats["full_scale"], dtype=torch.float32)
    delta_scale = torch.as_tensor(stats["delta_scale"], dtype=torch.float32)
    multipliers, scale = fit_delta_weights(target, strata, train, settings)
    if objective == "global":
        multipliers = np.ones_like(multipliers)
    weights = torch.as_tensor(multipliers[strata], dtype=torch.float32)
    train_ids = torch.as_tensor(np.flatnonzero(train))
    best_rank = float("inf")
    best_state = {}
    best_row = {}
    history, snapshots = [], []
    maximum_steps = max(settings["step_budgets"])
    step = 0
    start = perf_counter()
    while step < maximum_steps:
        shuffled = train_ids[torch.randperm(len(train_ids), generator=generator)]
        for batch_ids in shuffled.split(settings["batch_size"]):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            training_loss = pair_loss(
                model, {name: value[batch_ids] for name, value in tensors.items()},
                full_scale, delta_scale, settings["positive_work_weight"],
                delta_weights=weights[batch_ids],
            )
            if not torch.isfinite(training_loss):
                raise ValueError("Nonfinite wide-domain training loss")
            training_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            step += 1
            at_budget = step in settings["step_budgets"]
            if at_budget or step % settings["validation_interval"] == 0:
                model.eval()
                prediction = validation_prediction(model, validation)
                scores, fractions = gated_errors(prediction, target[valid], strata[valid], config["acceptance"])
                with torch.inference_mode():
                    valid_loss = float(pair_loss(
                        model, validation, full_scale, delta_scale, settings["positive_work_weight"],
                        delta_weights=weights[valid],
                    ))
                scheduler.step(valid_loss)
                rank = max(fractions.values())
                row = {
                    "step": step, "lr": optimizer.param_groups[0]["lr"],
                    "validation_loss": valid_loss, "max_gate_fraction": rank,
                    "gate_fractions": fractions,
                }
                history.append(row)
                if rank < best_rank - settings["minimum_improvement"]:
                    best_rank = rank
                    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                    best_row = {**row, "validation_scores": scores}
                if at_budget:
                    snapshots.append({"budget_steps": step, **best_row})
            if step == maximum_steps:
                break
    summary = {
        "train_pairs": int(train.sum()), "seed": seed, "objective": objective,
        "elapsed_seconds": perf_counter() - start, "stopping_reason": "fixed optimizer step budget",
        "normalization": stats, "prior_coefficients": prior.tolist(),
        "low_wind_train_xy_rms_with_floor_by_speed_band": scale.tolist(),
        "delta_loss_multiplier_by_speed_band": multipliers.tolist(), "snapshots": snapshots,
    }
    return summary, history, best_state


def train_study(manifest_path: Path, output_dir: Path) -> Path:
    """Compare nested counts, step ceilings and weighted/unweighted seed restarts."""
    data, manifest = load_dataset(manifest_path)
    config = manifest["config"]
    if manifest["kind"] != DEV_KIND or set(data["split"]) != {0, 1}:
        raise ValueError("Wide training accepts only the development dataset")
    validate_gate_limits(config.get("acceptance"))
    settings = config["training"]
    validate_sample_budgets(settings.get("sample_budgets"), int(np.count_nonzero(data["split"] == 0)))
    reference_path = manifest_path.parent / manifest["reference_bundle"]
    reference = load_reference_bundle(reference_path)
    vehicle = json.loads((reference_path.parent / "vehicle_parameters.json").read_text())
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(settings["torch_threads"])
    torch.use_deterministic_algorithms(True)
    results, histories = [], {}
    best_trial, best_state = None, {}
    best_score = float("inf")
    for train_count in settings["sample_budgets"]:
        for objective in settings["objectives"]:
            for seed in settings["initialization_seeds"]:
                name = f"{train_count}_{objective}_{seed}"
                summary, history, state = train_trial(data, reference, config, train_count, seed, objective)
                histories[name] = history
                results.append(summary)
                score = summary["snapshots"][-1]["max_gate_fraction"]
                overlap = summary["snapshots"][-1]["validation_scores"]["overlap"]
                print(f"{name}: worst gate fraction={score:.4f}, "
                      f"overlap NRMSE={overlap['normalized_nonzero_vector_rmse']:.3%}, "
                      f"P95={overlap['vector_error_p95_m_s2']:.6f} m/s²", flush=True)
                if score < best_score:
                    best_trial, best_state, best_score = summary, state, score
    checkpoint = output_dir / "candidate.pt"
    saved = {
        "kind": WIDE_KIND, "feature_version": "body_air_velocity_omega_6d_v1",
        "architecture": "fixed nonnegative quadratic prior + 6-64-64-3 Tanh residual",
        "state_dict": best_state, "normalization": best_trial["normalization"],
        "prior_coefficients": best_trial["prior_coefficients"],
        "prior_source": "nominal ±unit-axis least squares; not measured",
        "reference_sha256": reference["reference_sha256"], "reference_config": reference,
        "xml_mesh_dependencies": vehicle["dependencies"],
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "dataset_archive_sha256": manifest["archive_sha256"],
        "mass_total_kg": reference["mass_total_kg"], "arm_pose": reference["initial_joints"],
        "coordinates_units": COORDINATES, "wind_max_horizontal_m_s": 10.,
        "wind_domain": "horizontal resultant 0–10 m/s; fixed initial arm geometry",
        "training_config": settings, "selected_trial": best_trial,
        "sampling_seed": config["train"]["seed"], "state_domain": reference["inspection_state_domain"],
        "device": "cpu", "dtype": "float32", "deterministic_algorithms": True,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "numpy": str(np.__version__), "platform": platform.platform()},
        "source_sha256": source_digests(), "status": "synthetic 10 m/s reference surrogate candidate",
    }
    with checkpoint.open("xb") as stream:
        torch.save(saved, stream)
    write_json(output_dir / "training_summary.json", {
        "study_kind": STUDY_KIND, "training_config": settings, "acceptance": config["acceptance"],
        "candidate_sha256": file_sha256(checkpoint), "candidate_trial": best_trial, "trials": results,
    })
    write_json(output_dir / "validation_history.json", histories)
    print(f"Selected candidate: {checkpoint} ({best_score:.4f} of worst development gate)", flush=True)
    return checkpoint
