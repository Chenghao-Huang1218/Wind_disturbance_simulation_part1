"""Generate, fit and evaluate the grouped X4-950 drag surrogate."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from ..wind_backends import (
    X4_WIDE_WIND_CDA_M2, HorizontalGust, reference_horizontal_drag_area_m2,
)
from ..wind_randomization import DRAW_GROUPS, drag_group, draw_drag_scales
from .dataset_v1 import file_sha256
from .model_v1 import fit_quadratic_prior, load_checkpoint, predict_increment
from .model_v3 import FEATURE_VERSION, KIND, MAX_TRAINED_VARIATION, X4WindAeroV3
from .reference_aerodynamics import reference_wrench_body, validate_wide_inspection_domain
from .sampling import sample_wide_partition
from .study_v2 import COORDINATES, write_json

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE = (
    ROOT / "src/wind_field_simulation/assets/checkpoints/x4_950_wind_10_v2/reference_config.json"
)
TRAIN_SEED, VALIDATION_SEED, TEST_SEED, OPTIMIZER_SEED = 17653, 18659, 19663, 47
TRAIN_PAIRS, VALIDATION_PAIRS, TEST_PAIRS = 30000, 6000, 6000
GATES = {"xy_p95_n": .15, "xy_nrmse": .02, "counterfactual_nrmse_each": .08}


def group_references(reference: dict[str, Any]) -> list[dict[str, Any]]:
    """Partition frozen proxies with the same grouping as runtime analytic drag."""
    return [
        {**reference, "proxies": [proxy for proxy in reference["proxies"]
                                  if drag_group(proxy) == group]}
        for group in DRAW_GROUPS
    ]


def sample_partition(
    reference: dict[str, Any], *, seed: int, count: int, split: int,
) -> dict[str, NDArray]:
    """Generate independent groups of 10 paired state queries per gust process.

    Args:
        reference: Frozen 0–10 m/s teacher.
        seed: State, gust schedule and drag-draw namespace for this partition.
        count: Row count, subject to the existing wide state sampler's domain.
        split: 0 train, 1 validation, 2 test. Only test includes joint corners.

    Returns:
        State, exact replay inputs, three unit-scale full acceleration targets,
        weighted full forces and their raw same-state increment before CdA gain.
    """
    data = sample_wide_partition(seed, count, split)
    base_wind = data["wind_world"].copy()
    groups = count // 10
    rng = np.random.default_rng([seed, 20])
    amplitude = rng.uniform(.5, 2.0, groups)
    period = rng.uniform(2.0, 6.0, groups)
    # Preserve actual zero/10 m/s cases and a subset of steady weak winds.
    strata = data["speed_stratum"][::10]
    amplitude[(strata == 0) | (strata == 10) | (np.arange(groups) % 5 == 0)] = 0
    phase = np.tile([0., .125, .25, .5, .75, 1.125, 1.5, 1.875, 2.25, 2.5], (groups, 1))
    elapsed = (phase * period[:, None]).ravel()
    winds = np.empty_like(base_wind)
    drag_scales = np.empty((groups, len(DRAW_GROUPS)))
    wind_seeds = np.empty(groups, dtype=np.int64)
    corner = np.empty(groups, dtype=bool)
    for group_index in range(groups):
        # Seed families are disjoint by split seed and group. Attempt remains
        # replayable by recording the final accepted seed for both draws.
        challenge = split == 2 and group_index % 4 == 0
        for attempt in range(1000):
            wind_seed = (seed * 10**10 + group_index * 1000 + attempt)
            values = list(draw_drag_scales(MAX_TRAINED_VARIATION, wind_seed).values())
            in_corner = bool(np.all(np.abs(np.asarray(values) - 1.0) >= .12))
            if in_corner == challenge:
                break
        else:
            raise RuntimeError("Could not satisfy drag-group sampling constraint")
        wind_seeds[group_index] = wind_seed
        drag_scales[group_index] = values
        corner[group_index] = in_corner
        gust = HorizontalGust(amplitude[group_index], period[group_index], wind_seed)
        start = group_index * 10
        for index in range(start, start + 10):
            winds[index] = gust.velocity(base_wind[index], elapsed[index], 10.0)

    rotation = data["rotation_body_to_world"]
    velocity = data["velocity_world"]
    relative_world = np.stack((velocity - winds, velocity), axis=1)
    relative_body = np.einsum("nji,nkj->nki", rotation, relative_world)
    group_force_body = np.stack([
        reference_wrench_body(relative_body, data["omega_body"][:, None, :], subset)[0]
        for subset in group_references(reference)
    ], axis=2)
    row_scales = np.repeat(drag_scales, 10, axis=0)
    pair_force_body = np.einsum("nkgj,ng->nkj", group_force_body, row_scales)
    pair_force_world = np.einsum("nij,nkj->nki", rotation, pair_force_body)
    data.update(
        wind_world=winds, base_wind_world=base_wind, gust_elapsed=elapsed,
        gust_amplitude=np.repeat(amplitude, 10), gust_period=np.repeat(period, 10),
        wind_seed=np.repeat(wind_seeds, 10), drag_scales=row_scales,
        drag_corner=np.repeat(corner, 10), air_velocity_body_pair=relative_body,
        group_acceleration_body_pair=group_force_body / reference["mass_total_kg"],
        force_body_pair=pair_force_body, force_world_pair=pair_force_world,
        delta_force_world=pair_force_world[:, 0] - pair_force_world[:, 1],
    )
    return data


def source_hashes() -> dict[str, str]:
    """Fingerprint implementation inputs, including the unchanged v2 teacher."""
    files = (
        "scripts/run_x4_wind_randomized_study.py",
        "src/wind_field_simulation/wind_backends.py",
        "src/wind_field_simulation/wind_randomization.py",
        "src/wind_field_simulation/training/randomized_v3.py",
        "src/wind_field_simulation/training/model_v3.py",
        "src/wind_field_simulation/training/model_v1.py",
        "src/wind_field_simulation/training/reference_aerodynamics.py",
        "src/wind_field_simulation/training/sampling.py",
    )
    return {name: file_sha256(ROOT / name) for name in files}


def prepare(reference_path: Path, out: Path) -> Path:
    """Create the train/validation archives and their immutable provenance."""
    out.mkdir(parents=True, exist_ok=False)
    reference = json.loads(reference_path.read_text())
    validate_wide_inspection_domain(reference)
    manifest: dict[str, Any] = {
        "reference_path": str(reference_path.resolve()),
        "reference_file_sha256": file_sha256(reference_path),
        "drag_group_order": list(DRAW_GROUPS), "drag_variation": MAX_TRAINED_VARIATION,
        "sources": source_hashes(), "numpy": np.__version__,
    }
    for name, seed, count, split in (
        ("train", TRAIN_SEED, TRAIN_PAIRS, 0),
        ("validation", VALIDATION_SEED, VALIDATION_PAIRS, 1),
    ):
        print(f"Generating {name}: {count} pairs in {count // 10} complete gust groups", flush=True)
        path = out / f"{name}.npz"
        np.savez_compressed(path, **sample_partition(reference, seed=seed, count=count, split=split))
        manifest[name] = {"seed": seed, "pairs": count, "sha256": file_sha256(path)}
    path = out / "manifest.json"
    write_json(path, manifest)
    return path


def tensors(data: dict[str, NDArray]) -> dict[str, torch.Tensor]:
    """Convert arrays used by deterministic CPU training into float32 tensors."""
    keys = (
        "air_velocity_body_pair", "omega_body", "group_acceleration_body_pair",
        "drag_scales", "rotation_body_to_world",
    )
    return {name: torch.tensor(data[name], dtype=torch.float32) for name in keys}


def group_prediction(network: X4WindAeroV3, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Evaluate both branches as a single batch, retaining group contributions."""
    velocity = batch["air_velocity_body_pair"].reshape(-1, 3)
    omega = batch["omega_body"].repeat_interleave(2, dim=0)
    return network.group_acceleration(velocity, omega).reshape(-1, 2, 3, 3)


def world_increment(
    group_pair: torch.Tensor, batch: dict[str, torch.Tensor], scales: torch.Tensor,
) -> torch.Tensor:
    """Combine the same draw on each branch and rotate the increment to W."""
    body = torch.einsum("ngi,ng->ni", group_pair[:, 0] - group_pair[:, 1], scales)
    return torch.bmm(batch["rotation_body_to_world"], body[..., None]).squeeze(-1)


def load_archive(path: Path) -> dict[str, NDArray]:
    """Load numeric arrays without pickle support."""
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def train(manifest_path: Path, out: Path, *, steps: int) -> Path:
    """Fit group and paired-XY targets with model selection on validation only."""
    if steps <= 0:
        raise ValueError("Training steps must be positive")
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(manifest_path.read_text())
    reference_path = Path(manifest["reference_path"])
    if file_sha256(reference_path) != manifest["reference_file_sha256"]:
        raise ValueError("Reference changed after data generation")
    reference = json.loads(reference_path.read_text())
    partitions = {}
    for name in ("train", "validation"):
        path = manifest_path.with_name(f"{name}.npz")
        if file_sha256(path) != manifest[name]["sha256"]:
            raise ValueError(f"{name} archive changed after generation")
        partitions[name] = load_archive(path)
    for key in ("group_id", "wind_seed"):
        if np.intersect1d(partitions["train"][key], partitions["validation"][key]).size:
            raise ValueError(f"{key} overlaps training and validation")

    torch.set_num_threads(1)
    torch.manual_seed(OPTIMIZER_SEED)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(OPTIMIZER_SEED)
    development = partitions["train"]
    relative = development["air_velocity_body_pair"]
    omega_pair = np.broadcast_to(development["omega_body"][:, None, :], relative.shape)
    state = np.concatenate((relative, omega_pair), axis=-1).reshape(-1, 6)
    targets = development["group_acceleration_body_pair"]
    stats = {
        "input_mean": state.mean(axis=0).tolist(),
        "input_std": np.maximum(state.std(axis=0), 1e-6).tolist(),
        "group_scale": np.maximum(targets.std(axis=(0, 1)), .005).tolist(),
    }
    prior = np.stack([fit_quadratic_prior(subset) for subset in group_references(reference)])
    network = X4WindAeroV3(**stats, prior_coefficients=prior)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.003, weight_decay=1e-6)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=.00015)
    training, validation = (tensors(partitions[name]) for name in ("train", "validation"))
    weighted_scales = training["drag_scales"]
    train_delta = world_increment(training["group_acceleration_body_pair"], training, weighted_scales)
    delta_scale = train_delta[:, :2].std(dim=0).clamp_min(.05)
    validation_delta = world_increment(
        validation["group_acceleration_body_pair"], validation, validation["drag_scales"],
    )
    history = []
    best_score = float("inf")
    best_state = None
    for step in range(1, steps + 1):
        rows = rng.integers(len(relative), size=256)
        batch = {name: tensor[rows] for name, tensor in training.items()}
        group_pair = group_prediction(network, batch)
        delta = world_increment(group_pair, batch, batch["drag_scales"])
        group_error = (group_pair - batch["group_acceleration_body_pair"]) / network.group_scale
        delta_error = (delta[:, :2] - train_delta[rows, :2]) / delta_scale
        loss = group_error.square().mean() + delta_error.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 250 == 0 or step in (1, steps):
            with torch.inference_mode():
                predicted = world_increment(
                    group_prediction(network, validation), validation, validation["drag_scales"],
                )
                error = torch.linalg.vector_norm((predicted - validation_delta)[:, :2], dim=1)
                score = float(torch.quantile(error, .95))
            history.append({"step": step, "validation_xy_p95_m_s2": score})
            if score < best_score:
                best_score = score
                best_state = {key: tensor.clone() for key, tensor in network.state_dict().items()}
            print(f"Step {step}/{steps}: validation XY P95 {score:.6g} m/s²", flush=True)

    vehicle = json.loads(reference_path.with_name("vehicle_parameters.json").read_text())
    checkpoint = out / "candidate.pt"
    torch.save({
        "kind": KIND, "feature_version": FEATURE_VERSION, "drag_group_order": list(DRAW_GROUPS),
        "max_trained_variation": MAX_TRAINED_VARIATION, "wind_max_horizontal_m_s": 10.0,
        "coordinates_units": COORDINATES,
        "wind_domain": "horizontal resultant 0–10 m/s; fixed initial arm geometry",
        "reference_config": reference, "reference_sha256": reference["reference_sha256"],
        "mass_total_kg": reference["mass_total_kg"], "arm_pose": reference["initial_joints"],
        "xml_mesh_dependencies": vehicle["dependencies"],
        "state_domain": reference["inspection_state_domain"], "normalization": stats,
        "prior_coefficients": prior.tolist(), "state_dict": best_state,
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "dataset_archive_sha256": manifest["train"]["sha256"],
    }, checkpoint)
    write_json(out / "training.json", {
        "seed": OPTIMIZER_SEED, "steps": steps, "batch_size": 256,
        "best": min(history, key=lambda row: row["validation_xy_p95_m_s2"]),
        "history": history, "checkpoint_sha256": file_sha256(checkpoint),
        "torch": str(torch.__version__), "numpy": np.__version__, "python": platform.python_version(),
        "sources": source_hashes(),
    })
    return checkpoint


def force_metrics(predicted: NDArray, reference: NDArray) -> dict[str, Any]:
    """Report vector error statistics on a set of paired XY forces in N."""
    error = predicted - reference
    norm = np.linalg.norm(error, axis=-1)
    rms = float(np.sqrt(np.mean(norm**2)))
    reference_rms = float(np.sqrt(np.mean(np.sum(reference**2, axis=-1))))
    return {
        "pairs": len(norm), "reference_rms_n": reference_rms, "vector_rmse_n": rms,
        "xy_nrmse": rms / reference_rms if reference_rms else None,
        "axis_rmse_n": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "xy_p95_n": float(np.quantile(norm, .95)), "xy_max_n": float(norm.max()),
    }


def benchmark(
    data: dict[str, NDArray], reference_path: Path, baseline: Path, checkpoint: Path,
) -> dict[str, Any]:
    """Replay eight held-out processes through the complete MuJoCo force path."""
    import time
    from contextlib import ExitStack

    from .. import simulation as sim

    names = ("analytic", "v2", "v3")
    range_indices = np.flatnonzero(
        (data["gust_amplitude"][::10] > 0) & (np.arange(len(data["wind_seed"]) // 10) % 4 != 3)
    )
    selected = range_indices[np.linspace(0, len(range_indices) - 1, 8, dtype=int)]
    elapsed_ns: dict[str, list[int]] = {name: [] for name in names}
    forces: dict[str, list[NDArray]] = {name: [] for name in names}
    mass = json.loads(reference_path.read_text())["mass_total_kg"]
    max_analytic_label_error = 0.
    zero_vertical = True
    rng = np.random.default_rng(TEST_SEED)
    for group_index in selected:
        start = int(group_index) * 10
        wind_seed = int(data["wind_seed"][start])
        with ExitStack() as cleanup:
            apps = {}
            for name in names:
                argv = [
                    "--no-viewer", "--reference-config", str(reference_path),
                    "--wind-backend", "analytic" if name == "analytic" else "pinn-x4",
                    "--drag-variation", str(MAX_TRAINED_VARIATION), "--wind-seed", str(wind_seed),
                    "--gust-amplitude", str(data["gust_amplitude"][start]),
                    "--gust-period", str(data["gust_period"][start]),
                ]
                if name != "analytic":
                    argv += ["--pinn-checkpoint", str(baseline if name == "v2" else checkpoint)]
                app = sim.WindSimulationApp(sim.parse_config(argv))
                cleanup.callback(app._close_runtime)
                apps[name] = app
                app.wind_field = sim.ConstantWindField(data["base_wind_world"][start], 10.)
                app.flight_control.wind_enabled = True
                app._gust_start_s = 0.
                if app.wind_cda_m2 != X4_WIDE_WIND_CDA_M2:
                    raise RuntimeError("Runtime CdA differs from the experiment protocol")
            for step in range(500):
                row = start + step % 10
                # Replay logged elapsed times exactly and retain the EMA across calls.
                for name in rng.permutation(names):
                    app = apps[name]
                    app.data.time = data["gust_elapsed"][row]
                    qpos = app.bindings.main_qpos_adr
                    qvel = app.bindings.main_dof_adr
                    app.data.qpos[qpos + 3:qpos + 7] = data["quaternion_wxyz"][row]
                    app.data.qvel[qvel:qvel + 3] = data["velocity_world"][row]
                    app.data.qvel[qvel + 3:qvel + 6] = data["omega_body"][row]
                    begin = time.perf_counter_ns()
                    wind, _, applied = app._apply_wind_force()
                    duration = time.perf_counter_ns() - begin
                    np.testing.assert_array_equal(wind, data["wind_world"][row])
                    actual = app.data.xfrc_applied[app.bindings.quadrotor_body_id, :3]
                    np.testing.assert_array_equal(applied, actual)
                    zero_vertical &= bool(actual[2] == 0.)
                    if name == "analytic":
                        error = app.last_raw_residual_acceleration * mass - data["delta_force_world"][row]
                        max_analytic_label_error = max(max_analytic_label_error, np.linalg.norm(error))
                    if step >= 100:
                        elapsed_ns[name].append(duration)
                        forces[name].append(applied[:2])
    result = {}
    for name in names:
        p50, p95, p99 = np.quantile(elapsed_ns[name], [.5, .95, .99]) * 1e-6
        result[name] = {"calls": len(elapsed_ns[name]), "p50_ms": float(p50),
                        "p95_ms": float(p95), "p99_ms": float(p99), **force_metrics(
                            np.asarray(forces[name]), np.asarray(forces["analytic"]),
                        )}
    return {"groups": selected.tolist(), "runtime_analytic_label_max_n": float(max_analytic_label_error),
            "all_actual_fz_zero": zero_vertical, "cpu_threads": torch.get_num_threads(),
            "methods": result}


def evaluate(manifest_path: Path, checkpoint: Path, baseline: Path, out: Path) -> Path:
    """Freeze hashes and gates before generating the independent test records."""
    from .model_v3 import load_randomized_checkpoint

    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(manifest_path.read_text())
    reference_path = Path(manifest["reference_path"])
    reference = json.loads(reference_path.read_text())
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (saved["reference_config"] != reference
            or file_sha256(reference_path) != manifest["reference_file_sha256"]
            or saved["dataset_manifest_sha256"] != file_sha256(manifest_path)):
        raise ValueError("Candidate, reference and archive provenance differ")
    network = load_randomized_checkpoint(saved)
    legacy, legacy_metadata = load_checkpoint(baseline)
    if legacy_metadata["reference_config"] != reference:
        raise ValueError("Matched v2 must use the same analytic reference")
    torch.set_num_threads(1)
    protocol_path = out / "protocol.json"
    write_json(protocol_path, {
        "test_seed": TEST_SEED, "test_pairs": TEST_PAIRS, "gates": GATES,
        "reference_file_sha256": file_sha256(reference_path),
        "candidate_sha256": file_sha256(checkpoint), "baseline_sha256": file_sha256(baseline),
        "development_manifest_sha256": file_sha256(manifest_path), "sources": source_hashes(),
    })
    print(f"Generating independent test: {TEST_PAIRS} pairs; protocol {file_sha256(protocol_path)}",
          flush=True)
    test = sample_partition(reference, seed=TEST_SEED, count=TEST_PAIRS, split=2)
    for partition in ("train", "validation"):
        path = manifest_path.with_name(f"{partition}.npz")
        if file_sha256(path) != manifest[partition]["sha256"]:
            raise ValueError(f"{partition} archive changed after generation")
        other = load_archive(path)
        for key in ("group_id", "wind_seed"):
            if np.intersect1d(other[key], test[key]).size:
                raise ValueError(f"{key} overlaps between {partition} and test")
        if test["drag_corner"].sum() == 0 or other["drag_corner"].any():
            raise ValueError("Joint corners must be exclusive to test")
        train_scales = {tuple(values) for values in other["drag_scales"][::10]}
        if train_scales.intersection(map(tuple, test["drag_scales"][::10])):
            raise ValueError(f"Multiplier tuples overlap between {partition} and test")
    archive_path = out / "test.npz"
    np.savez_compressed(archive_path, **test)
    mass = reference["mass_total_kg"]
    gain = X4_WIDE_WIND_CDA_M2 / reference_horizontal_drag_area_m2(reference)
    gain_mass = gain * mass
    target = test["delta_force_world"][:, :2] * gain
    batch = tensors(test)
    with torch.inference_mode():
        group_pair = group_prediction(network, batch)
        # Runtime sums each branch before subtraction. Use the same arithmetic
        # here, so archived metrics include float32 subtractive cancellation.
        scaled = (group_pair * batch["drag_scales"][:, None, :, None]).sum(dim=2)
        nominal = group_pair.sum(dim=2)
        rotation = test["rotation_body_to_world"]
        v3 = np.einsum("nij,nj->ni", rotation, (scaled[:, 0] - scaled[:, 1]).numpy())
        ablated = np.einsum("nij,nj->ni", rotation, (nominal[:, 0] - nominal[:, 1]).numpy())
    v2 = predict_increment(legacy, test["velocity_world"], rotation, test["omega_body"],
                           test["wind_world"])
    speed = np.linalg.norm(test["wind_world"], axis=-1)
    masks = {
        "all": np.ones(len(speed), dtype=bool), "drag_corners": test["drag_corner"],
        "gust_processes": test["gust_amplitude"] > 0,
        "zero_wind": speed == 0., "weak_nonzero": (speed > 0.) & (speed < .25),
        "at_10_m_s": np.isclose(speed, 10., atol=1e-10, rtol=0),
    }
    methods = {"analytic": target, "v2": v2[:, :2] * gain_mass,
               "v3_nominal_ablation": ablated[:, :2] * gain_mass,
               "v3": v3[:, :2] * gain_mass}
    subsets = {
        label: {name: force_metrics(predicted[mask], target[mask])
                for name, predicted in methods.items()}
        for label, mask in masks.items()
    }
    # By construction dF/ds_g is the unit-group contribution. Measure the
    # response to moving that group alone from 0.8 to 1.2 at every held-out state.
    group_target = test["group_acceleration_body_pair"]
    group_axes = np.einsum("nij,ngj->ngi", rotation,
                           group_target[:, 0] - group_target[:, 1])
    group_predicted = np.einsum("nij,ngj->ngi", rotation, (group_pair[:, 0] - group_pair[:, 1]).numpy())
    counterfactual = {}
    for index, group in enumerate(DRAW_GROUPS):
        teacher_response = group_axes[:, index, :2] * (.4 * gain_mass)
        predicted_response = group_predicted[:, index, :2] * (.4 * gain_mass)
        counterfactual[group] = force_metrics(predicted_response, teacher_response)

    metrics = subsets["all"]["v3"]
    print("Benchmarking production force application on matched held-out states", flush=True)
    timings = benchmark(test, reference_path, baseline, checkpoint)
    report = {
        "protocol_sha256": file_sha256(protocol_path), "test_archive_sha256": file_sha256(archive_path),
        "nominal_xy_gain": gain, "test_groups": TEST_PAIRS // 10,
        "test_groups_with_nonzero_gust_amplitude": int(masks["gust_processes"].sum() // 10),
        "force_metrics": subsets, "counterfactual_scale_0p8_to_1p2": counterfactual,
        "force_path_benchmark": timings,
        "gates": {
            "xy_p95": metrics["xy_p95_n"] <= GATES["xy_p95_n"],
            "xy_nrmse": metrics["xy_nrmse"] <= GATES["xy_nrmse"],
            "improves_on_v2": metrics["vector_rmse_n"] < subsets["all"]["v2"]["vector_rmse_n"],
            "improves_on_nominal_ablation": (
                metrics["vector_rmse_n"] < subsets["all"]["v3_nominal_ablation"]["vector_rmse_n"]
            ),
            "zero_wind": subsets["zero_wind"]["v3"]["xy_max_n"] == 0.,
            "counterfactual_each": all(item["xy_nrmse"] <= GATES["counterfactual_nrmse_each"]
                                       for item in counterfactual.values()),
            "runtime_label_replay": timings["runtime_analytic_label_max_n"] <= 1e-10,
            "runtime_fz": timings["all_actual_fz_zero"],
        },
    }
    path = out / "evaluation.json"
    write_json(path, report)
    print(f"Evaluation gates: {report['gates']}", flush=True)
    return path
