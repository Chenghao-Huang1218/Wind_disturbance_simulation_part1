"""X4 body-frame aerodynamic surrogate, separate from legacy 17D PINN."""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Any
from numpy.typing import NDArray
from torch import Tensor, nn

from .reference_aerodynamics import (
    LEGACY_WIND_MAX_M_S,
    WIDE_WIND_MAX_M_S,
    reference_wind_limit,
    reference_wrench_body,
    validate_wide_inspection_domain,
)

KIND = "x4_950_wind_increment_xy_v1"
WIDE_KIND = "x4_950_wind_increment_xy_v2"


def fit_normalization(
    core_pair: NDArray, full_pair: NDArray, delta: NDArray, train_mask: NDArray,
) -> dict[str, list[float]]:
    """Fit separate axis statistics using only training rows and both pair arms.

    Args:
        core_pair: Body velocity/angular rate [N,2,6].
        full_pair: Full B acceleration targets [N,2,3], m/s².
        delta: Applied W increments [N,3], m/s².
        train_mask: Boolean [N] marking only the training partition.

    Returns:
        JSON-compatible mean/std and separate full/delta output std scales.
        Output scales have the specified 0.05 m/s² floor, no XY symmetrization.
    """
    if not np.any(train_mask):
        raise ValueError("Training partition is empty")
    inputs = core_pair[train_mask].reshape(-1, 6)
    return {"input_mean": inputs.mean(axis=0).tolist(),
            "input_std": np.maximum(inputs.std(axis=0), 1e-6).tolist(),
            "full_scale": np.maximum(full_pair[train_mask].reshape(-1, 3).std(axis=0), .05).tolist(),
            "delta_scale": np.maximum(delta[train_mask].std(axis=0), .05).tolist()}


def fit_quadratic_prior(reference: dict) -> NDArray[np.float64]:
    """Fit nonnegative diagonal drag coefficients from nominal ±unit axes.

    This is a six-point least-squares prior from the chosen teacher, not a
    physical parameter identification. Cross-axis/omega effects stay residual.
    """
    axes = np.concatenate((np.eye(3), -np.eye(3)))
    acceleration = reference_wrench_body(axes, np.zeros_like(axes), reference)[0]
    acceleration /= reference["mass_total_kg"]
    features = -axes * np.abs(axes)
    return np.maximum(0, np.sum(features * acceleration, axis=0)
                      / np.sum(features**2, axis=0))


class X4WindAeroV1(nn.Module):
    """Fixed nonnegative quadratic drag prior plus a 6→64→64→3 residual.

    Args:
        input_mean: Training mean for [v_air_B, omega_B].
        input_std: Training per-axis standard deviations.
        output_scale: Training full-acceleration scales (m/s²).
        prior_coefficients: Nonnegative nominal unit-axis drag coefficients.
        wind_max_horizontal_m_s: Recorded horizontal setting limit for paired
            inference. The six-dimensional core inputs have their own domain.
    """

    def __init__(self, input_mean: NDArray, input_std: NDArray,
                 output_scale: NDArray, prior_coefficients: NDArray, *,
                 wind_max_horizontal_m_s: float = LEGACY_WIND_MAX_M_S) -> None:
        super().__init__()
        if wind_max_horizontal_m_s not in (LEGACY_WIND_MAX_M_S, WIDE_WIND_MAX_M_S):
            raise ValueError("Unsupported wind_max_horizontal_m_s")
        self.wind_max_horizontal_m_s = float(wind_max_horizontal_m_s)
        for name, values, length in (
            ("input_mean", input_mean, 6), ("input_std", input_std, 6),
            ("output_scale", output_scale, 3), ("prior_coefficients", prior_coefficients, 3),
        ):
            array = np.asarray(values, dtype=np.float32)
            if array.shape != (length,) or not np.isfinite(array).all():
                raise ValueError(f"Invalid {name}")
            if name in ("input_std", "output_scale") and np.any(array <= 0):
                raise ValueError(f"Nonpositive {name}")
            if name == "prior_coefficients" and np.any(array < 0):
                raise ValueError("Prior must be nonnegative")
            self.register_buffer(name, torch.tensor(array))
        self.residual = nn.Sequential(nn.Linear(6, 64), nn.Tanh(), nn.Linear(64, 64),
                                      nn.Tanh(), nn.Linear(64, 3))
        # Start at the archived quadratic prior, not a random aerodynamic force.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, air_velocity_body: Tensor, angular_velocity_body: Tensor) -> Tensor:
        """Return complete B-frame aerodynamic acceleration [N,3], m/s²."""
        if (air_velocity_body.ndim != 2 or air_velocity_body.shape[-1] != 3
                or angular_velocity_body.shape != air_velocity_body.shape):
            raise ValueError("Inputs must both be [N,3]")
        core = torch.cat((air_velocity_body, angular_velocity_body), dim=1)
        residual = self.residual((core - self.input_mean) / self.input_std)
        return (-self.prior_coefficients * air_velocity_body * air_velocity_body.abs()
                + self.output_scale * residual)


@torch.inference_mode()
def predict_increment(
    model: X4WindAeroV1, velocity_world: NDArray, rotation: NDArray,
    omega_body: NDArray, wind_world: NDArray, *, project_xy: bool = True,
) -> NDArray[np.float64]:
    """Evaluate pairs in one CPU batch and return raw XY W increments.

    Args:
        model: CPU float32 surrogate in eval mode.
        velocity_world: Root O velocities [N,3], m/s.
        rotation: Proper B→W matrices [N,3,3].
        omega_body: Angular rates [N,3], rad/s.
        wind_world: Horizontal wind [N,3], within the model's wind domain.
        project_xy: Project here for offline labels (default); False returns
            complete raw W differences for runtime diagnostics. The app then
            performs its own unchanged XY projection before filtering.

    Returns:
        World acceleration [N,3], m/s²; exact zero at w=0. Z is exactly zero
        when project_xy=True. Raw Z never bypasses the application's projection.
        No EMA, force limiting, mass multiplication or runtime integration.
    """
    velocity_world, rotation, omega_body, wind_world = (
        np.asarray(value, dtype=np.float64)
        for value in (velocity_world, rotation, omega_body, wind_world)
    )
    count = len(velocity_world)
    if (velocity_world.shape != (count, 3) or wind_world.shape != (count, 3)
            or omega_body.shape != (count, 3) or rotation.shape != (count, 3, 3)
            or not all(np.isfinite(value).all() for value in
                       (velocity_world, rotation, omega_body, wind_world))):
        raise ValueError("Finite batch state shapes are required")
    wind_limit = model.wind_max_horizontal_m_s
    if (np.any(wind_world[:, 2] != 0)
            or np.any(np.hypot(wind_world[:, 0], wind_world[:, 1])
                      > np.nextafter(wind_limit, np.inf))):
        raise ValueError(f"Require horizontal wind <={wind_limit:g} m/s")
    if (not np.allclose(rotation.swapaxes(1, 2) @ rotation, np.eye(3), atol=1e-8)
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-8)):
        raise ValueError("Require proper body-to-world rotation")
    output = np.zeros((count, 3), dtype=np.float64)
    active = np.any(wind_world != 0, axis=1)
    if not np.any(active):
        return output
    relative = np.stack((velocity_world[active] - wind_world[active],
                         velocity_world[active]), axis=1)
    body_pair = np.einsum("nji,nkj->nki", rotation[active], relative)
    omega_pair = np.repeat(omega_body[active, None, :], 2, axis=1)
    acceleration = model(torch.tensor(body_pair.reshape(-1, 3), dtype=torch.float32),
                         torch.tensor(omega_pair.reshape(-1, 3), dtype=torch.float32))
    if not torch.isfinite(acceleration).all():
        raise ValueError("Nonfinite surrogate output; unsupported state magnitude")
    pair = acceleration.numpy().reshape(-1, 2, 3)
    delta = np.einsum("nij,nj->ni", rotation[active], pair[:, 0] - pair[:, 1])
    output[active] = delta
    if project_xy:
        output[:, 2] = 0.0
    return output


def load_checkpoint(path: Path) -> tuple[X4WindAeroV1, dict[str, Any]]:
    """Load a self-describing X4 v1 or v2 candidate, rejecting legacy kinds.

    Args:
        path: X4 six-dimensional checkpoint.

    Returns:
        Model in eval mode and checkpoint metadata (including state tensors).
    """
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("kind") not in (KIND, WIDE_KIND):
        raise ValueError("Unsupported checkpoint kind; expected X4 v1 or v2")
    if saved.get("feature_version") != "body_air_velocity_omega_6d_v1":
        raise ValueError("Unsupported feature version")
    if saved["kind"] == WIDE_KIND:
        wind_limit = saved.get("wind_max_horizontal_m_s")
        if type(wind_limit) not in (int, float) or wind_limit != WIDE_WIND_MAX_M_S:
            raise ValueError("X4 v2 wind_max_horizontal_m_s must be 10")
        reference = saved.get("reference_config")
        if not isinstance(reference, dict) or reference_wind_limit(reference) != wind_limit:
            raise ValueError("X4 v2 checkpoint wind domain differs from its embedded reference")
        validate_wide_inspection_domain(reference)
        if saved.get("state_domain") != reference["inspection_state_domain"]:
            raise ValueError("X4 v2 checkpoint state_domain differs from its embedded reference")
    else:
        wind_limit = LEGACY_WIND_MAX_M_S
        recorded = saved.get("wind_max_horizontal_m_s", wind_limit)
        if type(recorded) not in (int, float) or recorded != wind_limit:
            raise ValueError("X4 v1 wind_max_horizontal_m_s must be 3")
    stats = saved["normalization"]
    model = X4WindAeroV1(stats["input_mean"], stats["input_std"], stats["full_scale"],
                         saved["prior_coefficients"], wind_max_horizontal_m_s=wind_limit)
    model.load_state_dict(saved["state_dict"], strict=True)
    for name, expected in (("input_mean", stats["input_mean"]),
                            ("input_std", stats["input_std"]),
                            ("output_scale", stats["full_scale"]),
                            ("prior_coefficients", saved["prior_coefficients"])):
        if not torch.equal(getattr(model, name), torch.tensor(expected, dtype=torch.float32)):
            raise ValueError("Checkpoint normalization/prior disagrees with metadata")
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Nonfinite checkpoint")
    return model.eval(), saved
