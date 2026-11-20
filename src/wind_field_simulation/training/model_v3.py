"""Group-conditioned quadratic prior with learned body-frame residuals."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor, nn

from ..wind_randomization import DRAW_GROUPS
from .reference_aerodynamics import reference_wind_limit, validate_wide_inspection_domain

KIND = "x4_950_wind_increment_xy_v3"
FEATURE_VERSION = "body_air_velocity_omega_drag_groups_9d_v3"
MAX_TRAINED_VARIATION = .2


class X4WindAeroV3(nn.Module):
    """Predict acceleration from six kinematic inputs and three physical scales.

    Each unit-multiplier group has a quadratic drag prior and a learned 3D
    residual. Their weighted sum enforces the analytic teacher's linearity in
    proxy area (F = -0.5 rho Cd A u |u| along each proxy axis).
    """

    wind_max_horizontal_m_s = 10.0

    def __init__(
        self, input_mean: ArrayLike, input_std: ArrayLike,
        group_scale: ArrayLike, prior_coefficients: ArrayLike,
    ) -> None:
        """Initialize from training-only state/target scales and teacher priors."""
        super().__init__()
        for name, value, shape in (
            ("input_mean", input_mean, (6,)), ("input_std", input_std, (6,)),
            ("group_scale", group_scale, (3, 3)),
            ("prior_coefficients", prior_coefficients, (3, 3)),
        ):
            array = np.asarray(value, dtype=np.float32)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"Invalid {name}")
            if name in ("input_std", "group_scale") and np.any(array <= 0):
                raise ValueError(f"Nonpositive {name}")
            if name == "prior_coefficients" and np.any(array < 0):
                raise ValueError("Prior must be nonnegative")
            self.register_buffer(name, torch.from_numpy(array.copy()))

        self.residual = nn.Sequential(
            nn.Linear(6, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 9),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def group_acceleration(self, air_velocity_body: Tensor, omega_body: Tensor) -> Tensor:
        """Return unit-scale acceleration [N, group, B-axis], in m/s²."""
        state = torch.cat((air_velocity_body, omega_body), dim=-1)
        quadratic = -air_velocity_body * air_velocity_body.abs()
        residual = self.residual((state - self.input_mean) / self.input_std)
        return (self.prior_coefficients * quadratic[:, None, :]
                + self.group_scale * residual.reshape(-1, 3, 3))

    def forward(self, air_velocity_body: Tensor, omega_body: Tensor, drag_scales: Tensor) -> Tensor:
        """Combine [N,3] group multipliers with each state's group accelerations."""
        return (self.group_acceleration(air_velocity_body, omega_body)
                * drag_scales[..., None]).sum(dim=1)


class FixedDragWindAero(nn.Module):
    """Adapt v3 to the existing pair predictor with one fixed draw per run."""

    wind_max_horizontal_m_s = X4WindAeroV3.wind_max_horizontal_m_s

    def __init__(self, network: X4WindAeroV3, drag_scales: dict[str, float]) -> None:
        super().__init__()
        self.network = network
        self.register_buffer("drag_scales", torch.tensor(
            [drag_scales[group] for group in DRAW_GROUPS], dtype=torch.float32,
        ))

    def forward(self, air_velocity_body: Tensor, omega_body: Tensor) -> Tensor:
        """Return full acceleration for both branches using the same multipliers."""
        return self.network(air_velocity_body, omega_body, self.drag_scales)


def load_randomized_checkpoint(saved: dict[str, Any]) -> X4WindAeroV3:
    """Validate v3's feature order, physical domain and trained tensors."""
    if (saved.get("kind") != KIND or saved.get("feature_version") != FEATURE_VERSION
            or saved.get("drag_group_order") != list(DRAW_GROUPS)
            or saved.get("max_trained_variation") != MAX_TRAINED_VARIATION
            or saved.get("wind_max_horizontal_m_s") != 10.0):
        raise ValueError("Unsupported X4 v3 feature or wind/drag domain")
    reference = saved.get("reference_config")
    if not isinstance(reference, dict) or reference_wind_limit(reference) != 10.0:
        raise ValueError("X4 v3 requires an embedded 0–10 m/s reference")
    validate_wide_inspection_domain(reference)
    if saved.get("state_domain") != reference["inspection_state_domain"]:
        raise ValueError("X4 v3 checkpoint state domain differs from reference")
    normalization = saved["normalization"]
    buffers = {**normalization, "prior_coefficients": saved["prior_coefficients"]}
    network = X4WindAeroV3(**buffers)
    network.load_state_dict(saved["state_dict"], strict=True)
    for name, expected in buffers.items():
        if not torch.equal(getattr(network, name), torch.tensor(expected, dtype=torch.float32)):
            raise ValueError("X4 v3 checkpoint normalization/prior disagrees with metadata")
    if not all(torch.isfinite(value).all() for value in network.state_dict().values()):
        raise ValueError("Nonfinite X4 v3 checkpoint")
    return network.eval()
