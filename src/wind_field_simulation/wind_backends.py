"""Runtime adapters sharing the existing force-only wind increment pipeline."""

import json
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np
import torch
from numpy.typing import NDArray

from .training.reference_aerodynamics import paired_wind_labels, reference_wind_limit
from .training.dataset_v1 import file_sha256
from .training.model_v1 import load_checkpoint, predict_increment
from .training.model_v3 import (
    KIND as RANDOMIZED_KIND, MAX_TRAINED_VARIATION, FixedDragWindAero,
    load_randomized_checkpoint,
)
from .wind_randomization import draw_drag_scales, with_drag_scales
from .training.sampling import body_to_world_rotation
from .training.xml_parameters import load_reference_bundle

FloatArray = NDArray[np.float64]
X4_WIDE_WIND_CDA_M2 = 0.25


def reference_horizontal_drag_area_m2(reference: dict[str, Any]) -> float:
    """Compute the reference drag area at the stationary, level diagonal anchor.

    Args:
        reference: Validated wide reference with its frozen geometry and density.

    Returns:
        Along-wind force divided by dynamic pressure at [4, 4, 0] m/s.
    """
    wind = np.array([4.0, 4.0, 0.0])
    speed = np.linalg.norm(wind)
    force = paired_wind_labels(
        [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], wind, reference,
    )["delta_force_world_n"]
    density = reference["assumptions"]["density_kg_m3"]
    return float((force @ (wind / speed)) / (0.5 * density * speed**2))


class AnalyticalWindResidualModel:
    """Same-state wind increment; shares the app's existing force pipeline."""

    def __init__(
        self, reference_path: Path, xml_path: Path, model: mujoco.MjModel,
        *, drag_variation: float = 0.0, wind_seed: int = 42,
    ) -> None:
        """Validate the frozen bundle against the compiled runtime assembly.

        Args:
            reference_path: Exported, fingerprinted reference_config.json.
            xml_path: Actual scene used by the application.
            model: Already compiled runtime model, checked without mutation.
            drag_variation: Fractional half-width of each group's drag multiplier.
            wind_seed: Seed for the fixed drag draw for this run.
        """
        self.reference = load_reference_bundle(reference_path, xml_path)
        self.reference_sha256 = self.reference["reference_sha256"]
        self.mass_total_kg = float(self.reference["mass_total_kg"])
        self.wind_max_horizontal_m_s = reference_wind_limit(self.reference)
        # Check individual masses, not cached body_subtreemass after mutation.
        vehicle_path = reference_path.parent / "vehicle_parameters.json"
        vehicle = json.loads(vehicle_path.read_text())
        for body in vehicle["bodies"]:
            actual = model.body_mass[model.body(body["name"]).id]
            if not np.isclose(actual, body["mass_kg"], rtol=0, atol=1e-12):
                raise ValueError("Runtime model mass differs from reference")
        for name, joint in self.reference["initial_joints"].items():
            start = int(model.jnt_qposadr[model.joint(name).id])
            expected = np.asarray(joint["qpos"])
            actual = model.qpos0[start:start + expected.size]
            if not np.allclose(actual, expected, rtol=0, atol=1e-12):
                raise ValueError("Runtime model initial joints differ")
        if model.opt.density != 0 or model.opt.viscosity != 0:
            raise ValueError("Runtime fluid forces must stay disabled")
        self.drag_group_scales: dict[str, float] = {}
        self._evaluation_reference = self.reference
        if drag_variation:
            self.drag_group_scales = draw_drag_scales(drag_variation, wind_seed)
            self._evaluation_reference = with_drag_scales(self.reference, self.drag_group_scales)
        # Validate proxy arrays and physical assumptions before connecting PX4.
        self.predict([0, 0, 0], [1, 0, 0, 0], [0, 0, 0], [0, 0, 0])

    def predict(
        self,
        velocity_world: Sequence[float],
        quaternion_wxyz: Sequence[float],
        angular_velocity_body: Sequence[float],
        wind_world: Sequence[float],
    ) -> FloatArray:
        """Return raw world-frame wind acceleration increment in m/s².

        Args:
            velocity_world: Root O velocity in world coordinates, m/s.
            quaternion_wxyz: Unit body-to-world WXYZ quaternion.
            angular_velocity_body: Body angular rate, rad/s.
            wind_world: World horizontal air velocity within the reference domain.

        Returns:
            Complete raw same-state delta force divided by total mass.
            The caller performs XY projection, EMA and limiting. No moment,
            no no-wind drag, and no live arm geometry is additionally applied.
        """
        for name, values, size in (
            ("velocity", velocity_world, 3),
            ("quaternion", quaternion_wxyz, 4),
            ("angular rate", angular_velocity_body, 3),
            ("wind", wind_world, 3),
        ):
            if np.asarray(values).shape != (size,):
                raise ValueError(f"{name} must have shape ({size},)")
        paired = paired_wind_labels(
            velocity_world, quaternion_wxyz, angular_velocity_body,
            wind_world, self._evaluation_reference,
        )
        return paired["delta_force_world_n"] / self.mass_total_kg


class HorizontalGust:
    """Seeded horizontal pulses with a raised-cosine envelope.

    Args:
        amplitude_m_s: Peak magnitude of each gust before wind-domain limiting.
        period_s: Simulation seconds per pulse.
        seed: Experiment seed, independent of the drag-parameter draw.
    """

    def __init__(self, amplitude_m_s: float, period_s: float, seed: int) -> None:
        self.amplitude_m_s = amplitude_m_s
        self.period_s = period_s
        self.seed = seed
        self._pulse = -1
        self._peak_xy = np.zeros(2, dtype=np.float64)

    def velocity(
        self, base_wind: FloatArray, elapsed_s: float, wind_limit_m_s: float,
    ) -> FloatArray:
        """Return the horizontal sum of the baseline and the current pulse.

        Args:
            base_wind: Accepted world-frame wind vector, m/s.
            elapsed_s: Simulation time since the wind was enabled.
            wind_limit_m_s: Maximum resultant horizontal wind, m/s.

        Returns:
            World-frame wind with a bounded horizontal resultant and zero Z.
        """
        pulse, offset = divmod(max(0.0, elapsed_s), self.period_s)
        if pulse != self._pulse:
            self._pulse = int(pulse)
            angle = np.random.default_rng([self.seed, 1, self._pulse]).uniform(0, 2 * np.pi)
            self._peak_xy = self.amplitude_m_s * np.array([np.cos(angle), np.sin(angle)])
        wind = base_wind.copy()
        wind[:2] += self._peak_xy * np.sin(np.pi * offset / self.period_s) ** 2
        speed = np.hypot(*wind[:2])
        if speed > wind_limit_m_s:
            wind[:2] *= np.nextafter(wind_limit_m_s, 0.0) / speed
        return wind


class X4PinnWindResidualModel:
    """Validated CPU surrogate adapter for versioned X4 models.

    Args:
        checkpoint_path: An X4 v1, v2 or v3 candidate.
        reference_path: Frozen reference bundle used for the candidate labels.
        xml_path: Actual scene used by the app.
        model: Compiled runtime model, checked without modifying its state.
    """

    def __init__(
        self, checkpoint_path: Path, reference_path: Path,
        xml_path: Path, model: mujoco.MjModel,
        *, drag_variation: float = 0.0, wind_seed: int = 42,
    ) -> None:
        # Reuse the same XML closure, mass, initial joints and fluid checks as
        # the analytic runtime. No reference force is added during inference.
        validated = AnalyticalWindResidualModel(reference_path, xml_path, model)
        metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.drag_group_scales: dict[str, float] = {}
        if isinstance(metadata, dict) and metadata.get("kind") == RANDOMIZED_KIND:
            if drag_variation > MAX_TRAINED_VARIATION:
                raise ValueError("X4 v3 supports --drag-variation up to 0.20")
            self.drag_group_scales = draw_drag_scales(drag_variation, wind_seed)
            self.network = FixedDragWindAero(
                load_randomized_checkpoint(metadata), self.drag_group_scales,
            ).eval()
        else:
            self.network, metadata = load_checkpoint(checkpoint_path)
        self.wind_max_horizontal_m_s = self.network.wind_max_horizontal_m_s
        if self.wind_max_horizontal_m_s != validated.wind_max_horizontal_m_s:
            raise ValueError("X4 checkpoint wind limit differs from runtime reference")
        vehicle = json.loads(
            (reference_path.parent / "vehicle_parameters.json").read_text())
        expected = {
            "reference_sha256": validated.reference_sha256,
            "reference_config": validated.reference,
            "mass_total_kg": validated.mass_total_kg,
            "arm_pose": validated.reference["initial_joints"],
            "xml_mesh_dependencies": vehicle["dependencies"],
            "coordinates_units": (
                "B FLU at root O; W ENU-like; WXYZ B→W; m/s, rad/s, m/s²"),
            "wind_domain": (
                f"horizontal resultant 0–{self.wind_max_horizontal_m_s:g} m/s; "
                "fixed initial arm geometry"),
        }
        for field, value in expected.items():
            if metadata.get(field) != value:
                raise ValueError(f"X4 checkpoint {field} differs from runtime reference")
        for field in ("dataset_manifest_sha256", "dataset_archive_sha256"):
            digest = metadata.get(field)
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)):
                raise ValueError(f"X4 checkpoint has invalid {field}")
        self.reference_sha256 = validated.reference_sha256
        self.reference = validated.reference
        self.mass_total_kg = validated.mass_total_kg
        self.checkpoint_sha256 = file_sha256(checkpoint_path)
        self.dataset_manifest_sha256 = metadata["dataset_manifest_sha256"]
        self.dataset_archive_sha256 = metadata["dataset_archive_sha256"]
        self.inference_device = "cpu"
        self.inference_dtype = "float32"
        # Small batches are faster and reproducible without a large BLAS pool.
        # Set once in this new backend's owning process; legacy path unchanged.
        torch.set_num_threads(1)
        self.torch_threads = torch.get_num_threads()
        self.predict([0, 0, 0], [1, 0, 0, 0], [0, 0, 0], [0, 0, 0])

    def predict(
        self, velocity_world: Sequence[float], quaternion_wxyz: Sequence[float],
        angular_velocity_body: Sequence[float], wind_world: Sequence[float],
    ) -> FloatArray:
        """Return raw full W wind increment; leave XY/EMA/limit to the app.

        Args:
            velocity_world: Root O velocity [3], m/s.
            quaternion_wxyz: Unit B→W WXYZ [4].
            angular_velocity_body: Current body angular rate [3], rad/s.
            wind_world: World horizontal wind [3] within the checkpoint domain.

        Returns:
            Raw three-axis acceleration difference [3], m/s². Zero wind is
            exactly zero. No no-wind baseline drag or torque is applied.
        """
        values = []
        for name, value, size in (
            ("velocity", velocity_world, 3), ("quaternion", quaternion_wxyz, 4),
            ("angular rate", angular_velocity_body, 3), ("wind", wind_world, 3),
        ):
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (size,) or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite with shape ({size},)")
            values.append(array[None, :])
        velocity, quaternion, omega, wind = values
        rotation = body_to_world_rotation(quaternion)
        return predict_increment(self.network, velocity, rotation, omega, wind,
                                  project_xy=False)[0]
