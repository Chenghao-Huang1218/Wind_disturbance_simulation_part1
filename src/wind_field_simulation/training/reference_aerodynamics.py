"""Independent, additive component drag reference in SI units (float64).

Scalar drag magnitude follows D = 0.5 rho Cd A V² (NASA Glenn):
https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/drag-equation/
Applying it independently along each frozen proxy axis is our approximation,
not a calibrated X4-950 model or a MuJoCo fluid solver. No PINN dependencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
LEGACY_WIND_MAX_M_S = 3.0
WIDE_WIND_MAX_M_S = 10.0
INSPECTION_STATE_DOMAIN = {
    "root_horizontal_speed_max_m_s": 3.0,
    "root_vertical_velocity_m_s": [-1.0, 1.0],
    "roll_pitch_max_deg": 20.0,
    "yaw_deg": [-180.0, 180.0],
    "angular_speed_max_rad_s": 1.5,
}


def reference_wind_limit(reference: dict[str, Any]) -> float:
    """Read the supported horizontal magnitude from reference assumptions.

    Args:
        reference: Frozen reference with wind-domain assumptions.

    Returns:
        Maximum horizontal wind magnitude in m/s.

    Raises:
        ValueError: Unsupported or incomplete wind-domain metadata.
    """
    if not isinstance(reference, dict):
        raise ValueError("Reference config must be a JSON object")
    assumptions = reference.get("assumptions")
    if not isinstance(assumptions, dict):
        raise ValueError("Missing reference wind assumptions")
    maximum = assumptions.get("wind_max_horizontal_m_s")
    if (type(maximum) not in (int, float)
            or maximum not in (LEGACY_WIND_MAX_M_S, WIDE_WIND_MAX_M_S)):
        raise ValueError("Unsupported reference wind_max_horizontal_m_s")
    default_z = 0.0 if maximum == LEGACY_WIND_MAX_M_S else None
    for field in ("wind_z_m_s", "applied_force_z_n"):
        value = assumptions.get(field, default_z)
        if type(value) not in (int, float) or value != 0:
            raise ValueError(f"Reference {field} must be zero")
    return float(maximum)


def validate_wide_inspection_domain(reference: dict[str, Any]) -> None:
    """Check the recorded state envelope of a wide reference at startup.

    Args:
        reference: Wide artifact with a structured inspection state domain.

    Raises:
        ValueError: The envelope differs from the supported training protocol.
    """
    domain = reference.get("inspection_state_domain")
    if not isinstance(domain, dict):
        raise ValueError("Missing wide reference inspection_state_domain")
    for field, expected in INSPECTION_STATE_DOMAIN.items():
        value = domain.get(field)
        if isinstance(expected, list):
            valid = (isinstance(value, list) and value == expected
                     and all(type(bound) in (int, float) for bound in value))
        else:
            valid = type(value) in (int, float) and value == expected
        if not valid:
            raise ValueError(f"Unsupported wide reference inspection_state_domain.{field}")


def _vectors(value: ArrayLike, length: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if (array.ndim == 0 or array.shape[-1] != length
            or not np.all(np.isfinite(array))):
        raise ValueError(
            f"{name} must be finite with trailing dimension {length}"
        )
    return array


def _reference_arrays(
    reference: dict[str, Any],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Validate the frozen configuration before evaluating physical quantities."""
    assumptions = reference["assumptions"]
    density = float(assumptions["density_kg_m3"])
    cd = _vectors(assumptions["cd_axes"], 3, "Cd")
    if (not np.isfinite(density) or density < 0 or cd.shape != (3,)
            or np.any(cd < 0)):
        raise ValueError("Density and Cd must be finite and nonnegative")
    proxies = reference["proxies"]
    if not proxies:
        raise ValueError("Reference needs at least one proxy")
    centers = _vectors(
        [item["center_body_m"] for item in proxies], 3, "proxy centers"
    )
    rotations = np.asarray(
        [item["rotation_body_from_proxy"] for item in proxies], dtype=float
    )
    areas = _vectors([item["areas_m2"] for item in proxies], 3, "proxy areas")
    count = len(proxies)
    if (centers.shape != (count, 3) or areas.shape != (count, 3)
            or np.any(areas <= 0)):
        raise ValueError("Proxy centers/areas must be [components,3], areas positive")
    if (rotations.shape != (count, 3, 3) or not np.all(np.isfinite(rotations))
            or not np.allclose(
                rotations.swapaxes(-1, -2) @ rotations, np.eye(3),
                atol=1e-10, rtol=0,
            )
            or not np.allclose(
                np.linalg.det(rotations), 1, atol=1e-10, rtol=0,
            )):
        raise ValueError("Proxy rotations must be proper orthonormal matrices")
    return centers, rotations, 0.5 * density * cd * areas


def reference_wrench_body(
    air_velocity_body: ArrayLike,
    angular_velocity_body: ArrayLike,
    reference: dict[str, Any],
) -> tuple[FloatArray, FloatArray]:
    """Evaluate the full rigid, frozen-geometry drag wrench about root O.

    Args:
        air_velocity_body: Root O velocity relative to air (v_O - w), in B,
            m/s; shape (...,3), never the CoM velocity unless O equals CoM.
        angular_velocity_body: Angular velocity in B, rad/s, broadcastable
            (...,3). Frozen arm joint velocities and visual spin are excluded.
        reference: Exported proxy centers r_j, rotations Q_j, axis areas,
            density and Cd assumptions.

    Returns:
        Force in B (N), and diagnostic moment about O in B (N m), each
        (...,3). No projection, EMA, limiting, gravity or thrust is included.

    Raises:
        ValueError: Shapes, configuration or finite arithmetic are invalid.
    """
    velocity, omega = np.broadcast_arrays(
        _vectors(air_velocity_body, 3, "air velocity"),
        _vectors(angular_velocity_body, 3, "angular velocity"))
    centers, rotations, coefficients = _reference_arrays(reference)
    try:
        with np.errstate(over="raise", invalid="raise"):
            point_velocity = velocity[..., None, :] + np.cross(
                omega[..., None, :], centers
            )
            # Q maps proxy axes to B: local = Q^T point_velocity.
            local = np.einsum("cji,...cj->...ci", rotations, point_velocity)
            local_force = -coefficients * local * np.abs(local)
            body_forces = np.einsum("cji,...ci->...cj", rotations, local_force)
            force = np.sum(body_forces, axis=-2)
            torque = np.sum(np.cross(centers, body_forces), axis=-2)
    except FloatingPointError as error:
        raise ValueError(
            "Reference arithmetic overflow; unsupported state magnitude"
        ) from error
    return force, torque


def _body_to_world(quaternion: FloatArray) -> FloatArray:
    """Convert unit WXYZ body-to-world quaternions to rotation matrices."""
    norm = np.linalg.norm(quaternion, axis=-1)
    if not np.allclose(norm, 1, atol=1e-8, rtol=0):
        raise ValueError("quaternion_wxyz must be unit length")
    scalar, axis_x, axis_y, axis_z = np.moveaxis(quaternion, -1, 0)
    entries = (
        1 - 2 * (axis_y**2 + axis_z**2),
        2 * (axis_x * axis_y - scalar * axis_z),
        2 * (axis_x * axis_z + scalar * axis_y),
        2 * (axis_x * axis_y + scalar * axis_z),
        1 - 2 * (axis_x**2 + axis_z**2),
        2 * (axis_y * axis_z - scalar * axis_x),
        2 * (axis_x * axis_z - scalar * axis_y),
        2 * (axis_y * axis_z + scalar * axis_x),
        1 - 2 * (axis_x**2 + axis_y**2),
    )
    return np.stack(entries, axis=-1).reshape(
        quaternion.shape[:-1] + (3, 3)
    )


def paired_wind_labels(
    velocity_world: ArrayLike,
    quaternion_wxyz: ArrayLike,
    angular_velocity_body: ArrayLike,
    wind_world: ArrayLike,
    reference: dict[str, Any],
) -> dict[str, FloatArray]:
    """Evaluate same-state full wind/no-wind forces and horizontal increment.

    Args:
        velocity_world: Root O velocity in W, m/s, shape (...,3).
        quaternion_wxyz: Unit quaternion, B to W, shape (...,4).
        angular_velocity_body: Angular velocity in B, rad/s, shape (...,3).
        wind_world: Air velocity in W, m/s, shape (...,3); w_z=0 and
            horizontal magnitude within the reference domain. Other arrays
            may broadcast.
        reference: Frozen geometry and assumptions; total assembly mass kg.

    Returns:
        Complete three-axis forces (N) and moments about O (N m) in W for
        both evaluations, their full force difference, and P_xy delta_F/m
        (m/s²). Zero wind yields exactly zero labels; label Z is exactly zero.
        Moments are diagnostics and are never applied by this offline tool.

    Raises:
        ValueError: Invalid mass, state, quaternion, wind domain or arithmetic.
    """
    velocity = _vectors(velocity_world, 3, "velocity_world")
    quaternion = _vectors(quaternion_wxyz, 4, "quaternion_wxyz")
    omega = _vectors(angular_velocity_body, 3, "angular_velocity_body")
    wind = _vectors(wind_world, 3, "wind_world")
    mass = float(reference["mass_total_kg"])
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Total assembly mass must be finite and positive")
    wind_limit = reference_wind_limit(reference)
    wind_speed = np.hypot(wind[..., 0], wind[..., 1])
    if (np.any(wind[..., 2] != 0)
            or np.any(wind_speed > np.nextafter(wind_limit, np.inf))):
        raise ValueError(f"Require horizontal wind with resultant speed <={wind_limit:g} m/s")
    shape = np.broadcast_shapes(
        velocity.shape[:-1], quaternion.shape[:-1],
        omega.shape[:-1], wind.shape[:-1],
    )
    velocity, wind, omega = (
        np.broadcast_to(value, shape + (3,))
        for value in (velocity, wind, omega)
    )
    quaternion = np.broadcast_to(quaternion, shape + (4,))
    rotation = _body_to_world(quaternion)
    # Evaluate both states in one batch with identical geometry and angular rate.
    relative_world = np.stack((velocity - wind, velocity), axis=-2)
    relative_body = np.einsum("...ji,...kj->...ki", rotation, relative_world)
    force_body, torque_body = reference_wrench_body(
        relative_body, omega[..., None, :], reference
    )
    force_world = np.einsum("...ij,...kj->...ki", rotation, force_body)
    torque_world = np.einsum("...ij,...kj->...ki", rotation, torque_body)
    difference = force_world[..., 0, :] - force_world[..., 1, :]
    acceleration = difference / mass
    acceleration[..., 2] = 0.0
    # Explicit same-state zero guarantee, including signed floating zeros.
    acceleration = np.where(np.all(wind == 0, axis=-1)[..., None], 0.0, acceleration)
    return {"force_wind_world_n": force_world[..., 0, :],
            "force_no_wind_world_n": force_world[..., 1, :],
            "torque_wind_world_nm": torque_world[..., 0, :],
            "torque_no_wind_world_nm": torque_world[..., 1, :],
            "delta_force_world_n": difference,
            "wind_increment_acceleration_world_m_s2": acceleration}
