"""Shared X4-950 drag groups and reproducible run-level draws."""

from typing import Any

import numpy as np

DRAW_GROUPS = ("frame", "booms_and_motors", "arm")


def draw_drag_scales(variation: float, seed: int) -> dict[str, float]:
    """Draw ordered physical multipliers uniformly in [1−variation, 1+variation]."""
    if not 0 <= variation < 1 or seed < 0:
        raise ValueError("Drag variation must be in [0, 1) with a nonnegative seed")
    generator = np.random.default_rng([seed, 0])
    values = generator.uniform(1.0 - variation, 1.0 + variation, len(DRAW_GROUPS))
    return dict(zip(DRAW_GROUPS, values.tolist(), strict=True))


def drag_group(proxy: dict[str, Any]) -> str:
    """Return the simulation drag group for a frozen aerodynamic proxy."""
    if proxy["mass_owner"] != "quadrotor_950":
        return "arm"
    if proxy["id"].startswith(("collision-arm-", "collision-motor-")):
        return "booms_and_motors"
    return "frame"


def with_drag_scales(reference: dict[str, Any], scales: dict[str, float]) -> dict[str, Any]:
    """Return analytic evaluation geometry with each proxy area scaled once."""
    proxies = []
    for proxy in reference["proxies"]:
        scale = scales[drag_group(proxy)]
        proxies.append({**proxy, "areas_m2": [area * scale for area in proxy["areas_m2"]]})
    return {**reference, "proxies": proxies}
