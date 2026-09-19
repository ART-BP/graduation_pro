"""Configured full-revolution ray pattern for the XT16-style sensor."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from isaaclab.sensors.ray_caster import patterns
from isaaclab.utils import configclass

from go2w_terrain_planner.mapping.isaac_raycast_lidar import (
    build_spinning_lidar_pattern,
)


def xt16_pattern(cfg: "Xt16PatternCfg", device: str):
    """Generate one independent ray for every configured channel/azimuth."""
    return build_spinning_lidar_pattern(
        cfg.vertical_angles_deg,
        cfg.horizontal_resolution_deg,
        device,
    )


@configclass
class Xt16PatternCfg(patterns.PatternBaseCfg):
    """RayCaster pattern preserving an explicit per-channel elevation table."""

    func: Callable = xt16_pattern
    vertical_angles_deg: Sequence[float] = tuple(
        -15.0 + 2.0 * index for index in range(16)
    )
    horizontal_resolution_deg: float = 0.18
