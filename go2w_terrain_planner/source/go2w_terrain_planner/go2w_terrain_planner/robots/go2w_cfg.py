"""Stable phase boundary for replacing the kinematic proxy with Go2W physics."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Go2wIntegrationConfig:
    """Asset and ROS contract used by the future phase-two integration."""

    mode: str = "phase1_kinematic_proxy"
    asset_path: str = ""
    command_topic: str = "/cmd_vel"
    odometry_topic: str = "/Odometry"
    grid_map_topic: str = "/local_environment/grid_map"
    local_goal_topic: str = "/local_goal"

    def resolved_asset_path(self) -> Path | None:
        value = self.asset_path or os.environ.get("GO2W_ASSET_PATH", "")
        if not value:
            return None
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Go2W资产不存在：{path}")
        return path


GO2W_INTEGRATION = Go2wIntegrationConfig()
