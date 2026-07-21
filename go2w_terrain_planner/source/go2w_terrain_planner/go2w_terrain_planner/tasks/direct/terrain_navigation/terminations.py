"""Termination checks shared by the task and unit tests."""

from __future__ import annotations


def terrain_failure_state(
    collision_height_m,
    hazard_height_m,
    entry_alignment,
    commanded_linear_speed,
    actual_linear_speed,
    *,
    collision_height_threshold_m: float,
    maximum_command_speed_mps: float,
    unstable_risk_threshold: float,
    terrain_tilt_gain_rad: float,
    entry_tilt_gain: float,
    maximum_tilt_rad: float,
    fall_tilt_rad: float,
):
    """Return monotonic terrain risk, tilt, collision, fall and instability flags."""
    import torch

    if collision_height_threshold_m <= 0.0 or maximum_command_speed_mps <= 0.0:
        raise ValueError("碰撞高度和最大指令速度必须大于0")
    if not 0.0 <= unstable_risk_threshold <= 1.0:
        raise ValueError("失稳风险阈值必须位于[0,1]")
    if terrain_tilt_gain_rad <= 0.0 or entry_tilt_gain < 0.0:
        raise ValueError("地形倾斜增益无效")
    risk = torch.clamp(hazard_height_m / collision_height_threshold_m, 0.0, 1.0)
    speed_ratio = torch.clamp(
        torch.abs(commanded_linear_speed) / maximum_command_speed_mps, 0.0, 1.0
    )
    alignment_penalty = 1.0 + entry_tilt_gain * (1.0 - torch.clamp(entry_alignment, 0.0, 1.0))
    active_risk = torch.where(risk >= unstable_risk_threshold, risk, torch.zeros_like(risk))
    tilt_angle = terrain_tilt_gain_rad * active_risk * speed_ratio * alignment_penalty
    moving = torch.maximum(
        torch.abs(commanded_linear_speed), torch.abs(actual_linear_speed)
    ) > 0.05
    collision = (collision_height_m >= collision_height_threshold_m) & moving
    fallen = tilt_angle >= fall_tilt_rad
    unstable = fallen | (tilt_angle >= maximum_tilt_rad)
    return risk, tilt_angle, collision, fallen, unstable


def termination_flags(
    distance,
    collision,
    unstable,
    stuck_time,
    position,
    bad_observation_steps,
    *,
    goal_tolerance_m: float,
    stuck_timeout_s: float,
    maximum_distance_m: float,
    maximum_bad_observation_steps: int,
):
    import torch

    reached = distance <= goal_tolerance_m
    stuck = stuck_time >= stuck_timeout_s
    out_of_bounds = torch.linalg.vector_norm(position, dim=-1) >= maximum_distance_m
    observation_failure = bad_observation_steps >= maximum_bad_observation_steps
    terminated = reached | collision | unstable | stuck | out_of_bounds | observation_failure
    return terminated, reached, stuck, out_of_bounds, observation_failure
