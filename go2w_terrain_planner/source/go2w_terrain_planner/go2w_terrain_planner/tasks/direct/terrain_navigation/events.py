"""Vectorized episode randomization events."""

from __future__ import annotations


def sample_local_goals(robot_pose, minimum_distance_m: float, maximum_distance_m: float):
    """Sample world-frame local goals around each robot pose."""
    import torch

    if robot_pose.ndim != 2 or robot_pose.shape[-1] != 3:
        raise ValueError("robot_pose必须为[B,3]")
    if minimum_distance_m <= 0.0 or maximum_distance_m <= minimum_distance_m:
        raise ValueError("局部目标距离范围无效")
    count = robot_pose.shape[0]
    distance = minimum_distance_m + (maximum_distance_m - minimum_distance_m) * torch.rand(
        count, device=robot_pose.device
    )
    angle = -torch.pi + 2.0 * torch.pi * torch.rand(count, device=robot_pose.device)
    offset = torch.stack((distance * torch.cos(angle), distance * torch.sin(angle)), dim=-1)
    return robot_pose[:, :2] + offset
