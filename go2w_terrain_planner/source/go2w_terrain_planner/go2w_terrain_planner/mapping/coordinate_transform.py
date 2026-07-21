"""SE(2) coordinate transforms used by simulation and the future ROS adapter."""

from __future__ import annotations

import math


def wrap_angle(angle):
    """Wrap NumPy scalars/arrays or torch tensors to ``[-pi, pi)``."""
    try:
        import torch

        if isinstance(angle, torch.Tensor):
            return torch.atan2(torch.sin(angle), torch.cos(angle))
    except ImportError:
        pass
    import numpy as np

    return np.arctan2(np.sin(angle), np.cos(angle))


def encode_local_goal(robot_pose, goal_xy, maximum_distance: float):
    """Encode world-frame goals as ``[normalized_distance, sin, cos]``."""
    import torch

    if maximum_distance <= 0.0:
        raise ValueError("maximum_distance必须大于0")
    pose = torch.as_tensor(robot_pose, dtype=torch.float32)
    goal = torch.as_tensor(goal_xy, dtype=torch.float32, device=pose.device)
    if pose.shape[-1] != 3 or goal.shape[-1] != 2 or pose.shape[:-1] != goal.shape[:-1]:
        raise ValueError("robot_pose和goal_xy形状必须分别为[...,3]和[...,2]")
    delta = goal - pose[..., :2]
    distance = torch.linalg.vector_norm(delta, dim=-1)
    bearing_world = torch.atan2(delta[..., 1], delta[..., 0])
    bearing_robot = wrap_angle(bearing_world - pose[..., 2])
    return torch.stack(
        (torch.clamp(distance / maximum_distance, 0.0, 1.0), torch.sin(bearing_robot), torch.cos(bearing_robot)),
        dim=-1,
    )


def relative_pose_deltas(poses):
    """Return adjacent SE(2) motion expressed in each previous body frame."""
    import torch

    if poses.ndim != 3 or poses.shape[-1] != 3:
        raise ValueError("poses形状必须为[B,T,3]")
    world_delta = poses[:, 1:, :2] - poses[:, :-1, :2]
    yaw = poses[:, :-1, 2]
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    dx = cosine * world_delta[..., 0] + sine * world_delta[..., 1]
    dy = -sine * world_delta[..., 0] + cosine * world_delta[..., 1]
    dyaw = wrap_angle(poses[:, 1:, 2] - poses[:, :-1, 2])
    return torch.stack((dx, dy, dyaw), dim=-1)


def warp_map_sequence(
    maps,
    source_poses,
    target_pose,
    extent_m: float,
    *,
    source_ground_reference_z=None,
    target_ground_reference_z=None,
    normalized_ground_height_scale_m: float | None = None,
):
    """Warp robot-centric historical maps into the current robot frame.

    Args:
        maps: ``[B,T,4,H,W]`` maps expressed in each source robot frame.
        source_poses: Corresponding world-frame ``[B,T,3]`` poses.
        target_pose: Current world-frame ``[B,3]`` pose.
        extent_m: Physical square-map side length.
        source_ground_reference_z: Absolute supporting-ground reference of each
            source map, shaped ``[B,T]``.
        target_ground_reference_z: Absolute supporting-ground reference of the
            target frame, shaped ``[B]``.
        normalized_ground_height_scale_m: Meter scale used to normalize channel
            zero. Required when height references are supplied.
    """
    import torch
    import torch.nn.functional as functional

    if maps.ndim != 5 or maps.shape[2] != 4:
        raise ValueError("maps形状必须为[B,T,4,H,W]")
    if source_poses.shape != (maps.shape[0], maps.shape[1], 3) or target_pose.shape != (maps.shape[0], 3):
        raise ValueError("地图和位姿批次尺寸不一致")
    if extent_m <= 0.0:
        raise ValueError("extent_m必须大于0")
    has_source_reference = source_ground_reference_z is not None
    has_target_reference = target_ground_reference_z is not None
    if has_source_reference != has_target_reference:
        raise ValueError("source和target高度参考必须同时提供")
    if has_source_reference:
        source_ground_reference_z = torch.as_tensor(
            source_ground_reference_z, dtype=maps.dtype, device=maps.device
        )
        target_ground_reference_z = torch.as_tensor(
            target_ground_reference_z, dtype=maps.dtype, device=maps.device
        )
        if source_ground_reference_z.shape != maps.shape[:2]:
            raise ValueError("source_ground_reference_z形状必须为[B,T]")
        if target_ground_reference_z.shape != (maps.shape[0],):
            raise ValueError("target_ground_reference_z形状必须为[B]")
        if normalized_ground_height_scale_m is None or normalized_ground_height_scale_m <= 0.0:
            raise ValueError("提供高度参考时必须给出有效的归一化高度尺度")

    batch, sequence, _, height, width = maps.shape
    device = maps.device
    dtype = maps.dtype
    x_axis = torch.linspace(-0.5 * extent_m, 0.5 * extent_m, height, device=device, dtype=dtype)
    y_axis = torch.linspace(-0.5 * extent_m, 0.5 * extent_m, width, device=device, dtype=dtype)
    target_x, target_y = torch.meshgrid(x_axis, y_axis, indexing="ij")
    target_x = target_x.view(1, 1, height, width)
    target_y = target_y.view(1, 1, height, width)

    target_yaw = target_pose[:, 2].view(batch, 1, 1, 1)
    world_x = (
        target_pose[:, 0].view(batch, 1, 1, 1)
        + torch.cos(target_yaw) * target_x
        - torch.sin(target_yaw) * target_y
    )
    world_y = (
        target_pose[:, 1].view(batch, 1, 1, 1)
        + torch.sin(target_yaw) * target_x
        + torch.cos(target_yaw) * target_y
    )

    source_yaw = source_poses[..., 2].view(batch, sequence, 1, 1)
    delta_x = world_x - source_poses[..., 0].view(batch, sequence, 1, 1)
    delta_y = world_y - source_poses[..., 1].view(batch, sequence, 1, 1)
    source_x = torch.cos(source_yaw) * delta_x + torch.sin(source_yaw) * delta_y
    source_y = -torch.sin(source_yaw) * delta_x + torch.cos(source_yaw) * delta_y

    grid_column = 2.0 * source_y / extent_m
    grid_row = 2.0 * source_x / extent_m
    grid = torch.stack((grid_column, grid_row), dim=-1).reshape(batch * sequence, height, width, 2)
    flat = maps.reshape(batch * sequence, 4, height, width)
    validity = flat[:, 3:4]
    warped_weight = functional.grid_sample(
        validity, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    weighted_ground = functional.grid_sample(
        flat[:, 0:1] * validity, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    ground = torch.where(
        warped_weight > 1.0e-6,
        weighted_ground / torch.clamp(warped_weight, min=1.0e-6),
        torch.zeros_like(weighted_ground),
    )
    nearest_validity = functional.grid_sample(
        validity, grid, mode="nearest", padding_mode="zeros", align_corners=True
    )
    height_range = functional.grid_sample(
        flat[:, 1:2], grid, mode="nearest", padding_mode="zeros", align_corners=True
    )
    height_range = torch.where(
        nearest_validity > 0.5, height_range, torch.zeros_like(height_range)
    )
    masks = functional.grid_sample(flat[:, 2:], grid, mode="nearest", padding_mode="zeros", align_corners=True)
    result = torch.cat((ground, height_range, masks), dim=1).reshape(batch, sequence, 4, height, width)
    # Preserve frames that are already expressed in the target frame exactly.
    # This avoids amplified round-off when a synthetic validity channel is very
    # close to zero.
    identity = torch.isclose(
        source_poses,
        target_pose[:, None, :],
        rtol=0.0,
        atol=1.0e-7,
    ).all(dim=-1)
    result = torch.where(identity[:, :, None, None, None], maps, result)
    if has_source_reference:
        vertical_offset = (
            source_ground_reference_z - target_ground_reference_z[:, None]
        ) / normalized_ground_height_scale_m
        shifted_ground = torch.clamp(
            result[:, :, 0] + vertical_offset[:, :, None, None], min=-1.0, max=1.0
        )
        result[:, :, 0] = torch.where(
            result[:, :, 3] > 0.5, shifted_ground, result[:, :, 0]
        )
    if not torch.isfinite(result).all():
        raise RuntimeError("地图对齐结果包含NaN或Inf")
    return result


def world_aligned_map_to_robot_frame(world_aligned_map, robot_yaw, extent_m: float):
    """Rotate an odom-axis Grid Map into the robot-centric convention used by the Actor."""
    import torch

    maps = torch.as_tensor(world_aligned_map, dtype=torch.float32)
    squeeze = maps.ndim == 3
    if squeeze:
        maps = maps.unsqueeze(0)
    if maps.ndim != 4 or maps.shape[1] != 4:
        raise ValueError("world_aligned_map必须为[4,H,W]或[B,4,H,W]")
    yaw = torch.as_tensor(robot_yaw, dtype=maps.dtype, device=maps.device)
    if yaw.ndim == 0:
        yaw = yaw.expand(maps.shape[0])
    if yaw.shape != (maps.shape[0],):
        raise ValueError("robot_yaw必须是标量或[B]")
    source_pose = torch.zeros((maps.shape[0], 1, 3), dtype=maps.dtype, device=maps.device)
    target_pose = torch.zeros((maps.shape[0], 3), dtype=maps.dtype, device=maps.device)
    target_pose[:, 2] = yaw
    result = warp_map_sequence(maps[:, None], source_pose, target_pose, extent_m)[:, 0]
    return result[0] if squeeze else result
