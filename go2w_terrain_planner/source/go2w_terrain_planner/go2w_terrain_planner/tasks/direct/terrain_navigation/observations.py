"""Tensor-only observation assembly shared by training checks and the task."""

from __future__ import annotations

from go2w_terrain_planner.utils.tensor_checks import require_finite


def build_compact_map_observation(
    current_fused_map,
    previous_aligned_fused_map,
    aligned_observation_history,
):
    """融合本地地图编码（7 通道输出）

        输入：当前融合地图（4 通道）、上一张对齐的融合地图、对齐的观测时序历史
        处理：计算地形近期变化（地面高度变、高度差变）、观测年龄（多久未被新数据覆盖）、观测置信度（时间加权）
        输出：7 通道张量 [B, 7, H, W]，其中前 4 通道是原始融合地图，后 3 通道分别为：
        recent_change: [0,1] 最近一步内的几何或掩膜变化程度
        age: [0,1] 最老被观测样本的相对年龄（越新为 0，越老为 1）
        confidence: [0,1] 基于历史观测加权的置信度（时间衰减权）
    """
    import torch

    if (
        current_fused_map.ndim != 4
        or current_fused_map.shape[1] != 4
        or previous_aligned_fused_map.shape != current_fused_map.shape
    ):
        raise ValueError("当前与上一张融合地图必须为同形状[B,4,H,W]")
    if (
        aligned_observation_history.ndim != 5
        or aligned_observation_history.shape[0] != current_fused_map.shape[0]
        or aligned_observation_history.shape[1] < 2
        or aligned_observation_history.shape[2:] != current_fused_map.shape[1:]
    ):
        raise ValueError("对齐观测历史必须为[B,T,4,H,W]且T>=2")

    current_valid = current_fused_map[:, 3] > 0.5
    previous_valid = previous_aligned_fused_map[:, 3] > 0.5
    common_valid = current_valid & previous_valid
    ground_change = 0.5 * (
        current_fused_map[:, 0] - previous_aligned_fused_map[:, 0]
    ).abs().clamp(max=2.0)
    range_change = (
        current_fused_map[:, 1] - previous_aligned_fused_map[:, 1]
    ).abs().clamp(max=1.0)
    geometric_change = torch.maximum(ground_change, range_change)
    geometric_change = torch.where(
        common_valid, geometric_change, torch.zeros_like(geometric_change)
    )
    mask_change = torch.maximum(
        (current_fused_map[:, 2] - previous_aligned_fused_map[:, 2]).abs(),
        (current_fused_map[:, 3] - previous_aligned_fused_map[:, 3]).abs(),
    )
    recent_change = torch.maximum(geometric_change, mask_change).clamp(0.0, 1.0)

    observed_history = aligned_observation_history[:, :, 2].clamp(0.0, 1.0)
    history_length = observed_history.shape[1]
    indices = torch.arange(
        history_length,
        dtype=torch.long,
        device=observed_history.device,
    )[None, :, None, None]
    latest_observed = torch.where(
        observed_history > 0.5,
        indices,
        torch.full_like(indices, -1),
    ).amax(dim=1)
    age = (
        (history_length - 1 - latest_observed).to(current_fused_map.dtype)
        / float(history_length - 1)
    ).clamp(0.0, 1.0)

    recency_weights = torch.linspace(
        0.25,
        1.0,
        history_length,
        dtype=current_fused_map.dtype,
        device=current_fused_map.device,
    )[None, :, None, None]
    confidence = (
        (observed_history * recency_weights).sum(dim=1)
        / recency_weights.sum()
    ).clamp(0.0, 1.0)

    output = torch.cat(
        (
            current_fused_map,
            recent_change[:, None],
            age[:, None],
            confidence[:, None],
        ),
        dim=1,
    )
    require_finite(output, "紧凑地图观测")
    return output


def normalize_policy_auxiliary_inputs(
    current_velocity,
    command_history,
    motion_history,
    *,
    velocity_scale,
    map_extent_m: float,
):
    """非地图输入归一化

        输入：当前速度、指令历史、运动历史（轨迹）+ 物理尺度参数
        处理：
        速度与指令 / velocity_scale（通常是最大线速度与角速度，例如 [1.0, 1.0]）
        运动中的 xy 位移 / (0.5 × map_extent_m)；偏航角 / π
        输出：归一化后的速度、指令、运动张量（[-1, 1] 范围便于网络学习）"""
    import torch

    if map_extent_m <= 0.0:
        raise ValueError("map_extent_m必须大于0")
    scale = torch.as_tensor(
        velocity_scale,
        dtype=current_velocity.dtype,
        device=current_velocity.device,
    )
    if scale.shape != (2,) or not bool(torch.isfinite(scale).all()) or bool(
        torch.any(scale <= 0.0)
    ):
        raise ValueError("velocity_scale必须为两个有限正数")
    normalized_motion = motion_history.clone()
    normalized_motion[..., :2] /= 0.5 * float(map_extent_m)
    normalized_motion[..., 2] /= torch.pi
    normalized_velocity = current_velocity / scale
    normalized_commands = command_history / scale
    require_finite(normalized_velocity, "归一化当前速度")
    require_finite(normalized_commands, "归一化指令历史")
    require_finite(normalized_motion, "归一化运动历史")
    return normalized_velocity, normalized_commands, normalized_motion


def assemble_policy_observation(
    compact_map,
    goal,
    current_velocity,
    command_history,
    motion_history,
    imu_history=None,
    *,
    velocity_scale,
    map_extent_m: float,
    expected_dimension: int | None = None,
):
    """完整观测拼装

        输入：紧凑地图、目标位置、当前速度、指令历史、运动历史、可选 IMU 历史 + 物理参数
        处理：
        调用上面两个函数进行地图编码与归一化
        将所有张量按维度 -1 拼接为单一向量（展平二维/三维部分）
        输出：单一向量 [B, D]，其中 D = 地图展平维度(7×H×W) + 目标(3) + 速度(2) + 指令展平(T×2) + 运动展平(T×3) [+ IMU展平(T×5)]
        可选校验：检查输出维度是否与预期相符（用于捕捉配置错误）"""
    import torch

    if compact_map.ndim != 4:
        raise ValueError("compact_map必须为[B,C,H,W]")
    batch = compact_map.shape[0]
    if goal.shape != (batch, 3):
        raise ValueError("goal必须为[B,3]")
    if current_velocity.shape != (batch, 2):
        raise ValueError("current_velocity必须为[B,2]")
    if command_history.ndim != 3 or command_history.shape[0] != batch or command_history.shape[2] != 2:
        raise ValueError("command_history必须为[B,T,2]")
    if motion_history.shape != (batch, command_history.shape[1], 3):
        raise ValueError("motion_history必须为与指令历史等长的[B,T,3]")
    (
        normalized_velocity,
        normalized_commands,
        normalized_motion,
    ) = normalize_policy_auxiliary_inputs(
        current_velocity,
        command_history,
        motion_history,
        velocity_scale=velocity_scale,
        map_extent_m=map_extent_m,
    )
    values = [
        compact_map.flatten(1),
        goal,
        normalized_velocity,
        normalized_commands.flatten(1),
        normalized_motion.flatten(1),
    ]
    if imu_history is not None:
        if imu_history.shape != (batch, command_history.shape[1], 5):
            raise ValueError("imu_history必须为与运动历史等长的[B,T,5]")
        values.append(imu_history.flatten(1))
    output = torch.cat(values, dim=-1)
    if expected_dimension is not None and output.shape[-1] != expected_dimension:
        raise ValueError(
            f"policy观测维度错误：actual={output.shape[-1]}, expected={expected_dimension}"
        )
    require_finite(output, "policy观测")
    return output
