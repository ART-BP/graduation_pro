"""Shared Grid Map preprocessing for simulation, datasets, and deployment."""

from __future__ import annotations

import numpy as np


def _validate_parameters(max_abs_relative_height: float, max_height_range: float) -> None:
    if max_abs_relative_height <= 0.0:
        raise ValueError("max_abs_relative_height必须大于0")
    if max_height_range <= 0.0:
        raise ValueError("max_height_range必须大于0")


def preprocess_grid_map(
    ground_height: np.ndarray,
    height_range: np.ndarray,
    observed_mask: np.ndarray,
    *,
    ground_fill_value: float = 0.0,
    range_fill_value: float = 0.0,
    max_abs_relative_height: float = 1.5,
    max_height_range: float = 3.0,
    normalize: bool = False,
) -> np.ndarray:
    """Convert the three published Grid Map layers into a finite four-channel tensor.

    ``ground_height`` must already be relative to the supporting ground below
    the robot; this function never subtracts the FAST-LIO IMU pose. Channel
    order is ``relative_ground_height``, ``height_range``,
    ``observed_mask`` and ``height_valid_mask``. ``observed_mask`` is kept
    independent from validity because a LiDAR ray may traverse a cell without
    providing a ground return.
    """
    ground = np.asarray(ground_height, dtype=np.float32)
    height_diff = np.asarray(height_range, dtype=np.float32)
    observed = np.asarray(observed_mask, dtype=np.float32)

    if ground.ndim != 2:
        raise ValueError(f"ground_height必须是二维数组，实际shape={ground.shape}")
    if ground.shape != height_diff.shape or ground.shape != observed.shape:
        raise ValueError(
            "三个地图图层的尺寸必须一致："
            f"ground={ground.shape}, height_range={height_diff.shape}, observed={observed.shape}"
        )
    _validate_parameters(max_abs_relative_height, max_height_range)

    height_valid = np.isfinite(ground) & np.isfinite(height_diff)
    relative_ground = np.clip(
        ground,
        -max_abs_relative_height,
        max_abs_relative_height,
    )
    clean_range = np.clip(height_diff, 0.0, max_height_range)
    if normalize:
        relative_ground = relative_ground / np.float32(max_abs_relative_height)
        clean_range = clean_range / np.float32(max_height_range)

    clean_ground = np.where(height_valid, relative_ground, np.float32(ground_fill_value))
    clean_range = np.where(height_valid, clean_range, np.float32(range_fill_value))
    clean_observed = np.nan_to_num(observed, nan=0.0, posinf=1.0, neginf=0.0)
    clean_observed = np.clip(clean_observed, 0.0, 1.0)

    output = np.stack(
        (clean_ground, clean_range, clean_observed, height_valid.astype(np.float32)), axis=0
    ).astype(np.float32, copy=False)
    if output.shape != (4, *ground.shape) or not np.isfinite(output).all():
        raise RuntimeError("地图预处理输出形状异常或仍包含NaN/Inf")
    return output


def preprocess_grid_map_torch(
    ground_height,
    height_range,
    observed_mask,
    *,
    ground_fill_value: float = 0.0,
    range_fill_value: float = 0.0,
    max_abs_relative_height: float = 1.5,
    max_height_range: float = 3.0,
    normalize: bool = True,
):
    """Batched PyTorch equivalent of :func:`preprocess_grid_map`.

    Inputs may have shape ``[H, W]`` or ``[B, H, W]`` and ground height must
    already use the supporting-ground reference. The output channel is inserted
    directly before the two spatial dimensions.
    """
    import torch

    _validate_parameters(max_abs_relative_height, max_height_range)
    ground = torch.as_tensor(ground_height, dtype=torch.float32)
    height_diff = torch.as_tensor(height_range, dtype=torch.float32, device=ground.device)
    observed = torch.as_tensor(observed_mask, dtype=torch.float32, device=ground.device)
    if ground.ndim not in (2, 3):
        raise ValueError(f"地图必须为[H,W]或[B,H,W]，实际shape={tuple(ground.shape)}")
    if ground.shape != height_diff.shape or ground.shape != observed.shape:
        raise ValueError("三个地图图层的尺寸必须一致")

    valid = torch.isfinite(ground) & torch.isfinite(height_diff)
    relative = torch.clamp(
        ground,
        -max_abs_relative_height,
        max_abs_relative_height,
    )
    clean_range = torch.clamp(height_diff, 0.0, max_height_range)
    if normalize:
        relative = relative / max_abs_relative_height
        clean_range = clean_range / max_height_range
    relative = torch.where(valid, relative, torch.full_like(relative, ground_fill_value))
    clean_range = torch.where(valid, clean_range, torch.full_like(clean_range, range_fill_value))
    observed = torch.nan_to_num(observed, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
    output = torch.stack((relative, clean_range, observed, valid.float()), dim=-3)
    if not torch.isfinite(output).all():
        raise RuntimeError("PyTorch地图预处理结果包含NaN或Inf")
    return output


def downsample_map_tensor(map_tensor, output_size: int):
    """Downsample ``[..., C, H, W]`` while keeping validity masks conservative."""
    import torch
    import torch.nn.functional as functional

    if map_tensor.ndim < 3 or map_tensor.shape[-3] != 4:
        raise ValueError("map_tensor末三维必须为[4,H,W]")
    if output_size <= 0:
        raise ValueError("output_size必须大于0")
    if map_tensor.shape[-2:] == (output_size, output_size):
        return map_tensor

    leading = map_tensor.shape[:-3]
    flat = map_tensor.reshape(-1, 4, map_tensor.shape[-2], map_tensor.shape[-1])
    if output_size < flat.shape[-2] and output_size < flat.shape[-1]:
        valid = flat[:, 3:4]
        valid_fraction = functional.adaptive_avg_pool2d(valid, (output_size, output_size))
        ground_sum = functional.adaptive_avg_pool2d(
            flat[:, 0:1] * valid, (output_size, output_size)
        )
        ground = torch.where(
            valid_fraction > 1.0e-6,
            ground_sum / torch.clamp(valid_fraction, min=1.0e-6),
            torch.zeros_like(ground_sum),
        )
        height_range = functional.adaptive_max_pool2d(
            flat[:, 1:2] * valid, (output_size, output_size)
        )
        observed = functional.adaptive_max_pool2d(flat[:, 2:3], (output_size, output_size))
        valid_mask = (valid_fraction > 0.0).to(flat.dtype)
        result = torch.cat((ground, height_range, observed, valid_mask), dim=1)
    else:
        heights = functional.interpolate(
            flat[:, :2], size=(output_size, output_size), mode="bilinear", align_corners=False
        )
        masks = functional.interpolate(flat[:, 2:], size=(output_size, output_size), mode="nearest")
        result = torch.cat((heights, masks), dim=1)
    return result.reshape(*leading, 4, output_size, output_size)
