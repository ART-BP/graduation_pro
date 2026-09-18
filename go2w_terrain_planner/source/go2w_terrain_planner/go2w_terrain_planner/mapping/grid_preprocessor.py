"""GPU map preprocessing shared by simulation, training, and deployment."""

from __future__ import annotations

from go2w_terrain_planner.utils.tensor_checks import require_finite


def _validate_parameters(max_abs_relative_height: float, max_height_range: float) -> None:
    if max_abs_relative_height <= 0.0:
        raise ValueError("max_abs_relative_height必须大于0")
    if max_height_range <= 0.0:
        raise ValueError("max_height_range必须大于0")


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
    """Convert three raw map layers into a finite four-channel tensor.

    Inputs may have shape ``[H, W]`` or ``[B, H, W]``. Ground height must
    already use the supporting-ground reference. Channel order is ground,
    height range, observed mask, and valid-height mask.
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
    require_finite(output, "PyTorch地图预处理结果")
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
