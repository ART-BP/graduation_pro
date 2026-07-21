from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]


def preprocess_grid_map(
    ground_height: NDArray,
    height_range: NDArray,
    observed_mask: NDArray,
    *,
    robot_z: float,
    ground_fill_value: float = 0.0,
    range_fill_value: float = 0.0,
    max_abs_relative_height: float = 1.5,
    max_height_range: float = 3.0,
) -> FloatArray:
    """
    将局部环境 Grid Map 转换为学习规划器输入。

    输入图层：
        ground_height:
            odom坐标系下的绝对地面高程。

        height_range:
            栅格历史观测中最高点与最低点之差。

        observed_mask:
            点云命中或激光射线经过为1，未知为0。

    输出图层顺序：
        0: relative_ground_height
        1: height_range
        2: observed_mask
        3: height_valid_mask

    Returns:
        float32数组，形状为[4, H, W]，不含NaN或Inf。
    """
    ground = np.asarray(ground_height, dtype=np.float32)
    height_diff = np.asarray(height_range, dtype=np.float32)
    observed = np.asarray(observed_mask, dtype=np.float32)

    if ground.ndim != 2:
        raise ValueError(
            f"ground_height必须是二维数组，实际shape={ground.shape}"
        )

    if ground.shape != height_diff.shape or ground.shape != observed.shape:
        raise ValueError(
            "三个地图图层的尺寸必须一致："
            f"ground={ground.shape}, "
            f"height_range={height_diff.shape}, "
            f"observed={observed.shape}"
        )

    if not np.isfinite(robot_z):
        raise ValueError(f"robot_z必须为有限数值，实际值={robot_z}")

    if max_abs_relative_height <= 0.0:
        raise ValueError("max_abs_relative_height必须大于0")

    if max_height_range <= 0.0:
        raise ValueError("max_height_range必须大于0")

    # observed=1不代表存在有效地面高程，因此单独生成有效性掩码。
    height_valid = np.isfinite(ground) & np.isfinite(height_diff)

    relative_ground = ground - np.float32(robot_z)
    relative_ground = np.clip(
        relative_ground,
        -max_abs_relative_height,
        max_abs_relative_height,
    )

    clean_ground = np.where(
        height_valid,
        relative_ground,
        np.float32(ground_fill_value),
    )

    clean_range = np.clip(height_diff, 0.0, max_height_range)
    clean_range = np.where(
        height_valid,
        clean_range,
        np.float32(range_fill_value),
    )

    clean_observed = np.nan_to_num(
        observed,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    clean_observed = np.clip(clean_observed, 0.0, 1.0)

    output = np.stack(
        (
            clean_ground,
            clean_range,
            clean_observed,
            height_valid.astype(np.float32),
        ),
        axis=0,
    ).astype(np.float32, copy=False)

    if output.shape != (4, *ground.shape):
        raise RuntimeError(f"输出形状异常：{output.shape}")

    if not np.isfinite(output).all():
        raise RuntimeError("地图预处理结果中仍存在NaN或Inf")

    return output
