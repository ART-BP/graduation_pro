"""Pure-PyTorch point-cloud projection matching the real local-map semantics.

This module deliberately has no ROS, PCL, Grid Map, or ``local_map_ws``
dependency.  It mirrors the deployable mapper's important observation rules:
crop the registered cloud, group points by 0.05 m cells, estimate ground with a
low percentile, estimate vertical span with robust tail percentiles, and keep
ray traversal independent from height validity.
"""

from __future__ import annotations

from dataclasses import dataclass

from go2w_terrain_planner.utils.tensor_checks import require_finite


@dataclass
class PointCloudProjectionConfig:
    """Point-cloud crop and robust per-cell projection parameters."""

    input_crop_length_x_m: float = 12.0
    input_crop_length_y_m: float = 12.0
    body_height_m: float = 0.50
    vertical_min_offset_m: float = -1.5
    vertical_max_offset_m: float = 1.5
    ground_percentile: float = 0.01
    span_lower_percentile: float = 0.05
    span_upper_percentile: float = 0.95
    minimum_points_per_cell: int = 1
    height_quantization_m: float = 1.0e-3
    maximum_ground_deviation_m: float = 0.12
    fusion_span_percentile: float = 0.75

    def validate(self) -> None:
        if self.input_crop_length_x_m <= 0.0 or self.input_crop_length_y_m <= 0.0:
            raise ValueError("点云水平裁剪范围必须大于0")
        if self.body_height_m <= 0.0:
            raise ValueError("仿真机体离地高度必须大于0")
        if self.vertical_max_offset_m <= self.vertical_min_offset_m:
            raise ValueError("点云垂直裁剪范围无效")
        if not 0.0 <= self.ground_percentile <= 1.0:
            raise ValueError("ground_percentile必须位于[0,1]")
        if not (
            0.0 <= self.span_lower_percentile
            <= self.span_upper_percentile
            <= 1.0
        ):
            raise ValueError("高度差分位数必须有序且位于[0,1]")
        if self.minimum_points_per_cell <= 0:
            raise ValueError("minimum_points_per_cell必须大于0")
        if self.height_quantization_m <= 0.0:
            raise ValueError("height_quantization_m必须大于0")
        if self.maximum_ground_deviation_m <= 0.0:
            raise ValueError("maximum_ground_deviation_m必须大于0")
        if not 0.0 <= self.fusion_span_percentile <= 1.0:
            raise ValueError("fusion_span_percentile必须位于[0,1]")


class SimulatedPointCloudProjector:
    """Project batched robot-frame point clouds into four map channels."""

    def __init__(self, cfg: PointCloudProjectionConfig | None = None) -> None:
        self.cfg = cfg or PointCloudProjectionConfig()
        self.cfg.validate()

    @staticmethod
    def _empty_measurements(batch_size: int, size: int, *, dtype, device):
        import torch

        ground = torch.full(
            (batch_size, size, size),
            torch.nan,
            dtype=dtype,
            device=device,
        )
        height_range = torch.full_like(ground, torch.nan)
        return ground, height_range

    def _robust_cell_statistics(self, pointcloud, extent_m: float, size: int):
        """Return the low ground quantile and robust span for every cell."""
        import torch

        batch_size = int(pointcloud.shape[0])
        dtype = pointcloud.dtype
        device = pointcloud.device
        ground, height_range = self._empty_measurements(
            batch_size,
            size,
            dtype=dtype,
            device=device,
        )

        half_crop_x = 0.5 * self.cfg.input_crop_length_x_m
        half_crop_y = 0.5 * self.cfg.input_crop_length_y_m
        half_extent = 0.5 * float(extent_m)
        relative_body_z = pointcloud[..., 2] - self.cfg.body_height_m
        valid = torch.isfinite(pointcloud).all(dim=-1)
        valid &= pointcloud[..., 0].abs() <= half_crop_x
        valid &= pointcloud[..., 1].abs() <= half_crop_y
        valid &= relative_body_z >= self.cfg.vertical_min_offset_m
        valid &= relative_body_z <= self.cfg.vertical_max_offset_m
        # The outer crop controls accepted input points. Only endpoints in the
        # inner rolling-map footprint contribute height measurements.
        valid &= pointcloud[..., 0] >= -half_extent
        valid &= pointcloud[..., 0] < half_extent
        valid &= pointcloud[..., 1] >= -half_extent
        valid &= pointcloud[..., 1] < half_extent
        resolution = float(extent_m) / int(size)
        rows = torch.floor(
            (pointcloud[..., 0] + half_extent) / resolution
        ).long().clamp(0, size - 1)
        columns = torch.floor(
            (pointcloud[..., 1] + half_extent) / resolution
        ).long().clamp(0, size - 1)
        batch_indices = torch.arange(
            batch_size, device=device, dtype=torch.long
        )[:, None].expand_as(rows)
        cells_per_environment = size * size
        global_cell = (
            batch_indices * cells_per_environment + rows * size + columns
        )[valid]
        heights = pointcloud[..., 2][valid]

        # Integer composite keys produce a lexicographic sort by cell and then
        # height without relying on unstable floating-point key spacing.
        vertical_minimum = self.cfg.body_height_m + self.cfg.vertical_min_offset_m
        vertical_span = (
            self.cfg.vertical_max_offset_m - self.cfg.vertical_min_offset_m
        )
        height_bin_count = int(
            round(vertical_span / self.cfg.height_quantization_m)
        ) + 2
        height_bin = torch.round(
            (heights - vertical_minimum) / self.cfg.height_quantization_m
        ).long().clamp(0, height_bin_count - 1)
        key = global_cell * height_bin_count + height_bin
        order = torch.argsort(key)
        sorted_cells = global_cell[order]
        sorted_heights = heights[order]
        unique_cells, counts = torch.unique_consecutive(
            sorted_cells,
            return_counts=True,
        )
        group_end = torch.cumsum(counts, dim=0)
        group_start = group_end - counts
        last_index = counts - 1
        usable = counts >= self.cfg.minimum_points_per_cell
        ground_offset = torch.floor(
            self.cfg.ground_percentile * last_index.to(dtype=dtype)
        ).long()
        lower_offset = torch.floor(
            self.cfg.span_lower_percentile * last_index.to(dtype=dtype)
        ).long()
        upper_offset = torch.ceil(
            self.cfg.span_upper_percentile * last_index.to(dtype=dtype)
        ).long()
        selected_cells = unique_cells[usable]
        selected_ground = sorted_heights[(group_start + ground_offset)[usable]]
        selected_lower = sorted_heights[(group_start + lower_offset)[usable]]
        selected_upper = sorted_heights[(group_start + upper_offset)[usable]]

        flat_ground = ground.reshape(-1)
        flat_range = height_range.reshape(-1)
        flat_ground[selected_cells] = selected_ground
        flat_range[selected_cells] = (selected_upper - selected_lower).clamp(min=0.0)
        return ground, height_range

    def project(
        self,
        pointcloud,
        ray_observed,
        *,
        extent_m: float,
        size: int,
        maximum_relative_height_m: float,
        maximum_height_range_m: float,
        ground_fill_value: float,
        range_fill_value: float,
        normalize_heights: bool,
    ):
        """Return ``[B,4,H,W]`` from ``[B,N,3]`` robot-frame points."""
        from .grid_preprocessor import preprocess_grid_map_torch

        if pointcloud.ndim != 3 or pointcloud.shape[-1] != 3:
            raise ValueError("pointcloud必须为[B,N,3]")
        if ray_observed.shape != (pointcloud.shape[0], size, size):
            raise ValueError("ray_observed必须为[B,H,W]")
        ground, height_range = self._robust_cell_statistics(
            pointcloud,
            extent_m,
            size,
        )
        result = preprocess_grid_map_torch(
            ground,
            height_range,
            ray_observed.to(dtype=pointcloud.dtype),
            max_abs_relative_height=maximum_relative_height_m,
            max_height_range=maximum_height_range_m,
            ground_fill_value=ground_fill_value,
            range_fill_value=range_fill_value,
            normalize=normalize_heights,
        )
        require_finite(result, "仿真点云投影地图")
        return result
