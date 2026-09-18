"""Batched XT16-style lidar observation model for analytic training terrain.

The phase-one terrain is randomized analytically on every episode reset, so it
cannot be observed by Isaac Lab's static-mesh ``RayCaster`` without rebuilding
the scene.  This module performs the same first-hit ray query directly against
the analytic terrain.  It produces a sparse point cloud and projects the hits
and traversed free cells into the four-channel map used by the deployable
policy.
"""

from __future__ import annotations

from dataclasses import dataclass

from go2w_terrain_planner.utils.tensor_checks import require_finite

from .simulated_pointcloud_projector import (
    PointCloudProjectionConfig,
    SimulatedPointCloudProjector,
)


DEFAULT_XT16_VERTICAL_ANGLES_DEG = tuple(-15.0 + 2.0 * index for index in range(16))


@dataclass
class Xt16LidarConfig:
    """Configurable scan pattern and point-cloud corruption model."""

    channels: int = 16
    vertical_angles_deg: tuple[float, ...] = DEFAULT_XT16_VERTICAL_ANGLES_DEG
    points_per_frame: int = 32000
    # Expensive terrain intersections are evaluated on a coarser angular
    # anchor scan, then range-continuous sectors are interpolated to the full
    # 32000-slot XT16 scan. This preserves training throughput.
    horizontal_resolution_deg: float = 2.0
    maximum_interpolation_range_difference_m: float = 0.50
    minimum_range_m: float = 0.20
    maximum_range_m: float = 12.0
    ray_step_m: float = 0.08
    mount_height_m: float = 0.45
    scan_frequency_hz: float = 10.0
    motion_distortion: bool = True
    ray_chunk_size: int = 256
    range_noise_std_m: float = 0.010
    height_noise_std_m: float = 0.015
    pose_xy_noise_std_m: float = 0.015
    pose_yaw_noise_std_rad: float = 0.01
    time_jitter_std_s: float = 0.005
    beam_dropout_probability: float = 0.03
    return_dropout_probability: float = 0.02
    projection_config: PointCloudProjectionConfig | None = None

    def validate(self) -> None:
        if self.channels <= 0 or len(self.vertical_angles_deg) != self.channels:
            raise ValueError("XT16垂直角数量必须与channels一致")
        if self.points_per_frame <= 0 or self.points_per_frame % self.channels != 0:
            raise ValueError("XT16 points_per_frame必须为channels的正整数倍")
        if not all(-90.0 < float(angle) < 90.0 for angle in self.vertical_angles_deg):
            raise ValueError("XT16垂直角必须位于(-90,90)度")
        if not 0.0 < self.horizontal_resolution_deg <= 20.0:
            raise ValueError("XT16水平角分辨率必须位于(0,20]度")
        if not 0.0 <= self.minimum_range_m < self.maximum_range_m:
            raise ValueError("XT16量程必须满足0 <= minimum < maximum")
        if self.ray_step_m <= 0.0 or self.ray_step_m > self.maximum_range_m:
            raise ValueError("XT16射线步长无效")
        if self.maximum_interpolation_range_difference_m <= 0.0:
            raise ValueError("XT16插值量程差阈值必须大于0")
        if self.mount_height_m <= 0.0 or self.scan_frequency_hz <= 0.0:
            raise ValueError("XT16安装高度和扫描频率必须大于0")
        if self.ray_chunk_size <= 0:
            raise ValueError("XT16射线分块大小必须大于0")
        for name in (
            "range_noise_std_m",
            "height_noise_std_m",
            "pose_xy_noise_std_m",
            "pose_yaw_noise_std_rad",
            "time_jitter_std_s",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"XT16 {name}不能为负数")
        for name in ("beam_dropout_probability", "return_dropout_probability"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"XT16 {name}必须位于[0,1]")
        if self.beam_dropout_probability + self.return_dropout_probability > 1.0:
            raise ValueError("XT16整束丢失率与回波丢失率之和不能超过1")
        (self.projection_config or PointCloudProjectionConfig()).validate()


class SimulatedXt16Lidar:
    """Generate first-return lidar points and a projected local elevation map."""

    def __init__(self, num_envs: int, device, cfg: Xt16LidarConfig | None = None) -> None:
        import torch

        self.cfg = cfg or Xt16LidarConfig()
        self.cfg.validate()
        if num_envs <= 0:
            raise ValueError("XT16环境数量必须大于0")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)

        trace_horizontal_count = max(
            1, int(round(360.0 / self.cfg.horizontal_resolution_deg))
        )
        trace_horizontal_angles = torch.linspace(
            -torch.pi,
            torch.pi,
            trace_horizontal_count + 1,
            device=self.device,
            dtype=torch.float32,
        )[:-1]
        vertical_angles = torch.deg2rad(
            torch.tensor(
                self.cfg.vertical_angles_deg,
                device=self.device,
                dtype=torch.float32,
            )
        )
        pitch, azimuth = torch.meshgrid(
            vertical_angles,
            trace_horizontal_angles,
            indexing="ij",
        )
        self.pitch = pitch.reshape(-1)
        self.azimuth = azimuth.reshape(-1)
        self.cos_pitch = torch.cos(self.pitch)
        self.sin_pitch = torch.sin(self.pitch)
        # One full rotation is centered around the pose timestamp.
        self.scan_time_offset = (
            self.azimuth / (2.0 * torch.pi * self.cfg.scan_frequency_hz)
        )
        self.num_trace_rays = int(self.azimuth.numel())
        self.num_rays = int(self.cfg.points_per_frame)
        self.raw_horizontal_count = self.num_rays // self.cfg.channels

        raw_position = (
            torch.arange(
                self.raw_horizontal_count,
                device=self.device,
                dtype=torch.float32,
            )
            * float(trace_horizontal_count)
            / float(self.raw_horizontal_count)
        )
        left_horizontal = torch.floor(raw_position).long() % trace_horizontal_count
        right_horizontal = (left_horizontal + 1) % trace_horizontal_count
        alpha = raw_position - torch.floor(raw_position)
        channel_offset = (
            torch.arange(self.cfg.channels, device=self.device, dtype=torch.long)
            * trace_horizontal_count
        )[:, None]
        self.raw_left_index = (channel_offset + left_horizontal[None, :]).reshape(-1)
        self.raw_right_index = (channel_offset + right_horizontal[None, :]).reshape(-1)
        self.raw_interpolation_alpha = alpha.repeat(self.cfg.channels)
        self.projector = SimulatedPointCloudProjector(
            self.cfg.projection_config or PointCloudProjectionConfig()
        )
        self.last_pointcloud = torch.full(
            (self.num_envs, self.num_rays, 3),
            torch.nan,
            dtype=torch.float32,
            device=self.device,
        )
        self.last_hit_mask = torch.zeros(
            (self.num_envs, self.num_rays),
            dtype=torch.bool,
            device=self.device,
        )

    def _densify_anchor_scan(self, trace_points, trace_ranges, trace_hit_mask):
        """Interpolate a coarse first-hit scan into exactly 32000 XT16 slots."""
        import torch

        left_points = trace_points[:, self.raw_left_index]
        right_points = trace_points[:, self.raw_right_index]
        left_ranges = trace_ranges[:, self.raw_left_index]
        right_ranges = trace_ranges[:, self.raw_right_index]
        left_valid = trace_hit_mask[:, self.raw_left_index]
        right_valid = trace_hit_mask[:, self.raw_right_index]
        alpha = self.raw_interpolation_alpha.to(dtype=trace_points.dtype)[None, :]
        continuous = (
            left_valid
            & right_valid
            & (
                (left_ranges - right_ranges).abs()
                <= self.cfg.maximum_interpolation_range_difference_m
            )
        )
        interpolated_points = torch.lerp(
            left_points,
            right_points,
            alpha[:, :, None],
        )
        prefer_left = alpha <= 0.5
        choose_left = left_valid & (~right_valid | prefer_left)
        nearest_points = torch.where(
            choose_left[:, :, None],
            left_points,
            right_points,
        )
        nearest_valid = left_valid | right_valid
        points = torch.where(
            continuous[:, :, None],
            interpolated_points,
            nearest_points,
        )
        valid = continuous | nearest_valid
        points = torch.where(
            valid[:, :, None],
            points,
            torch.full_like(points, torch.nan),
        )
        return points, valid

    @staticmethod
    def _grid_indices(local_x, local_y, extent_m: float, size: int):
        import torch

        resolution = float(extent_m) / int(size)
        half_extent = 0.5 * float(extent_m)
        row = torch.floor((local_x + half_extent) / resolution).long()
        column = torch.floor((local_y + half_extent) / resolution).long()
        inside = (
            (row >= 0)
            & (row < size)
            & (column >= 0)
            & (column < size)
        )
        return row.clamp(0, size - 1), column.clamp(0, size - 1), inside

    def generate_map(
        self,
        terrain_model,
        pose,
        velocity=None,
        *,
        env_ids=None,
        return_ground_reference: bool = False,
    ):
        """Return a sparse lidar-derived map with shape ``[B,4,H,W]``."""
        import torch

        if pose.ndim != 2 or pose.shape[1] != 3:
            raise ValueError("XT16 pose必须为[B,3]")
        env_ids = terrain_model._environment_indices(pose.shape[0], env_ids)
        batch_size = int(env_ids.numel())
        if velocity is None:
            velocity = torch.zeros((batch_size, 2), dtype=pose.dtype, device=pose.device)
        if velocity.shape != (batch_size, 2):
            raise ValueError("XT16 velocity必须为[B,2]")

        size = int(terrain_model.cfg.size)
        cell_count = size * size
        dtype = pose.dtype
        ground_reference = terrain_model.ground_reference(pose, env_ids)
        randomization_scale = terrain_model.domain_randomization_scale[env_ids]

        noisy_pose = pose.clone()
        if self.cfg.pose_xy_noise_std_m > 0.0:
            noisy_pose[:, :2] += (
                self.cfg.pose_xy_noise_std_m
                * randomization_scale[:, None]
                * torch.randn_like(noisy_pose[:, :2])
            )
        if self.cfg.pose_yaw_noise_std_rad > 0.0:
            noisy_pose[:, 2] += (
                self.cfg.pose_yaw_noise_std_rad
                * randomization_scale
                * torch.randn_like(noisy_pose[:, 2])
            )
        scan_jitter = torch.zeros(batch_size, dtype=dtype, device=self.device)
        if self.cfg.time_jitter_std_s > 0.0:
            scan_jitter = (
                self.cfg.time_jitter_std_s
                * randomization_scale
                * torch.randn_like(scan_jitter)
            )

        # The point cloud honors the physical sensor range. Only the subsequent
        # grid projection is cropped to the 10 m local-map footprint.
        effective_maximum_range = self.cfg.maximum_range_m
        sample_ranges = torch.arange(
            self.cfg.minimum_range_m,
            effective_maximum_range + 0.5 * self.cfg.ray_step_m,
            self.cfg.ray_step_m,
            dtype=dtype,
            device=self.device,
        )
        if sample_ranges.numel() == 0:
            raise RuntimeError("XT16有效地图量程内没有射线采样点")

        observed_flat = torch.zeros(
            batch_size * cell_count,
            dtype=torch.bool,
            device=self.device,
        )
        batch_offsets = (
            torch.arange(batch_size, device=self.device, dtype=torch.long) * cell_count
        )
        current_yaw = pose[:, 2, None, None]
        current_cosine = torch.cos(current_yaw)
        current_sine = torch.sin(current_yaw)
        trace_points = torch.full(
            (batch_size, self.num_trace_rays, 3),
            torch.nan,
            dtype=dtype,
            device=self.device,
        )
        trace_ranges = torch.full(
            (batch_size, self.num_trace_rays),
            self.cfg.maximum_range_m,
            dtype=dtype,
            device=self.device,
        )
        trace_hit_mask = torch.zeros(
            (batch_size, self.num_trace_rays),
            dtype=torch.bool,
            device=self.device,
        )

        for ray_start in range(0, self.num_trace_rays, self.cfg.ray_chunk_size):
            ray_end = min(ray_start + self.cfg.ray_chunk_size, self.num_trace_rays)
            azimuth = self.azimuth[ray_start:ray_end].to(dtype=dtype)
            pitch = self.pitch[ray_start:ray_end].to(dtype=dtype)
            cos_pitch = self.cos_pitch[ray_start:ray_end].to(dtype=dtype)
            sin_pitch = self.sin_pitch[ray_start:ray_end].to(dtype=dtype)
            time_offset = self.scan_time_offset[ray_start:ray_end].to(dtype=dtype)
            time_offset = time_offset[None, :] + scan_jitter[:, None]
            if not self.cfg.motion_distortion:
                time_offset = torch.zeros_like(time_offset)

            base_yaw = noisy_pose[:, 2, None]
            sensor_yaw = base_yaw + velocity[:, 1, None] * time_offset
            travelled = velocity[:, 0, None] * time_offset
            origin_x = noisy_pose[:, 0, None] + travelled * torch.cos(base_yaw)
            origin_y = noisy_pose[:, 1, None] + travelled * torch.sin(base_yaw)
            ray_yaw = sensor_yaw + azimuth[None, :]
            direction_x = cos_pitch[None, :] * torch.cos(ray_yaw)
            direction_y = cos_pitch[None, :] * torch.sin(ray_yaw)
            direction_z = sin_pitch[None, :].expand(batch_size, -1)

            distances = sample_ranges[None, None, :]
            world_x = origin_x[:, :, None] + direction_x[:, :, None] * distances
            world_y = origin_y[:, :, None] + direction_y[:, :, None] * distances
            ray_z = (
                ground_reference[:, None, None]
                + self.cfg.mount_height_m
                + direction_z[:, :, None] * distances
            )
            ground, obstacle_height = terrain_model._terrain_surface(
                world_x,
                world_y,
                env_ids,
            )
            tolerance = 0.5 * self.cfg.ray_step_m
            ground_intersection = ray_z <= ground + tolerance
            obstacle_intersection = (
                (obstacle_height > 0.0)
                & (ray_z >= ground - tolerance)
                & (ray_z <= ground + obstacle_height + tolerance)
            )
            intersection = ground_intersection | obstacle_intersection
            has_hit = intersection.any(dim=-1)
            first_index = intersection.to(torch.int8).argmax(dim=-1)
            end_index = torch.where(
                has_hit,
                first_index,
                torch.full_like(first_index, sample_ranges.numel() - 1),
            )

            random_sample = torch.rand(
                has_hit.shape,
                dtype=dtype,
                device=self.device,
            )
            beam_dropout_probability = (
                self.cfg.beam_dropout_probability
                * randomization_scale[:, None]
            )
            return_dropout_probability = (
                self.cfg.return_dropout_probability
                * randomization_scale[:, None]
            )
            beam_dropped = random_sample < beam_dropout_probability
            return_dropped = (
                random_sample >= beam_dropout_probability
            ) & (
                random_sample
                < beam_dropout_probability + return_dropout_probability
            )
            local_azimuth = azimuth[None, :]
            occlusion_delta = torch.atan2(
                torch.sin(local_azimuth - terrain_model.occlusion_angle[env_ids, None]),
                torch.cos(local_azimuth - terrain_model.occlusion_angle[env_ids, None]),
            )
            occluded = terrain_model.occlusion_enabled[env_ids, None] & (
                torch.abs(occlusion_delta)
                < 0.5 * terrain_model.occlusion_width[env_ids, None]
            )
            active_beam = ~(beam_dropped | occluded)
            valid_return = has_hit & active_beam & ~return_dropped

            # Mark the 2-D cells traversed before the first return. Upward rays
            # only contribute when they hit an obstacle, avoiding false free
            # space observations above low terrain.
            path_index = torch.arange(
                sample_ranges.numel(), device=self.device
            )[None, None, :]
            path_valid = active_beam[:, :, None] & (
                path_index <= end_index[:, :, None]
            )
            path_valid &= (pitch[None, :, None] <= 0.0) | has_hit[:, :, None]
            delta_x = world_x - pose[:, 0, None, None]
            delta_y = world_y - pose[:, 1, None, None]
            local_x = current_cosine * delta_x + current_sine * delta_y
            local_y = -current_sine * delta_x + current_cosine * delta_y
            row, column, inside = self._grid_indices(
                local_x,
                local_y,
                terrain_model.cfg.extent_m,
                size,
            )
            path_valid &= inside
            global_path_index = (
                batch_offsets[:, None, None] + row * size + column
            )
            observed_flat[global_path_index[path_valid]] = True

            selected_distance = sample_ranges[first_index]
            if self.cfg.range_noise_std_m > 0.0:
                selected_distance = selected_distance + (
                    self.cfg.range_noise_std_m
                    * randomization_scale[:, None]
                    * torch.randn_like(selected_distance)
                )
            selected_distance = selected_distance.clamp(
                self.cfg.minimum_range_m,
                effective_maximum_range,
            )
            hit_world_x = origin_x + direction_x * selected_distance
            hit_world_y = origin_y + direction_y * selected_distance
            sampled_ground = ground.gather(-1, first_index.unsqueeze(-1)).squeeze(-1)
            sampled_obstacle = obstacle_intersection.gather(
                -1, first_index.unsqueeze(-1)
            ).squeeze(-1)
            ray_hit_z = (
                ground_reference[:, None]
                + self.cfg.mount_height_m
                + direction_z * selected_distance
            )
            hit_world_z = torch.where(sampled_obstacle, ray_hit_z, sampled_ground)
            if self.cfg.height_noise_std_m > 0.0:
                hit_world_z = hit_world_z + (
                    self.cfg.height_noise_std_m
                    * randomization_scale[:, None]
                    * torch.randn_like(hit_world_z)
                )
            hit_local_x = (
                torch.cos(pose[:, 2, None]) * (hit_world_x - pose[:, 0, None])
                + torch.sin(pose[:, 2, None]) * (hit_world_y - pose[:, 1, None])
            )
            hit_local_y = (
                -torch.sin(pose[:, 2, None]) * (hit_world_x - pose[:, 0, None])
                + torch.cos(pose[:, 2, None]) * (hit_world_y - pose[:, 1, None])
            )
            hit_relative_z = hit_world_z - ground_reference[:, None]

            points = torch.stack(
                (hit_local_x, hit_local_y, hit_relative_z), dim=-1
            )
            points = torch.where(
                valid_return[:, :, None],
                points,
                torch.full_like(points, torch.nan),
            )
            trace_points[:, ray_start:ray_end] = points
            trace_ranges[:, ray_start:ray_end] = torch.where(
                valid_return,
                selected_distance,
                torch.full_like(selected_distance, self.cfg.maximum_range_m),
            )
            trace_hit_mask[:, ray_start:ray_end] = valid_return

        all_points, all_hit_mask = self._densify_anchor_scan(
            trace_points,
            trace_ranges,
            trace_hit_mask,
        )
        observed = observed_flat.reshape(batch_size, size, size).to(dtype=dtype)
        result = self.projector.project(
            all_points,
            observed,
            extent_m=terrain_model.cfg.extent_m,
            size=size,
            maximum_relative_height_m=terrain_model.cfg.maximum_relative_height_m,
            maximum_height_range_m=terrain_model.cfg.maximum_height_range_m,
            ground_fill_value=terrain_model.cfg.ground_fill_value,
            range_fill_value=terrain_model.cfg.range_fill_value,
            normalize_heights=terrain_model.cfg.normalize_heights,
        )
        require_finite(result, "XT16投影局部地图")
        self.last_pointcloud[env_ids] = all_points.to(self.last_pointcloud.dtype)
        self.last_hit_mask[env_ids] = all_hit_mask
        if return_ground_reference:
            return result, ground_reference
        return result
