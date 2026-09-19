"""Convert native Isaac Lab lidar ray hits into the deployable terrain grid."""

from __future__ import annotations

from dataclasses import dataclass

from go2w_terrain_planner.utils.tensor_checks import require_finite

from .pointcloud_grid_projector import (
    PointCloudProjectionConfig,
    PointCloudGridProjector,
)


def build_spinning_lidar_pattern(
    vertical_angles_deg,
    horizontal_resolution_deg: float,
    device,
):
    """Return channel-major ray starts and directions without Isaac imports."""
    import torch

    vertical_angles = tuple(float(value) for value in vertical_angles_deg)
    if not vertical_angles:
        raise ValueError("激光雷达垂直角表不能为空")
    if horizontal_resolution_deg <= 0.0:
        raise ValueError("激光雷达水平分辨率必须大于0")
    horizontal_count_float = 360.0 / float(horizontal_resolution_deg)
    horizontal_count = int(round(horizontal_count_float))
    if abs(horizontal_count_float - horizontal_count) > 1.0e-6:
        raise ValueError("激光雷达水平分辨率必须整除360度")

    vertical = torch.deg2rad(
        torch.as_tensor(vertical_angles, dtype=torch.float32, device=device)
    )
    horizontal = (
        torch.arange(horizontal_count, dtype=torch.float32, device=device)
        * (2.0 * torch.pi / horizontal_count)
        - torch.pi
    )
    pitch, yaw = torch.meshgrid(vertical, horizontal, indexing="ij")
    directions = torch.stack(
        (
            torch.cos(pitch) * torch.cos(yaw),
            torch.cos(pitch) * torch.sin(yaw),
            torch.sin(pitch),
        ),
        dim=-1,
    ).reshape(-1, 3)
    return torch.zeros_like(directions), directions


def transform_lidar_rays_to_world(
    ray_starts_local,
    ray_directions_local,
    sensor_parent_positions_world,
    sensor_parent_yaw,
):
    """Transform a batched lidar pattern with authoritative planar poses.

    The phase-one navigation proxy already owns its exact ``(x, y, yaw)``
    state.  Using that state directly avoids creating a second PhysX rigid-body
    tensor view only to recover the same pose for the RayCaster.  The returned
    rays are still intersected with Isaac Lab's real Warp triangle mesh; this
    function only replaces the pose lookup, not the sensor simulation.
    """
    import torch

    positions = torch.as_tensor(sensor_parent_positions_world)
    yaw = torch.as_tensor(
        sensor_parent_yaw,
        dtype=positions.dtype,
        device=positions.device,
    )
    starts = torch.as_tensor(
        ray_starts_local,
        dtype=positions.dtype,
        device=positions.device,
    )
    directions = torch.as_tensor(
        ray_directions_local,
        dtype=positions.dtype,
        device=positions.device,
    )
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("sensor_parent_positions_world必须为[B,3]")
    batch_size = positions.shape[0]
    if yaw.shape != (batch_size,):
        raise ValueError("sensor_parent_yaw必须为[B]")
    if starts.ndim == 2:
        starts = starts.unsqueeze(0).expand(batch_size, -1, -1)
    if directions.ndim == 2:
        directions = directions.unsqueeze(0).expand(batch_size, -1, -1)
    if (
        starts.ndim != 3
        or directions.ndim != 3
        or starts.shape != directions.shape
        or starts.shape[0] != batch_size
        or starts.shape[-1] != 3
    ):
        raise ValueError("局部射线必须为[R,3]或[B,R,3]且起点、方向形状一致")

    cosine = torch.cos(yaw)[:, None]
    sine = torch.sin(yaw)[:, None]

    def rotate_planar(vectors):
        return torch.stack(
            (
                cosine * vectors[..., 0] - sine * vectors[..., 1],
                sine * vectors[..., 0] + cosine * vectors[..., 1],
                vectors[..., 2],
            ),
            dim=-1,
        )

    starts_world = rotate_planar(starts) + positions[:, None, :]
    directions_world = rotate_planar(directions)
    require_finite(starts_world, "激光射线世界系起点")
    require_finite(directions_world, "激光射线世界系方向")
    return starts_world.contiguous(), directions_world.contiguous()


@dataclass
class IsaacRaycastLidarConfig:
    """Measurement corruption and scan-layout parameters for native ray hits."""

    channels: int = 16
    vertical_angles_deg: tuple[float, ...] = tuple(
        -15.0 + 2.0 * index for index in range(16)
    )
    points_per_frame: int = 32000
    minimum_range_m: float = 0.20
    maximum_range_m: float = 12.0
    range_noise_std_m: float = 0.010
    height_noise_std_m: float = 0.015
    pose_xy_noise_std_m: float = 0.015
    pose_yaw_noise_std_rad: float = 0.01
    beam_dropout_probability: float = 0.03
    return_dropout_probability: float = 0.02
    projection_config: PointCloudProjectionConfig | None = None

    def validate(self) -> None:
        if self.channels <= 0 or len(self.vertical_angles_deg) != self.channels:
            raise ValueError("激光雷达垂直角数量必须与channels一致")
        if self.points_per_frame <= 0 or self.points_per_frame % self.channels != 0:
            raise ValueError("points_per_frame必须为channels的正整数倍")
        if not all(-90.0 < float(angle) < 90.0 for angle in self.vertical_angles_deg):
            raise ValueError("激光雷达垂直角必须位于(-90,90)度")
        if not 0.0 <= self.minimum_range_m < self.maximum_range_m:
            raise ValueError("激光雷达量程必须满足0 <= minimum < maximum")
        for name in (
            "range_noise_std_m",
            "height_noise_std_m",
            "pose_xy_noise_std_m",
            "pose_yaw_noise_std_rad",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name}不能为负数")
        for name in ("beam_dropout_probability", "return_dropout_probability"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name}必须位于[0,1]")
        if self.beam_dropout_probability + self.return_dropout_probability > 1.0:
            raise ValueError("整束丢失率与回波丢失率之和不能超过1")
        (self.projection_config or PointCloudProjectionConfig()).validate()


class IsaacRaycastLidarMapper:
    """Process 32000 independent native ray hits without synthesizing points."""

    def __init__(
        self,
        num_envs: int,
        device,
        cfg: IsaacRaycastLidarConfig | None = None,
    ) -> None:
        import torch

        self.cfg = cfg or IsaacRaycastLidarConfig()
        self.cfg.validate()
        if num_envs <= 0:
            raise ValueError("激光雷达环境数量必须大于0")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.horizontal_count = self.cfg.points_per_frame // self.cfg.channels
        self.vertical_angles = torch.deg2rad(
            torch.tensor(
                self.cfg.vertical_angles_deg,
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.projector = PointCloudGridProjector(
            self.cfg.projection_config or PointCloudProjectionConfig()
        )
        self.last_pointcloud = torch.full(
            (num_envs, self.cfg.points_per_frame, 3),
            torch.nan,
            dtype=torch.float32,
            device=self.device,
        )
        self.last_hit_mask = torch.zeros(
            (num_envs, self.cfg.points_per_frame),
            dtype=torch.bool,
            device=self.device,
        )

    def _ray_observed_mask(
        self,
        horizontal_ranges,
        physical_hit_mask,
        active_beam,
        *,
        extent_m: float,
        size: int,
    ):
        """Project true ray traversal into a robot-centric polar visibility mask."""
        import torch

        batch_size = horizontal_ranges.shape[0]
        ranges = horizontal_ranges.reshape(
            batch_size,
            self.cfg.channels,
            self.horizontal_count,
        )
        hits = physical_hit_mask.reshape_as(ranges)
        active = active_beam.reshape_as(ranges)
        maximum_horizontal = (
            self.cfg.maximum_range_m
            * torch.cos(self.vertical_angles)
        )[None, :, None].expand_as(ranges)
        downward = (self.vertical_angles <= 0.0)[None, :, None].expand_as(ranges)
        reach = torch.where(
            hits,
            ranges,
            torch.where(downward, maximum_horizontal, torch.zeros_like(ranges)),
        )
        reach = torch.where(active, reach, torch.zeros_like(reach))
        azimuth_reach = reach.amax(dim=1)

        resolution = float(extent_m) / int(size)
        axis = (
            torch.arange(size, dtype=ranges.dtype, device=ranges.device) + 0.5
        ) * resolution - 0.5 * float(extent_m)
        local_x, local_y = torch.meshgrid(axis, axis, indexing="ij")
        radius = torch.hypot(local_x, local_y)
        azimuth = torch.atan2(local_y, local_x)
        azimuth_index = torch.floor(
            (azimuth + torch.pi)
            * (self.horizontal_count / (2.0 * torch.pi))
        ).long() % self.horizontal_count
        cell_reach = azimuth_reach[:, azimuth_index]
        return (radius[None] <= cell_reach).to(ranges.dtype)

    def project(
        self,
        ray_hits_world,
        sensor_origins_world,
        robot_positions_world,
        robot_yaw,
        ground_reference_world_z,
        domain_randomization_scale,
        *,
        extent_m: float,
        size: int,
        maximum_relative_height_m: float,
        maximum_height_range_m: float,
        ground_fill_value: float,
        range_fill_value: float,
        normalize_heights: bool,
        env_ids=None,
    ):
        """Return a four-channel map from native world-frame ray hit positions."""
        import torch

        hits_world = torch.as_tensor(
            ray_hits_world,
            dtype=torch.float32,
            device=self.device,
        )
        batch_size = hits_world.shape[0]
        if hits_world.shape != (batch_size, self.cfg.points_per_frame, 3):
            raise ValueError("ray_hits_world形状必须为[B,points_per_frame,3]")
        sensor_origins_world = torch.as_tensor(
            sensor_origins_world, dtype=hits_world.dtype, device=self.device
        )
        robot_positions_world = torch.as_tensor(
            robot_positions_world, dtype=hits_world.dtype, device=self.device
        )
        robot_yaw = torch.as_tensor(
            robot_yaw, dtype=hits_world.dtype, device=self.device
        )
        ground_reference_world_z = torch.as_tensor(
            ground_reference_world_z, dtype=hits_world.dtype, device=self.device
        )
        randomization_scale = torch.as_tensor(
            domain_randomization_scale,
            dtype=hits_world.dtype,
            device=self.device,
        )
        if sensor_origins_world.shape != (batch_size, 3):
            raise ValueError("sensor_origins_world形状必须为[B,3]")
        if robot_positions_world.shape != (batch_size, 3):
            raise ValueError("robot_positions_world形状必须为[B,3]")
        if robot_yaw.shape != (batch_size,) or ground_reference_world_z.shape != (batch_size,):
            raise ValueError("robot_yaw和ground_reference_world_z必须为[B]")
        if randomization_scale.shape != (batch_size,):
            raise ValueError("domain_randomization_scale必须为[B]")

        physical_hit = torch.isfinite(hits_world).all(dim=-1)
        ray_delta = torch.where(
            physical_hit[:, :, None],
            hits_world - sensor_origins_world[:, None, :],
            torch.zeros_like(hits_world),
        )
        physical_range = torch.linalg.vector_norm(ray_delta, dim=-1)
        physical_hit &= torch.isfinite(physical_range)
        physical_hit &= physical_range >= self.cfg.minimum_range_m
        physical_hit &= physical_range <= self.cfg.maximum_range_m
        safe_range = torch.where(
            physical_hit,
            physical_range,
            torch.ones_like(physical_range),
        )
        ray_direction = ray_delta / safe_range[:, :, None]

        random_sample = torch.rand(
            physical_hit.shape,
            dtype=hits_world.dtype,
            device=self.device,
        )
        beam_probability = (
            self.cfg.beam_dropout_probability * randomization_scale[:, None]
        )
        return_probability = (
            self.cfg.return_dropout_probability * randomization_scale[:, None]
        )
        beam_dropped = random_sample < beam_probability
        return_dropped = (
            random_sample >= beam_probability
        ) & (
            random_sample < beam_probability + return_probability
        )
        active_beam = ~beam_dropped
        valid_return = physical_hit & active_beam & ~return_dropped

        noisy_range = physical_range
        if self.cfg.range_noise_std_m > 0.0:
            noisy_range = noisy_range + (
                self.cfg.range_noise_std_m
                * randomization_scale[:, None]
                * torch.randn_like(noisy_range)
            )
        noisy_range = noisy_range.clamp(
            self.cfg.minimum_range_m,
            self.cfg.maximum_range_m,
        )
        noisy_hits_world = sensor_origins_world[:, None, :] + ray_direction * noisy_range[:, :, None]
        if self.cfg.height_noise_std_m > 0.0:
            noisy_hits_world[..., 2] += (
                self.cfg.height_noise_std_m
                * randomization_scale[:, None]
                * torch.randn_like(noisy_hits_world[..., 2])
            )

        noisy_robot_position = robot_positions_world.clone()
        if self.cfg.pose_xy_noise_std_m > 0.0:
            noisy_robot_position[:, :2] += (
                self.cfg.pose_xy_noise_std_m
                * randomization_scale[:, None]
                * torch.randn_like(noisy_robot_position[:, :2])
            )
        noisy_yaw = robot_yaw
        if self.cfg.pose_yaw_noise_std_rad > 0.0:
            noisy_yaw = noisy_yaw + (
                self.cfg.pose_yaw_noise_std_rad
                * randomization_scale
                * torch.randn_like(noisy_yaw)
            )
        delta_x = noisy_hits_world[..., 0] - noisy_robot_position[:, None, 0]
        delta_y = noisy_hits_world[..., 1] - noisy_robot_position[:, None, 1]
        cosine = torch.cos(noisy_yaw)[:, None]
        sine = torch.sin(noisy_yaw)[:, None]
        local_x = cosine * delta_x + sine * delta_y
        local_y = -sine * delta_x + cosine * delta_y
        local_z = noisy_hits_world[..., 2] - ground_reference_world_z[:, None]
        pointcloud = torch.stack((local_x, local_y, local_z), dim=-1)
        pointcloud = torch.where(
            valid_return[:, :, None],
            pointcloud,
            torch.full_like(pointcloud, torch.nan),
        )

        horizontal_range = torch.linalg.vector_norm(ray_delta[..., :2], dim=-1)
        observed = self._ray_observed_mask(
            horizontal_range,
            physical_hit,
            active_beam,
            extent_m=extent_m,
            size=size,
        )
        result = self.projector.project(
            pointcloud,
            observed,
            extent_m=extent_m,
            size=size,
            maximum_relative_height_m=maximum_relative_height_m,
            maximum_height_range_m=maximum_height_range_m,
            ground_fill_value=ground_fill_value,
            range_fill_value=range_fill_value,
            normalize_heights=normalize_heights,
        )
        require_finite(result, "Isaac原生射线点云投影地图")

        if env_ids is None:
            if batch_size != self.num_envs:
                raise ValueError("部分环境输出必须提供env_ids")
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.shape != (batch_size,):
            raise ValueError("env_ids必须与射线批次等长")
        self.last_pointcloud[env_ids] = pointcloud
        self.last_hit_mask[env_ids] = valid_return
        return result
