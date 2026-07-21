"""Fast tensor-only approximation of the real rolling elevation-map interface.

This is the phase-one sensor model. It generates the same four actor channels
from analytic terrain geometry and observation corruption. A 16-line ray
caster can replace this class later without changing the actor interface.
"""

from __future__ import annotations

from dataclasses import dataclass


TERRAIN_NAMES = (
    "flat",
    "ramp",
    "step",
    "stairs",
    "rough",
    "pit",
    "wall",
    "pillar",
    "mixed",
    "multi_route",
)


def fuse_aligned_map_history(aligned_maps):
    """Fuse up to five aligned observations into one real-interface-equivalent map."""
    import torch

    if aligned_maps.ndim != 5 or aligned_maps.shape[2] != 4:
        raise ValueError("aligned_maps必须为[B,T,4,H,W]")
    valid = aligned_maps[:, :, 3] > 0.5
    count = valid.sum(dim=1)
    safe_count = count.clamp(min=1).to(aligned_maps.dtype)
    ground = (aligned_maps[:, :, 0] * valid).sum(dim=1) / safe_count
    height_range = torch.where(
        valid,
        aligned_maps[:, :, 1],
        torch.zeros_like(aligned_maps[:, :, 1]),
    ).amax(dim=1)
    observed = aligned_maps[:, :, 2].amax(dim=1)
    height_valid = count > 0
    ground = torch.where(height_valid, ground, torch.zeros_like(ground))
    height_range = torch.where(height_valid, height_range, torch.zeros_like(height_range))
    result = torch.stack((ground, height_range, observed, height_valid.float()), dim=1)
    if not torch.isfinite(result).all():
        raise RuntimeError("融合后的仿真地图包含NaN或Inf")
    return result


@dataclass
class SimulatedMapConfig:
    extent_m: float = 10.0
    size: int = 100
    maximum_relative_height_m: float = 1.5
    maximum_height_range_m: float = 3.0
    ground_fill_value: float = 0.0
    range_fill_value: float = 0.0
    normalize_heights: bool = True
    height_noise_std_m: float = 0.015
    range_noise_std_m: float = 0.010
    missing_probability: float = 0.03
    ray_only_probability: float = 0.02
    occlusion_sector_probability: float = 0.30
    occlusion_width_range_rad: tuple[float, float] = (0.10, 0.50)
    pose_xy_noise_std_m: float = 0.015
    pose_yaw_noise_std_rad: float = 0.01
    time_jitter_std_s: float = 0.005
    enabled_terrain_names: tuple[str, ...] = TERRAIN_NAMES
    ramp_slope_range: tuple[float, float] = (0.05, 0.35)
    step_height_range_m: tuple[float, float] = (0.05, 0.35)
    step_width_range_m: tuple[float, float] = (0.25, 0.60)
    rough_amplitude_range_m: tuple[float, float] = (0.01, 0.12)
    pit_depth_range_m: tuple[float, float] = (0.05, 0.40)
    obstacle_height_range_m: tuple[float, float] = (0.55, 1.20)
    friction_range: tuple[float, float] = (0.35, 1.20)
    robot_half_length_m: float = 0.35
    robot_half_width_m: float = 0.20
    truth_lookahead_m: float = 0.02


@dataclass
class TerrainTruthMetrics:
    """Noise-free terrain quantities used by the proxy execution model."""

    support_span_m: object
    maximum_discontinuity_m: object
    obstacle_height_m: object
    pit_depth_m: object
    entry_alignment: object
    ground_reference_z: object

    @property
    def hazard_height_m(self):
        import torch

        return torch.maximum(
            torch.maximum(self.support_span_m, self.maximum_discontinuity_m),
            self.pit_depth_m,
        )

    @property
    def collision_height_m(self):
        import torch

        return torch.maximum(self.maximum_discontinuity_m, self.obstacle_height_m)


class SimulatedLocalMap:
    """Generate batched robot-centric local maps on the training device."""

    def __init__(self, num_envs: int, device, cfg: SimulatedMapConfig | None = None) -> None:
        import torch

        self.cfg = cfg or SimulatedMapConfig()
        if num_envs <= 0 or self.cfg.size <= 1 or self.cfg.extent_m <= 0.0:
            raise ValueError("仿真地图配置无效")
        if self.cfg.maximum_relative_height_m <= 0.0 or self.cfg.maximum_height_range_m <= 0.0:
            raise ValueError("仿真地图高度归一化范围必须大于0")
        if not 0.0 <= self.cfg.missing_probability <= 1.0:
            raise ValueError("missing_probability必须位于[0,1]")
        if not 0.0 <= self.cfg.ray_only_probability <= 1.0:
            raise ValueError("ray_only_probability必须位于[0,1]")
        if self.cfg.missing_probability + self.cfg.ray_only_probability > 1.0:
            raise ValueError("缺失率与仅射线观测率之和不能超过1")
        if not 0.0 <= self.cfg.occlusion_sector_probability <= 1.0:
            raise ValueError("occlusion_sector_probability必须位于[0,1]")
        if (
            self.cfg.robot_half_length_m <= 0.0
            or self.cfg.robot_half_width_m <= 0.0
            or self.cfg.truth_lookahead_m < 0.0
        ):
            raise ValueError("机器人真值查询尺寸无效")
        self.num_envs = num_envs
        self.device = torch.device(device)
        unknown_names = set(self.cfg.enabled_terrain_names) - set(TERRAIN_NAMES)
        if unknown_names:
            raise ValueError(f"未知地形类型：{sorted(unknown_names)}")
        self.enabled_terrain_indices = torch.tensor(
            [TERRAIN_NAMES.index(name) for name in self.cfg.enabled_terrain_names],
            dtype=torch.long,
            device=self.device,
        )
        if self.enabled_terrain_indices.numel() == 0:
            raise ValueError("至少需要启用一种地形")
        axis = torch.linspace(
            -0.5 * self.cfg.extent_m,
            0.5 * self.cfg.extent_m,
            self.cfg.size,
            device=self.device,
        )
        self.local_x, self.local_y = torch.meshgrid(axis, axis, indexing="ij")
        self.local_x = self.local_x.unsqueeze(0)
        self.local_y = self.local_y.unsqueeze(0)
        self.local_angle = torch.atan2(self.local_y, self.local_x)
        self.local_radius = torch.sqrt(self.local_x.square() + self.local_y.square())
        self.half_cell_m = 0.5 * self.cfg.extent_m / self.cfg.size
        self.body_x_samples = torch.linspace(
            -self.cfg.robot_half_length_m,
            self.cfg.robot_half_length_m,
            5,
            device=self.device,
        )
        self.body_y_samples = torch.linspace(
            -self.cfg.robot_half_width_m,
            self.cfg.robot_half_width_m,
            3,
            device=self.device,
        )

        self.terrain_type = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.amplitude = torch.zeros(num_envs, device=self.device)
        self.feature_x = torch.zeros(num_envs, device=self.device)
        self.feature_y = torch.zeros(num_envs, device=self.device)
        self.feature_yaw = torch.zeros(num_envs, device=self.device)
        self.route_yaw = torch.zeros(num_envs, device=self.device)
        self.feature_width = torch.ones(num_envs, device=self.device)
        self.friction = torch.ones(num_envs, device=self.device)
        self.occlusion_enabled = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.occlusion_angle = torch.zeros(num_envs, device=self.device)
        self.occlusion_width = torch.zeros(num_envs, device=self.device)
        self.reset(torch.arange(num_envs, device=self.device))

    def reset(self, env_ids, maximum_terrain_index: int | None = None) -> None:
        import torch

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        count = env_ids.numel()
        maximum_index = (
            torch.full((count,), len(TERRAIN_NAMES) - 1, dtype=torch.long, device=self.device)
            if maximum_terrain_index is None
            else torch.as_tensor(maximum_terrain_index, dtype=torch.long, device=self.device)
        )
        if maximum_index.ndim == 0:
            maximum_index = maximum_index.expand(count)
        if maximum_index.shape != (count,):
            raise ValueError("maximum_terrain_index必须是标量或与env_ids等长的张量")
        maximum_index = maximum_index.clamp(0, len(TERRAIN_NAMES) - 1)
        eligible = self.enabled_terrain_indices[None, :] <= maximum_index[:, None]
        no_candidate = ~eligible.any(dim=1)
        if no_candidate.any():
            eligible[no_candidate, 0] = True
        selected_slots = torch.multinomial(eligible.float(), 1).squeeze(1)
        selected = self.enabled_terrain_indices[selected_slots]
        self.terrain_type[env_ids] = selected
        unit = torch.rand(count, device=self.device)
        amplitude = torch.zeros(count, device=self.device)

        def sample_range(bounds):
            return bounds[0] + (bounds[1] - bounds[0]) * unit

        amplitude = torch.where(selected == 1, sample_range(self.cfg.ramp_slope_range), amplitude)
        amplitude = torch.where(
            (selected == 2) | (selected == 3) | (selected == 8) | (selected == 9),
            sample_range(self.cfg.step_height_range_m),
            amplitude,
        )
        amplitude = torch.where(selected == 4, sample_range(self.cfg.rough_amplitude_range_m), amplitude)
        amplitude = torch.where(selected == 5, sample_range(self.cfg.pit_depth_range_m), amplitude)
        amplitude = torch.where(
            (selected == 6) | (selected == 7), sample_range(self.cfg.obstacle_height_range_m), amplitude
        )
        self.amplitude[env_ids] = amplitude
        route_yaw = -torch.pi + 2.0 * torch.pi * torch.rand(count, device=self.device)
        feature_distance = 0.8 + 1.0 * torch.rand(count, device=self.device)
        self.route_yaw[env_ids] = route_yaw
        self.feature_yaw[env_ids] = route_yaw
        self.feature_x[env_ids] = feature_distance * torch.cos(route_yaw)
        self.feature_y[env_ids] = feature_distance * torch.sin(route_yaw)
        width_min, width_max = self.cfg.step_width_range_m
        self.feature_width[env_ids] = width_min + (width_max - width_min) * torch.rand(count, device=self.device)
        friction_min, friction_max = self.cfg.friction_range
        self.friction[env_ids] = friction_min + (friction_max - friction_min) * torch.rand(
            count, device=self.device
        )
        self.occlusion_enabled[env_ids] = (
            torch.rand(count, device=self.device) < self.cfg.occlusion_sector_probability
        )
        self.occlusion_angle[env_ids] = -torch.pi + 2.0 * torch.pi * torch.rand(
            count, device=self.device
        )
        width_min, width_max = self.cfg.occlusion_width_range_rad
        self.occlusion_width[env_ids] = width_min + (width_max - width_min) * torch.rand(
            count, device=self.device
        )

    def sample_task_goals(
        self,
        robot_pose,
        minimum_distance_m: float,
        maximum_distance_m: float,
        env_ids=None,
    ):
        """Place each feature between the robot and a direction-guiding goal."""
        import torch

        if robot_pose.ndim != 2 or robot_pose.shape[-1] != 3:
            raise ValueError("robot_pose必须为[B,3]")
        if minimum_distance_m <= 0.0 or maximum_distance_m <= minimum_distance_m:
            raise ValueError("局部目标距离范围无效")
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if robot_pose.shape[0] != env_ids.numel():
            raise ValueError("robot_pose必须与env_ids等长")

        count = env_ids.numel()
        distance = minimum_distance_m + (maximum_distance_m - minimum_distance_m) * torch.rand(
            count, device=self.device
        )
        route_yaw = self.route_yaw[env_ids]
        forward = torch.stack((torch.cos(route_yaw), torch.sin(route_yaw)), dim=-1)
        lateral = torch.stack((-torch.sin(route_yaw), torch.cos(route_yaw)), dim=-1)
        goal_lateral = 0.15 * (2.0 * torch.rand(count, device=self.device) - 1.0)
        goals = robot_pose[:, :2] + distance[:, None] * forward + goal_lateral[:, None] * lateral

        feature_distance = torch.clamp(0.45 * distance, min=0.8, max=1.8)
        feature_lateral = 0.08 * (2.0 * torch.rand(count, device=self.device) - 1.0)
        feature_xy = (
            robot_pose[:, :2]
            + feature_distance[:, None] * forward
            + feature_lateral[:, None] * lateral
        )
        self.feature_x[env_ids] = feature_xy[:, 0]
        self.feature_y[env_ids] = feature_xy[:, 1]
        self.feature_yaw[env_ids] = route_yaw
        return goals

    def _environment_indices(self, batch_size: int, env_ids=None):
        import torch

        if env_ids is None:
            if batch_size != self.num_envs:
                raise ValueError("部分环境查询必须提供env_ids")
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.shape != (batch_size,):
            raise ValueError("env_ids必须与查询批次等长")
        return env_ids

    def _world_coordinates(self, pose, env_ids=None):
        import torch

        self._environment_indices(pose.shape[0], env_ids)
        yaw = pose[:, 2, None, None]
        cosine = torch.cos(yaw)
        sine = torch.sin(yaw)
        world_x = pose[:, 0, None, None] + cosine * self.local_x - sine * self.local_y
        world_y = pose[:, 1, None, None] + sine * self.local_x + cosine * self.local_y
        return world_x, world_y

    def _expanded(self, value, dimensions: int, env_ids):
        selected = value[env_ids]
        return selected.reshape((env_ids.numel(),) + (1,) * (dimensions - 1))

    def _terrain_surface(self, world_x, world_y, env_ids=None):
        import torch

        if world_x.shape != world_y.shape:
            raise ValueError("地形查询坐标形状错误")
        env_ids = self._environment_indices(world_x.shape[0], env_ids)
        dimensions = world_x.ndim
        terrain = self._expanded(self.terrain_type, dimensions, env_ids)
        amplitude = self._expanded(self.amplitude, dimensions, env_ids)
        feature_x = self._expanded(self.feature_x, dimensions, env_ids)
        feature_y = self._expanded(self.feature_y, dimensions, env_ids)
        feature_yaw = self._expanded(self.feature_yaw, dimensions, env_ids)
        width = self._expanded(self.feature_width, dimensions, env_ids)
        delta_x = world_x - feature_x
        delta_y = world_y - feature_y
        along = torch.cos(feature_yaw) * delta_x + torch.sin(feature_yaw) * delta_y
        cross = -torch.sin(feature_yaw) * delta_x + torch.cos(feature_yaw) * delta_y
        ground = torch.zeros_like(world_x)
        obstacle_range = torch.zeros_like(world_x)

        ramp = terrain == 1
        ramp_distance = torch.clamp(along, min=0.0, max=1.5)
        ground = torch.where(ramp, amplitude * ramp_distance, ground)

        step = terrain == 2
        ground = torch.where(step & (along > 0.0), amplitude, ground)

        stairs = terrain == 3
        stair_count = torch.clamp(torch.floor(along / width) + 1.0, 0.0, 4.0)
        ground = torch.where(stairs, stair_count * amplitude * 0.45, ground)

        rough = terrain == 4
        rough_patch = (torch.abs(along) < 1.25) & (torch.abs(cross) < 1.2)
        rough_height = amplitude * torch.sin(6.0 * along) * torch.cos(5.0 * cross)
        ground = torch.where(rough & rough_patch, rough_height, ground)

        pit = terrain == 5
        in_pit = (torch.abs(along) < width) & (torch.abs(cross) < 1.2 * width)
        ground = torch.where(pit & in_pit, -amplitude, ground)

        wall = terrain == 6
        wall_cells = (torch.abs(along) < 0.10) & (torch.abs(cross) < 1.8)
        obstacle_range = torch.where(wall & wall_cells, amplitude, obstacle_range)

        pillar = terrain == 7
        pillar_cells = along.square() + cross.square() < width.square()
        obstacle_range = torch.where(pillar & pillar_cells, amplitude, obstacle_range)

        mixed = terrain == 8
        mixed_step = mixed & (along > 0.0) & (cross < 0.0)
        mixed_wall = mixed & (torch.abs(along - 0.8) < 0.12) & (cross > 0.0)
        ground = torch.where(mixed_step, amplitude, ground)
        obstacle_range = torch.where(mixed_wall, 0.8 + amplitude, obstacle_range)

        multi_route = terrain == 9
        low_barrier = (torch.abs(along) < 0.18) & (torch.abs(cross) < 1.6)
        obstacle_range = torch.where(multi_route & low_barrier, amplitude, obstacle_range)
        return ground, obstacle_range

    def _height_and_obstacle_range(self, world_x, world_y, grid_yaw=None, env_ids=None):
        """Approximate each cell's point-wise max-minus-min height."""
        import torch

        env_ids = self._environment_indices(world_x.shape[0], env_ids)
        ground, obstacle_range = self._terrain_surface(world_x, world_y, env_ids)
        minimum_ground = ground
        maximum_ground = ground
        maximum_obstacle = obstacle_range
        grid_cosine = None
        grid_sine = None
        if grid_yaw is not None:
            if grid_yaw.shape != (env_ids.numel(),):
                raise ValueError("grid_yaw形状必须为[B]")
            yaw = grid_yaw.reshape((env_ids.numel(),) + (1,) * (world_x.ndim - 1))
            grid_cosine = torch.cos(yaw)
            grid_sine = torch.sin(yaw)
        for offset_x, offset_y in (
            (-self.half_cell_m, -self.half_cell_m),
            (-self.half_cell_m, self.half_cell_m),
            (self.half_cell_m, -self.half_cell_m),
            (self.half_cell_m, self.half_cell_m),
        ):
            if grid_yaw is None:
                world_offset_x = offset_x
                world_offset_y = offset_y
            else:
                world_offset_x = grid_cosine * offset_x - grid_sine * offset_y
                world_offset_y = grid_sine * offset_x + grid_cosine * offset_y
            sample_ground, sample_obstacle = self._terrain_surface(
                world_x + world_offset_x, world_y + world_offset_y, env_ids
            )
            minimum_ground = torch.minimum(minimum_ground, sample_ground)
            maximum_ground = torch.maximum(maximum_ground, sample_ground)
            maximum_obstacle = torch.maximum(maximum_obstacle, sample_obstacle)
        height_range = torch.maximum(maximum_ground - minimum_ground, maximum_obstacle)
        return ground, height_range

    def ground_reference(self, pose, env_ids=None):
        """Return absolute analytic supporting-ground height below each robot."""
        if pose.ndim != 2 or pose.shape[1] != 3:
            raise ValueError("pose形状必须为[B,3]")
        env_ids = self._environment_indices(pose.shape[0], env_ids)
        ground, _ = self._terrain_surface(
            pose[:, 0, None], pose[:, 1, None], env_ids
        )
        return ground[:, 0]

    def true_motion_metrics(self, pose, linear_velocity, env_ids=None) -> TerrainTruthMetrics:
        """Query noise-free terrain beneath the body and its short swept footprint."""
        import torch

        if pose.ndim != 2 or pose.shape[1] != 3 or linear_velocity.shape != (pose.shape[0],):
            raise ValueError("真值运动查询尺寸错误")
        env_ids = self._environment_indices(pose.shape[0], env_ids)
        batch_size = env_ids.numel()
        direction = torch.where(linear_velocity >= 0.0, 1.0, -1.0)
        body_x = self.body_x_samples.to(dtype=pose.dtype).expand(batch_size, -1)
        lookahead_x = direction[:, None] * (
            self.cfg.robot_half_length_m + self.cfg.truth_lookahead_m
        )
        body_x = torch.cat((body_x, lookahead_x), dim=1)
        body_y = self.body_y_samples.to(dtype=pose.dtype)
        sample_x = body_x[:, :, None].expand(-1, -1, body_y.numel())
        sample_y = body_y[None, None, :].expand(batch_size, body_x.shape[1], -1)
        yaw = pose[:, 2, None, None]
        world_x = pose[:, 0, None, None] + torch.cos(yaw) * sample_x - torch.sin(yaw) * sample_y
        world_y = pose[:, 1, None, None] + torch.sin(yaw) * sample_x + torch.cos(yaw) * sample_y
        ground, obstacle = self._terrain_surface(world_x, world_y, env_ids)
        ground_span = ground.amax(dim=(-2, -1)) - ground.amin(dim=(-2, -1))
        terrain = self.terrain_type[env_ids]
        amplitude = self.amplitude[env_ids]
        crosses_discontinuity = ground_span > 1.0e-5
        single_discontinuity = torch.zeros_like(ground_span)
        single_discontinuity = torch.where(
            crosses_discontinuity & (terrain == 2), amplitude, single_discontinuity
        )
        single_discontinuity = torch.where(
            crosses_discontinuity & (terrain == 3), 0.45 * amplitude, single_discontinuity
        )
        single_discontinuity = torch.where(
            crosses_discontinuity & ((terrain == 5) | (terrain == 8)),
            amplitude,
            single_discontinuity,
        )
        inside_pit = (terrain == 5) & (ground.amin(dim=(-2, -1)) < -1.0e-5)
        pit_depth = torch.where(inside_pit, amplitude, torch.zeros_like(amplitude))
        entry_alignment = torch.abs(torch.cos(pose[:, 2] - self.feature_yaw[env_ids]))
        return TerrainTruthMetrics(
            support_span_m=ground_span,
            maximum_discontinuity_m=single_discontinuity,
            obstacle_height_m=obstacle.amax(dim=(-2, -1)),
            pit_depth_m=pit_depth,
            entry_alignment=entry_alignment,
            ground_reference_z=self.ground_reference(pose, env_ids),
        )

    def generate(
        self,
        pose,
        velocity=None,
        *,
        return_ground_reference: bool = False,
        env_ids=None,
    ):
        """Return normalized maps with shape ``[B,4,H,W]``."""
        import torch
        import torch.nn.functional as functional

        if pose.ndim != 2 or pose.shape[1] != 3:
            raise ValueError("pose形状必须为[B,3]")
        env_ids = self._environment_indices(pose.shape[0], env_ids)
        batch_size = env_ids.numel()
        noisy_pose = pose.clone()
        noisy_pose[:, :2] += self.cfg.pose_xy_noise_std_m * torch.randn_like(noisy_pose[:, :2])
        noisy_pose[:, 2] += self.cfg.pose_yaw_noise_std_rad * torch.randn_like(noisy_pose[:, 2])
        if velocity is not None:
            if velocity.shape != (batch_size, 2):
                raise ValueError("velocity形状必须为[B,2]")
            time_offset = self.cfg.time_jitter_std_s * torch.randn_like(noisy_pose[:, 0])
            noisy_pose[:, 0] += velocity[:, 0] * torch.cos(noisy_pose[:, 2]) * time_offset
            noisy_pose[:, 1] += velocity[:, 0] * torch.sin(noisy_pose[:, 2]) * time_offset
            noisy_pose[:, 2] += velocity[:, 1] * time_offset
        world_x, world_y = self._world_coordinates(noisy_pose, env_ids)
        ground, height_range = self._height_and_obstacle_range(
            world_x, world_y, noisy_pose[:, 2], env_ids
        )

        # Actor heights are referenced to the supporting ground below the
        # robot, not to the IMU/body origin above it.
        ground_below_robot, _ = self._terrain_surface(
            noisy_pose[:, 0, None], noisy_pose[:, 1, None], env_ids
        )
        ground_reference_z = ground_below_robot[:, 0]
        measured_ground = (
            ground
            - ground_reference_z[:, None, None]
            + self.cfg.height_noise_std_m * torch.randn_like(ground)
        )
        height_range = torch.clamp(
            height_range + self.cfg.range_noise_std_m * torch.randn_like(height_range), min=0.0
        )

        coarse_size = max(2, self.cfg.size // 8)
        coarse_observation = torch.rand(
            (batch_size, 1, coarse_size, coarse_size),
            dtype=ground.dtype,
            device=ground.device,
        )
        observation_sample = functional.interpolate(
            coarse_observation, size=ground.shape[-2:], mode="nearest"
        )[:, 0]
        unknown = observation_sample < self.cfg.missing_probability
        angle_delta = torch.atan2(
            torch.sin(self.local_angle - self.occlusion_angle[env_ids, None, None]),
            torch.cos(self.local_angle - self.occlusion_angle[env_ids, None, None]),
        )
        occluded = (
            self.occlusion_enabled[env_ids, None, None]
            & (self.local_radius > 1.0)
            & (torch.abs(angle_delta) < 0.5 * self.occlusion_width[env_ids, None, None])
        )
        unknown |= occluded
        ray_only = (
            observation_sample >= self.cfg.missing_probability
        ) & (
            observation_sample
            < self.cfg.missing_probability + self.cfg.ray_only_probability
        ) & ~unknown
        valid = ~(unknown | ray_only)
        observed = valid | ray_only
        measured_ground = torch.where(
            valid, measured_ground, torch.full_like(measured_ground, float("nan"))
        )
        measured_range = torch.where(
            valid, height_range, torch.full_like(height_range, float("nan"))
        )
        from .grid_preprocessor import preprocess_grid_map_torch

        result = preprocess_grid_map_torch(
            measured_ground,
            measured_range,
            observed.float(),
            max_abs_relative_height=self.cfg.maximum_relative_height_m,
            max_height_range=self.cfg.maximum_height_range_m,
            ground_fill_value=self.cfg.ground_fill_value,
            range_fill_value=self.cfg.range_fill_value,
            normalize=self.cfg.normalize_heights,
        )
        if not torch.isfinite(result).all():
            raise RuntimeError("仿真局部地图包含NaN或Inf")
        if return_ground_reference:
            return result, ground_reference_z
        return result

    def forward_risk(self, local_map, linear_velocity=None) -> tuple:
        """Return terrain risk in the current forward or reverse motion direction."""
        import torch

        size = local_map.shape[-1]
        center = size // 2
        depth = max(2, size // 10)
        y0, y1 = size // 2 - max(1, size // 16), size // 2 + max(1, size // 16)
        front = local_map[:, :, center : min(size, center + depth), y0:y1]
        rear = local_map[:, :, max(0, center - depth) : center, y0:y1]
        front_risk = front[:, 1].amax(dim=(-2, -1))
        rear_risk = rear[:, 1].amax(dim=(-2, -1))
        front_unknown = 1.0 - front[:, 2].mean(dim=(-2, -1))
        rear_unknown = 1.0 - rear[:, 2].mean(dim=(-2, -1))
        if linear_velocity is None:
            terrain_risk = front_risk
            unknown_ratio = front_unknown
        else:
            forward = linear_velocity >= 0.0
            terrain_risk = torch.where(forward, front_risk, rear_risk)
            unknown_ratio = torch.where(forward, front_unknown, rear_unknown)
        if not self.cfg.normalize_heights:
            terrain_risk = torch.clamp(
                terrain_risk / self.cfg.maximum_height_range_m, min=0.0, max=1.0
            )
        return terrain_risk, unknown_ratio
