"""Privileged terrain truth used by the high-level kinematic execution model.

This module never produces actor observations. Actor maps are built only from
native Isaac Lab RayCaster returns; the formulas here mirror the selected
physical mesh tile for reward, termination, and proxy-motion calculations.
"""

from __future__ import annotations

from dataclasses import dataclass

from go2w_terrain_planner.utils.tensor_checks import require_finite

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
    "low_obstacle",
)


def fuse_aligned_map_history(
    aligned_maps,
    *,
    maximum_ground_deviation: float = 0.04,
    span_percentile: float = 0.75,
):
    """Fuse the densest temporally consistent ground cluster per cell."""
    import torch

    if aligned_maps.ndim != 5 or aligned_maps.shape[2] != 4:
        raise ValueError("aligned_maps必须为[B,T,4,H,W]")
    if maximum_ground_deviation <= 0.0:
        raise ValueError("maximum_ground_deviation必须大于0")
    if not 0.0 <= span_percentile <= 1.0:
        raise ValueError("span_percentile必须位于[0,1]")
    valid = aligned_maps[:, :, 3] > 0.5
    ground_history = aligned_maps[:, :, 0]
    history_length = aligned_maps.shape[1]
    cluster_counts = []
    for anchor_index in range(history_length):
        anchor = ground_history[:, anchor_index : anchor_index + 1]
        cluster_counts.append(
            (
                valid
                & valid[:, anchor_index : anchor_index + 1]
                & ((ground_history - anchor).abs() <= maximum_ground_deviation)
            ).sum(dim=1)
        )
    cluster_counts = torch.stack(cluster_counts, dim=1)
    recency = torch.arange(
        history_length,
        dtype=cluster_counts.dtype,
        device=cluster_counts.device,
    )[None, :, None, None]
    anchor_score = cluster_counts * (history_length + 1) + recency
    selected_anchor_index = anchor_score.argmax(dim=1, keepdim=True)
    selected_ground = ground_history.gather(1, selected_anchor_index).squeeze(1)
    consistent = valid & (
        (ground_history - selected_ground[:, None]).abs()
        <= maximum_ground_deviation
    )
    count = consistent.sum(dim=1)
    ground_samples = torch.where(
        consistent,
        ground_history,
        torch.full_like(ground_history, torch.inf),
    ).sort(dim=1).values
    ground_rank = ((count - 1).clamp(min=0) // 2).unsqueeze(1)
    ground = ground_samples.gather(1, ground_rank).squeeze(1)

    range_samples = torch.where(
        consistent,
        aligned_maps[:, :, 1],
        torch.full_like(aligned_maps[:, :, 1], torch.inf),
    ).sort(dim=1).values
    # 上四分位数保留稳定的局部高度突变，同时抑制单帧离群峰值。
    range_rank = torch.ceil(
        span_percentile
        * (count - 1).clamp(min=0).to(aligned_maps.dtype)
    ).long().unsqueeze(1)
    height_range = range_samples.gather(1, range_rank).squeeze(1)
    observed = aligned_maps[:, :, 2].amax(dim=1)
    height_valid = count > 0
    ground = torch.where(height_valid, ground, torch.zeros_like(ground))
    height_range = torch.where(height_valid, height_range, torch.zeros_like(height_range))
    result = torch.stack((ground, height_range, observed, height_valid.float()), dim=1)
    require_finite(result, "融合后的仿真地图")
    return result


@dataclass
class TerrainTruthConfig:
    extent_m: float = 10.0
    size: int = 200
    maximum_relative_height_m: float = 1.0
    maximum_height_range_m: float = 3.0
    ground_fill_value: float = 0.0
    range_fill_value: float = 0.0
    normalize_heights: bool = True
    enabled_terrain_names: tuple[str, ...] = TERRAIN_NAMES
    ramp_slope_range: tuple[float, float] = (0.05, 0.40)
    step_height_range_m: tuple[float, float] = (0.05, 0.35)
    step_width_range_m: tuple[float, float] = (0.25, 0.60)
    rough_amplitude_range_m: tuple[float, float] = (0.01, 0.12)
    pit_depth_range_m: tuple[float, float] = (0.05, 0.40)
    pit_half_width_range_m: tuple[float, float] = (0.25, 0.60)
    pit_curriculum_start_depth_max_m: float = 0.12
    pit_curriculum_start_half_width_max_m: float = 0.35
    pit_navigation_avoidance_depth_m: float = 0.18
    pit_goal_clearance_m: float = 0.80
    obstacle_height_range_m: tuple[float, float] = (0.55, 2.50)
    low_obstacle_height_range_m: tuple[float, float] = (0.06, 0.30)
    barrier_half_width_range_m: tuple[float, float] = (0.40, 2.00)
    challenge_barrier_width_scale: float = 0.55
    barrier_navigation_clearance_m: float = 0.35
    friction_range: tuple[float, float] = (0.35, 1.00)
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
            torch.maximum(
                torch.maximum(
                    self.support_span_m,
                    self.maximum_discontinuity_m,
                ),
                self.obstacle_height_m,
            ),
            self.pit_depth_m,
        )

    @property
    def collision_height_m(self):
        import torch

        return torch.maximum(self.maximum_discontinuity_m, self.obstacle_height_m)


@dataclass
class NavigationGuidance:
    """Privileged reward-shaping path solution around the active obstacle."""

    potential_m: object
    target_xy: object
    blocked: object


class TerrainTruthModel:
    """Track mesh-matched privileged terrain state on the training device."""

    def __init__(self, num_envs: int, device, cfg: TerrainTruthConfig | None = None) -> None:
        import torch

        self.cfg = cfg or TerrainTruthConfig()
        if num_envs <= 0 or self.cfg.size <= 1 or self.cfg.extent_m <= 0.0:
            raise ValueError("仿真地图配置无效")
        if self.cfg.maximum_relative_height_m <= 0.0 or self.cfg.maximum_height_range_m <= 0.0:
            raise ValueError("仿真地图高度归一化范围必须大于0")
        pit_depth_minimum, pit_depth_maximum = self.cfg.pit_depth_range_m
        pit_width_minimum, pit_width_maximum = self.cfg.pit_half_width_range_m
        if not (
            0.0 < pit_depth_minimum
            <= self.cfg.pit_curriculum_start_depth_max_m
            <= pit_depth_maximum
            and 0.0 < pit_width_minimum
            <= self.cfg.pit_curriculum_start_half_width_max_m
            <= pit_width_maximum
            and pit_depth_minimum
            <= self.cfg.pit_navigation_avoidance_depth_m
            <= pit_depth_maximum
            and self.cfg.pit_goal_clearance_m > 0.0
            and self.cfg.barrier_navigation_clearance_m > 0.0
        ):
            raise ValueError("pit渐进课程参数无效")
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
        self.barrier_half_width = torch.ones(num_envs, device=self.device)
        self.terrain_difficulty = torch.ones(num_envs, device=self.device)
        self.domain_randomization_scale = torch.zeros(num_envs, device=self.device)
        self.traversal_direction = torch.ones(num_envs, device=self.device)
        self.friction = torch.ones(num_envs, device=self.device)
        self.geometry_is_fixed = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )
        self.reset(torch.arange(num_envs, device=self.device))

    def apply_physical_geometry(self, env_ids, parameters: dict[str, object]) -> None:
        """Apply the exact metadata of the selected static mesh terrain tile."""
        import torch

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        required = (
            "terrain_type",
            "difficulty",
            "amplitude",
            "feature_x",
            "feature_y",
            "feature_yaw",
            "route_yaw",
            "feature_width",
            "barrier_half_width",
            "traversal_direction",
        )
        missing = set(required) - parameters.keys()
        if missing:
            raise ValueError(f"物理地形元数据缺少字段：{sorted(missing)}")
        for name in required:
            target_name = "terrain_difficulty" if name == "difficulty" else name
            target = getattr(self, target_name)
            dtype = torch.long if name == "terrain_type" else torch.float32
            value = torch.as_tensor(
                parameters[name], dtype=dtype, device=self.device
            )
            if value.shape != (env_ids.numel(),):
                raise ValueError(f"物理地形字段{name}必须与env_ids等长")
            target[env_ids] = value
        self.geometry_is_fixed[env_ids] = True

    def reset(
        self,
        env_ids,
        terrain_probabilities=None,
        terrain_difficulty=1.0,
        domain_randomization_scale=0.0,
    ) -> None:
        import torch

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        count = env_ids.numel()
        selected_difficulty = torch.as_tensor(
            terrain_difficulty, dtype=torch.float32, device=self.device
        )
        randomization_scale = torch.as_tensor(
            domain_randomization_scale, dtype=torch.float32, device=self.device
        )
        if selected_difficulty.ndim == 0:
            selected_difficulty = selected_difficulty.expand(count)
        if randomization_scale.ndim == 0:
            randomization_scale = randomization_scale.expand(count)
        if (
            selected_difficulty.shape != (count,)
            or randomization_scale.shape != (count,)
            or torch.any(
                (selected_difficulty < 0.0) | (selected_difficulty > 1.0)
            )
            or torch.any(
                (randomization_scale < 0.0) | (randomization_scale > 1.0)
            )
        ):
            raise ValueError("地形难度与域随机化强度必须位于[0,1]")
        if terrain_probabilities is None:
            sampling_weights = torch.zeros(
                (count, len(TERRAIN_NAMES)), device=self.device
            )
            sampling_weights[:, self.enabled_terrain_indices] = 1.0
        else:
            sampling_weights = torch.as_tensor(
                terrain_probabilities, dtype=torch.float32, device=self.device
            )
            if sampling_weights.shape != (count, len(TERRAIN_NAMES)):
                raise ValueError(
                    "terrain_probabilities必须为[B, terrain_count]"
                )
            enabled_mask = torch.zeros(
                len(TERRAIN_NAMES), dtype=torch.bool, device=self.device
            )
            enabled_mask[self.enabled_terrain_indices] = True
            sampling_weights = torch.where(
                enabled_mask[None, :],
                sampling_weights,
                torch.zeros_like(sampling_weights),
            )
        if (
            not torch.isfinite(sampling_weights).all()
            or torch.any(sampling_weights < 0.0)
            or torch.any(sampling_weights.sum(dim=1) <= 0.0)
        ):
            raise ValueError("地形采样权重无效或指向未启用地形")
        selected = torch.multinomial(sampling_weights, 1).squeeze(1)
        self.terrain_type[env_ids] = selected
        self.geometry_is_fixed[env_ids] = False
        self.terrain_difficulty[env_ids] = selected_difficulty
        self.domain_randomization_scale[env_ids] = randomization_scale
        stair_index = TERRAIN_NAMES.index("stairs")
        stair_descent = (selected == stair_index) & (
            torch.rand(count, device=self.device) < 0.5
        )
        self.traversal_direction[env_ids] = torch.where(
            stair_descent,
            -torch.ones(count, device=self.device),
            torch.ones(count, device=self.device),
        )
        unit = torch.rand(count, device=self.device)
        amplitude = torch.zeros(count, device=self.device)

        def sample_progressive_range(bounds):
            return bounds[0] + selected_difficulty * (bounds[1] - bounds[0]) * unit

        amplitude = torch.where(
            selected == TERRAIN_NAMES.index("ramp"),
            sample_progressive_range(self.cfg.ramp_slope_range),
            amplitude,
        )
        amplitude = torch.where(
            (selected == TERRAIN_NAMES.index("step"))
            | (selected == TERRAIN_NAMES.index("stairs"))
            | (selected == TERRAIN_NAMES.index("mixed"))
            | (selected == TERRAIN_NAMES.index("multi_route")),
            sample_progressive_range(self.cfg.step_height_range_m),
            amplitude,
        )
        amplitude = torch.where(
            selected == TERRAIN_NAMES.index("rough"),
            sample_progressive_range(self.cfg.rough_amplitude_range_m),
            amplitude,
        )
        pit_depth_minimum, pit_depth_maximum = self.cfg.pit_depth_range_m
        progressive_pit_depth_maximum = (
            self.cfg.pit_curriculum_start_depth_max_m
            + selected_difficulty
            * (
                pit_depth_maximum
                - self.cfg.pit_curriculum_start_depth_max_m
            )
        )
        pit_depth = pit_depth_minimum + (
            progressive_pit_depth_maximum - pit_depth_minimum
        ) * torch.rand(count, device=self.device)
        amplitude = torch.where(
            selected == TERRAIN_NAMES.index("pit"), pit_depth, amplitude
        )
        amplitude = torch.where(
            (selected == TERRAIN_NAMES.index("wall"))
            | (selected == TERRAIN_NAMES.index("pillar")),
            sample_progressive_range(self.cfg.obstacle_height_range_m),
            amplitude,
        )
        amplitude = torch.where(
            selected == TERRAIN_NAMES.index("low_obstacle"),
            sample_progressive_range(self.cfg.low_obstacle_height_range_m),
            amplitude,
        )
        self.amplitude[env_ids] = amplitude
        route_yaw = -torch.pi + 2.0 * torch.pi * torch.rand(count, device=self.device)
        feature_distance = 0.8 + 1.0 * torch.rand(count, device=self.device)
        self.route_yaw[env_ids] = route_yaw
        self.feature_yaw[env_ids] = route_yaw
        self.feature_x[env_ids] = feature_distance * torch.cos(route_yaw)
        self.feature_y[env_ids] = feature_distance * torch.sin(route_yaw)
        width_min, width_max = self.cfg.step_width_range_m
        feature_width = width_min + (width_max - width_min) * torch.rand(
            count, device=self.device
        )
        pit_width_minimum, pit_width_maximum = self.cfg.pit_half_width_range_m
        progressive_pit_width_maximum = (
            self.cfg.pit_curriculum_start_half_width_max_m
            + selected_difficulty
            * (
                pit_width_maximum
                - self.cfg.pit_curriculum_start_half_width_max_m
            )
        )
        pit_width = pit_width_minimum + (
            progressive_pit_width_maximum - pit_width_minimum
        ) * torch.rand(count, device=self.device)
        self.feature_width[env_ids] = torch.where(
            selected == TERRAIN_NAMES.index("pit"), pit_width, feature_width
        )
        barrier_min, barrier_max = self.cfg.barrier_half_width_range_m
        barrier_width = barrier_min + (
            barrier_max - barrier_min
        ) * torch.rand(count, device=self.device)
        progressive_barrier_scale = (
            self.cfg.challenge_barrier_width_scale
            + selected_difficulty
            * (1.0 - self.cfg.challenge_barrier_width_scale)
        )
        uses_barrier_width = (
            (selected == TERRAIN_NAMES.index("wall"))
            | (selected == TERRAIN_NAMES.index("mixed"))
            | (selected == TERRAIN_NAMES.index("multi_route"))
        )
        barrier_width = torch.where(
            uses_barrier_width,
            barrier_width * progressive_barrier_scale,
            barrier_width,
        )
        self.barrier_half_width[env_ids] = barrier_width
        friction_min, friction_max = self.cfg.friction_range
        randomized_friction = friction_min + (
            friction_max - friction_min
        ) * torch.rand(count, device=self.device)
        self.friction[env_ids] = 1.0 + randomization_scale * (
            randomized_friction - 1.0
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
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if robot_pose.shape[0] != env_ids.numel():
            raise ValueError("robot_pose必须与env_ids等长")

        count = env_ids.numel()
        minimum_distance = torch.as_tensor(
            minimum_distance_m, dtype=robot_pose.dtype, device=self.device
        )
        maximum_distance = torch.as_tensor(
            maximum_distance_m, dtype=robot_pose.dtype, device=self.device
        )
        if minimum_distance.ndim == 0:
            minimum_distance = minimum_distance.expand(count)
        if maximum_distance.ndim == 0:
            maximum_distance = maximum_distance.expand(count)
        if (
            minimum_distance.shape != (count,)
            or maximum_distance.shape != (count,)
            or torch.any(minimum_distance < 0.0)
            or torch.any(maximum_distance <= minimum_distance)
        ):
            raise ValueError("局部目标距离范围无效")
        route_yaw = self.route_yaw[env_ids]
        forward = torch.stack(
            (torch.cos(route_yaw), torch.sin(route_yaw)), dim=-1
        )
        feature_relative = torch.stack(
            (
                self.feature_x[env_ids] - robot_pose[:, 0],
                self.feature_y[env_ids] - robot_pose[:, 1],
            ),
            dim=-1,
        )
        feature_forward_distance = (
            feature_relative * forward
        ).sum(dim=-1).clamp(min=0.0)
        pit_minimum_distance = (
            feature_forward_distance
            + self.feature_width[env_ids]
            + self.cfg.pit_goal_clearance_m
        )
        pit_minimum_distance = torch.maximum(
            pit_minimum_distance,
            (
                self.feature_width[env_ids]
                + self.cfg.pit_goal_clearance_m
            )
            / 0.55,
        )
        effective_minimum_distance = torch.where(
            self.terrain_type[env_ids] == 5,
            torch.maximum(minimum_distance, pit_minimum_distance),
            minimum_distance,
        )
        if torch.any(maximum_distance <= effective_minimum_distance):
            raise ValueError("目标距离上限不足以把目标放置在pit远端安全区域")
        distance = effective_minimum_distance + (
            maximum_distance - effective_minimum_distance
        ) * torch.rand(count, device=self.device)
        lateral = torch.stack((-torch.sin(route_yaw), torch.cos(route_yaw)), dim=-1)
        goal_lateral = 0.15 * (2.0 * torch.rand(count, device=self.device) - 1.0)
        goals = robot_pose[:, :2] + distance[:, None] * forward + goal_lateral[:, None] * lateral

        feature_distance = torch.clamp(0.45 * distance, min=0.8, max=4.0)
        feature_distance = torch.minimum(feature_distance, 0.75 * distance)
        feature_lateral = 0.08 * (2.0 * torch.rand(count, device=self.device) - 1.0)
        feature_xy = (
            robot_pose[:, :2]
            + feature_distance[:, None] * forward
            + feature_lateral[:, None] * lateral
        )
        movable = ~self.geometry_is_fixed[env_ids]
        self.feature_x[env_ids] = torch.where(
            movable, feature_xy[:, 0], self.feature_x[env_ids]
        )
        self.feature_y[env_ids] = torch.where(
            movable, feature_xy[:, 1], self.feature_y[env_ids]
        )
        self.feature_yaw[env_ids] = torch.where(
            movable, route_yaw, self.feature_yaw[env_ids]
        )
        return goals

    @staticmethod
    def _segment_intersects_rectangle(
        start_along,
        start_cross,
        end_along,
        end_cross,
        minimum_along,
        maximum_along,
        minimum_cross,
        maximum_cross,
    ):
        """Vectorized line-segment/AABB intersection in obstacle coordinates."""
        import torch

        def slab(start, delta, minimum, maximum):
            parallel = delta.abs() <= 1.0e-6
            safe_delta = torch.where(parallel, torch.ones_like(delta), delta)
            first = (minimum - start) / safe_delta
            second = (maximum - start) / safe_delta
            enter = torch.minimum(first, second)
            leave = torch.maximum(first, second)
            inside = (start >= minimum) & (start <= maximum)
            positive_infinity = torch.full_like(start, torch.inf)
            negative_infinity = torch.full_like(start, -torch.inf)
            enter = torch.where(
                parallel,
                torch.where(inside, negative_infinity, positive_infinity),
                enter,
            )
            leave = torch.where(
                parallel,
                torch.where(inside, positive_infinity, negative_infinity),
                leave,
            )
            return enter, leave

        delta_along = end_along - start_along
        delta_cross = end_cross - start_cross
        along_enter, along_leave = slab(
            start_along, delta_along, minimum_along, maximum_along
        )
        cross_enter, cross_leave = slab(
            start_cross, delta_cross, minimum_cross, maximum_cross
        )
        enter = torch.maximum(
            torch.maximum(along_enter, cross_enter),
            torch.zeros_like(start_along),
        )
        leave = torch.minimum(
            torch.minimum(along_leave, cross_leave),
            torch.ones_like(start_along),
        )
        return enter <= leave

    def navigation_guidance(self, pose, goal_xy, env_ids=None) -> NavigationGuidance:
        """Return collision-free reward potential and its immediate target.

        Hazardous pits, walls, pillars, mixed-scene walls, and multi-route
        barriers are approximated by clearance-expanded rectangles. If the
        direct goal segment intersects the active rectangle, the shorter path
        around its two lateral sides supplies both the shaping potential and
        the direction used by heading rewards.
        """
        import torch

        if pose.ndim != 2 or pose.shape[1] != 3:
            raise ValueError("pose形状必须为[B,3]")
        if goal_xy.shape != (pose.shape[0], 2):
            raise ValueError("goal_xy形状必须为[B,2]")
        env_ids = self._environment_indices(pose.shape[0], env_ids)
        direct = torch.linalg.vector_norm(goal_xy - pose[:, :2], dim=-1)
        terrain = self.terrain_type[env_ids]
        width = self.feature_width[env_ids]
        barrier_width = self.barrier_half_width[env_ids]
        clearance = float(self.cfg.barrier_navigation_clearance_m)

        feature_x = self.feature_x[env_ids]
        feature_y = self.feature_y[env_ids]
        feature_yaw = self.feature_yaw[env_ids]
        cosine = torch.cos(feature_yaw)
        sine = torch.sin(feature_yaw)
        robot_dx = pose[:, 0] - feature_x
        robot_dy = pose[:, 1] - feature_y
        goal_dx = goal_xy[:, 0] - feature_x
        goal_dy = goal_xy[:, 1] - feature_y
        robot_along = cosine * robot_dx + sine * robot_dy
        robot_cross = -sine * robot_dx + cosine * robot_dy
        goal_along = cosine * goal_dx + sine * goal_dy
        goal_cross = -sine * goal_dx + cosine * goal_dy

        zeros = torch.zeros_like(direct)
        minimum_along = zeros.clone()
        maximum_along = zeros.clone()
        minimum_cross = zeros.clone()
        maximum_cross = zeros.clone()
        active = torch.zeros_like(terrain, dtype=torch.bool)

        pit = (terrain == 5) & (
            self.amplitude[env_ids] >= self.cfg.pit_navigation_avoidance_depth_m
        )
        pit_along_half = width + clearance
        pit_cross_half = 1.2 * width + clearance
        minimum_along = torch.where(pit, -pit_along_half, minimum_along)
        maximum_along = torch.where(pit, pit_along_half, maximum_along)
        minimum_cross = torch.where(pit, -pit_cross_half, minimum_cross)
        maximum_cross = torch.where(pit, pit_cross_half, maximum_cross)
        active |= pit

        wall_like = (terrain == 6) | (terrain == 9)
        wall_along_half = torch.full_like(direct, 0.10 + clearance)
        wall_cross_half = barrier_width + clearance
        minimum_along = torch.where(wall_like, -wall_along_half, minimum_along)
        maximum_along = torch.where(wall_like, wall_along_half, maximum_along)
        minimum_cross = torch.where(wall_like, -wall_cross_half, minimum_cross)
        maximum_cross = torch.where(wall_like, wall_cross_half, maximum_cross)
        active |= wall_like

        pillar = terrain == 7
        pillar_half = width + clearance
        minimum_along = torch.where(pillar, -pillar_half, minimum_along)
        maximum_along = torch.where(pillar, pillar_half, maximum_along)
        minimum_cross = torch.where(pillar, -pillar_half, minimum_cross)
        maximum_cross = torch.where(pillar, pillar_half, maximum_cross)
        active |= pillar

        mixed = terrain == 8
        mixed_along_half = torch.full_like(direct, 0.12 + clearance)
        minimum_along = torch.where(
            mixed, 0.8 - mixed_along_half, minimum_along
        )
        maximum_along = torch.where(
            mixed, 0.8 + mixed_along_half, maximum_along
        )
        minimum_cross = torch.where(
            mixed, torch.full_like(direct, -clearance), minimum_cross
        )
        maximum_cross = torch.where(
            mixed, barrier_width + clearance, maximum_cross
        )
        active |= mixed

        blocked = active & self._segment_intersects_rectangle(
            robot_along,
            robot_cross,
            goal_along,
            goal_cross,
            minimum_along,
            maximum_along,
            minimum_cross,
            maximum_cross,
        )
        forward = goal_along >= robot_along
        entry_along = torch.where(forward, minimum_along, maximum_along)
        exit_along = torch.where(forward, maximum_along, minimum_along)
        edge_length = (exit_along - entry_along).abs()
        upper_path = (
            torch.hypot(robot_along - entry_along, robot_cross - maximum_cross)
            + edge_length
            + torch.hypot(goal_along - exit_along, goal_cross - maximum_cross)
        )
        lower_path = (
            torch.hypot(robot_along - entry_along, robot_cross - minimum_cross)
            + edge_length
            + torch.hypot(goal_along - exit_along, goal_cross - minimum_cross)
        )
        use_upper = upper_path <= lower_path
        target_cross = torch.where(use_upper, maximum_cross, minimum_cross)
        at_selected_side = torch.where(
            use_upper,
            robot_cross >= maximum_cross - 1.0e-4,
            robot_cross <= minimum_cross + 1.0e-4,
        )
        past_entry = torch.where(
            forward,
            robot_along >= entry_along - 1.0e-4,
            robot_along <= entry_along + 1.0e-4,
        )
        use_exit_target = at_selected_side & past_entry
        target_along = torch.where(
            use_exit_target, exit_along, entry_along
        )
        cleared = at_selected_side & torch.where(
            forward,
            robot_along >= exit_along - 1.0e-4,
            robot_along <= exit_along + 1.0e-4,
        )
        blocked &= ~cleared
        target_world = torch.stack(
            (
                feature_x + cosine * target_along - sine * target_cross,
                feature_y + sine * target_along + cosine * target_cross,
            ),
            dim=-1,
        )
        full_detour = torch.where(use_upper, upper_path, lower_path)
        exit_detour = (
            torch.hypot(
                robot_along - exit_along,
                robot_cross - target_cross,
            )
            + torch.hypot(
                goal_along - exit_along,
                goal_cross - target_cross,
            )
        )
        detour = torch.where(use_exit_target, exit_detour, full_detour)
        return NavigationGuidance(
            potential_m=torch.where(blocked, detour, direct),
            target_xy=torch.where(blocked[:, None], target_world, goal_xy),
            blocked=blocked,
        )

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
        barrier_half_width = self._expanded(
            self.barrier_half_width, dimensions, env_ids
        )
        traversal_direction = self._expanded(
            self.traversal_direction, dimensions, env_ids
        )
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
        ascending_stair_count = torch.clamp(
            torch.floor(along / width) + 1.0,
            0.0,
            4.0,
        )
        descending_stair_count = torch.clamp(
            3.0 - torch.floor(along / width),
            0.0,
            4.0,
        )
        stair_count = torch.where(
            traversal_direction > 0.0,
            ascending_stair_count,
            descending_stair_count,
        )
        ground = torch.where(stairs, stair_count * amplitude * 0.45, ground)

        rough = terrain == 4
        rough_patch = (torch.abs(along) < 1.25) & (torch.abs(cross) < 1.2)
        rough_height = amplitude * torch.sin(6.0 * along) * torch.cos(5.0 * cross)
        ground = torch.where(rough & rough_patch, rough_height, ground)

        pit = terrain == 5
        in_pit = (torch.abs(along) < width) & (torch.abs(cross) < 1.2 * width)
        ground = torch.where(pit & in_pit, -amplitude, ground)

        wall = terrain == 6
        wall_cells = (torch.abs(along) < 0.10) & (
            torch.abs(cross) < barrier_half_width
        )
        obstacle_range = torch.where(wall & wall_cells, amplitude, obstacle_range)

        pillar = terrain == 7
        pillar_cells = along.square() + cross.square() < width.square()
        obstacle_range = torch.where(pillar & pillar_cells, amplitude, obstacle_range)

        mixed = terrain == 8
        mixed_ramp = 0.30 * amplitude * torch.clamp(
            along + 0.75, min=0.0, max=1.5
        )
        ground = torch.where(mixed, mixed_ramp, ground)
        mixed_step = mixed & (along > 0.0) & (cross < 0.0)
        mixed_wall = (
            mixed
            & (torch.abs(along - 0.8) < 0.12)
            & (cross > 0.0)
            & (cross < barrier_half_width)
        )
        ground = torch.where(mixed_step, mixed_ramp + amplitude, ground)
        obstacle_range = torch.where(mixed_wall, 0.8 + amplitude, obstacle_range)

        multi_route = terrain == 9
        low_barrier = (torch.abs(along) < 0.18) & (
            torch.abs(cross) < barrier_half_width
        )
        obstacle_range = torch.where(multi_route & low_barrier, amplitude, obstacle_range)

        low_obstacle = terrain == 10
        low_obstacle_cells = (torch.abs(along) < width) & (
            torch.abs(cross) < 0.6 * width
        )
        obstacle_range = torch.where(
            low_obstacle & low_obstacle_cells,
            amplitude,
            obstacle_range,
        )
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
