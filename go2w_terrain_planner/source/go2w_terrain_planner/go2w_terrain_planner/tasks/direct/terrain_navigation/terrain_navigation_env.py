"""Direct Isaac Lab task for learning high-level linear and yaw-rate commands."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.warp import raycast_mesh

from go2w_terrain_planner.mapping.coordinate_transform import encode_local_goal
from go2w_terrain_planner.mapping.grid_preprocessor import downsample_map_tensor
from go2w_terrain_planner.mapping.isaac_raycast_lidar import (
    IsaacRaycastLidarConfig,
    IsaacRaycastLidarMapper,
    transform_lidar_rays_to_world,
)
from go2w_terrain_planner.mapping.terrain_truth_model import (
    TerrainTruthModel,
    TerrainTruthConfig,
    TERRAIN_NAMES,
    fuse_aligned_map_history,
)
from go2w_terrain_planner.mapping.pointcloud_grid_projector import (
    PointCloudProjectionConfig,
)
from go2w_terrain_planner.mapping.temporal_grid_buffer import TemporalGridBuffer
from go2w_terrain_planner.robots.velocity_command_adapter import (
    ActionLimits,
    ExecutionModelConfig,
    VelocityCommandAdapter,
    VelocityExecutionModel,
)
from go2w_terrain_planner.utils.config_loader import load_project_config
from go2w_terrain_planner.utils.tensor_checks import require_finite

from .rewards import RewardWeights, navigation_reward
from .curriculum import CapabilityCurriculum, CurriculumStageSchedule
from .observations import (
    assemble_policy_observation,
    build_compact_map_observation,
)
from .terminations import (
    effective_terrain_entry_alignment,
    terrain_failure_state,
    termination_flags,
)
from .terrain_navigation_env_cfg import TerrainNavigationEnvCfg


class TerrainNavigationEnv(DirectRLEnv):
    cfg: TerrainNavigationEnvCfg

    def __init__(self, cfg: TerrainNavigationEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)
        project_config = load_project_config(cfg.project_config_directory or None)
        map_parameters = project_config["map"]
        history_parameters = project_config["history"]
        expected = (
            int(map_parameters["output_size"]),
            int(map_parameters["channels"]),
            int(map_parameters["actor_channels"]),
            int(history_parameters["command_length"]),
            int(history_parameters["motion_length"]),
        )
        actual = (
            cfg.map_size,
            cfg.map_channels,
            cfg.actor_map_channels,
            cfg.command_history_length,
            cfg.motion_history_length,
        )
        if actual != expected:
            raise ValueError(f"Isaac环境配置与项目YAML不一致：env={actual}, yaml={expected}")
        fixed_stage_sampling = cfg.curriculum_sampling_maximum_stage >= 0
        if fixed_stage_sampling and not (
            1
            <= cfg.curriculum_sampling_minimum_stage
            <= cfg.curriculum_sampling_maximum_stage
            <= cfg.curriculum_maximum_stage
        ):
            raise ValueError("测试课程阶段采样范围无效")
        self.pose = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.current_command = torch.zeros((self.num_envs, 2), device=self.device)
        self.previous_command = torch.zeros_like(self.current_command)
        self.current_distance = torch.zeros(self.num_envs, device=self.device)
        self.previous_navigation_potential = torch.zeros(
            self.num_envs, device=self.device
        )
        self.best_navigation_potential = torch.zeros(
            self.num_envs, device=self.device
        )
        self.current_navigation_potential = torch.zeros(
            self.num_envs, device=self.device
        )
        self.navigation_target_xy = torch.zeros(
            (self.num_envs, 2), device=self.device
        )
        self.navigation_path_blocked = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.path_increment = torch.zeros(self.num_envs, device=self.device)
        self.stuck_time = torch.zeros(self.num_envs, device=self.device)
        self.bad_observation_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.unstable = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.tilt_angle = torch.zeros(self.num_envs, device=self.device)
        self.reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stuck = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.observation_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.terrain_risk = torch.zeros(self.num_envs, device=self.device)
        self.observed_terrain_risk = torch.zeros(self.num_envs, device=self.device)
        self.true_height_range_m = torch.zeros(self.num_envs, device=self.device)
        self.current_ground_reference_z = torch.zeros(self.num_envs, device=self.device)
        self.unknown_ratio = torch.zeros(self.num_envs, device=self.device)
        self.action_limit_violation = torch.zeros(self.num_envs, device=self.device)
        self.map_is_healthy = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.safety_stop = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.safety_stop_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        sensor_parameters = project_config["sensor"]
        lidar_parameters = sensor_parameters["lidar"]
        projection_parameters = lidar_parameters["projection"]
        fusion_height_scale = (
            float(map_parameters["max_abs_relative_height_m"])
            if bool(map_parameters["normalize_heights"])
            else 1.0
        )
        self.sensor_fusion_maximum_ground_deviation = float(
            projection_parameters["maximum_ground_deviation_m"]
        ) / fusion_height_scale
        self.sensor_fusion_span_percentile = float(
            projection_parameters["fusion_span_percentile"]
        )
        terrain_parameters = project_config["terrain"]
        truth_cfg = TerrainTruthConfig(
            extent_m=cfg.map_extent_m,
            size=int(map_parameters["source_size"]),
            maximum_relative_height_m=float(map_parameters["max_abs_relative_height_m"]),
            maximum_height_range_m=float(map_parameters["max_height_range_m"]),
            ground_fill_value=float(map_parameters["ground_fill_value"]),
            range_fill_value=float(map_parameters["range_fill_value"]),
            normalize_heights=bool(map_parameters["normalize_heights"]),
            enabled_terrain_names=tuple(terrain_parameters["enabled_types"]),
            ramp_slope_range=tuple(terrain_parameters["ramp_slope_range"]),
            step_height_range_m=tuple(terrain_parameters["step_height_range_m"]),
            step_width_range_m=tuple(terrain_parameters["step_width_range_m"]),
            rough_amplitude_range_m=tuple(terrain_parameters["rough_amplitude_range_m"]),
            pit_depth_range_m=tuple(terrain_parameters["pit_depth_range_m"]),
            pit_half_width_range_m=tuple(
                terrain_parameters["pit_half_width_range_m"]
            ),
            pit_curriculum_start_depth_max_m=float(
                terrain_parameters["pit_curriculum_start_depth_max_m"]
            ),
            pit_curriculum_start_half_width_max_m=float(
                terrain_parameters["pit_curriculum_start_half_width_max_m"]
            ),
            pit_navigation_avoidance_depth_m=float(
                terrain_parameters["pit_navigation_avoidance_depth_m"]
            ),
            pit_goal_clearance_m=float(
                terrain_parameters["pit_goal_clearance_m"]
            ),
            obstacle_height_range_m=tuple(terrain_parameters["obstacle_height_range_m"]),
            low_obstacle_height_range_m=tuple(
                terrain_parameters["low_obstacle_height_range_m"]
            ),
            barrier_half_width_range_m=tuple(
                terrain_parameters["barrier_half_width_range_m"]
            ),
            challenge_barrier_width_scale=float(
                terrain_parameters["challenge_barrier_width_scale"]
            ),
            barrier_navigation_clearance_m=float(
                terrain_parameters["barrier_navigation_clearance_m"]
            ),
            friction_range=tuple(terrain_parameters["friction_range"]),
        )
        self.terrain_truth = TerrainTruthModel(self.num_envs, self.device, truth_cfg)
        projection_cfg = PointCloudProjectionConfig(
            input_crop_length_x_m=float(
                projection_parameters["input_crop_length_x_m"]
            ),
            input_crop_length_y_m=float(
                projection_parameters["input_crop_length_y_m"]
            ),
            body_height_m=float(projection_parameters["body_height_m"]),
            vertical_min_offset_m=float(
                projection_parameters["vertical_min_offset_m"]
            ),
            vertical_max_offset_m=float(
                projection_parameters["vertical_max_offset_m"]
            ),
            ground_percentile=float(
                projection_parameters["ground_percentile"]
            ),
            span_lower_percentile=float(
                projection_parameters["span_lower_percentile"]
            ),
            span_upper_percentile=float(
                projection_parameters["span_upper_percentile"]
            ),
            minimum_points_per_cell=int(
                projection_parameters["minimum_points_per_cell"]
            ),
            height_quantization_m=float(
                projection_parameters["height_quantization_m"]
            ),
            maximum_ground_deviation_m=float(
                projection_parameters["maximum_ground_deviation_m"]
            ),
            fusion_span_percentile=float(
                projection_parameters["fusion_span_percentile"]
            ),
        )
        self.lidar_mapper = IsaacRaycastLidarMapper(
            self.num_envs,
            self.device,
            IsaacRaycastLidarConfig(
                channels=int(lidar_parameters["channels"]),
                vertical_angles_deg=tuple(
                    float(value)
                    for value in lidar_parameters["vertical_angles_deg"]
                ),
                points_per_frame=int(lidar_parameters["points_per_frame"]),
                minimum_range_m=float(lidar_parameters["minimum_range_m"]),
                maximum_range_m=float(lidar_parameters["maximum_range_m"]),
                range_noise_std_m=float(sensor_parameters["range_noise_std_m"]),
                height_noise_std_m=float(sensor_parameters["height_noise_std_m"]),
                pose_xy_noise_std_m=float(sensor_parameters["pose_xy_noise_std_m"]),
                pose_yaw_noise_std_rad=float(
                    sensor_parameters["pose_yaw_noise_std_rad"]
                ),
                beam_dropout_probability=float(
                    sensor_parameters["beam_dropout_probability"]
                ),
                return_dropout_probability=float(
                    sensor_parameters["return_dropout_probability"]
                ),
                projection_config=projection_cfg,
            ),
        )
        if self.lidar_sensor.num_rays != self.lidar_mapper.cfg.points_per_frame:
            raise RuntimeError(
                "Isaac RayCaster射线数量与points_per_frame不一致："
                f"{self.lidar_sensor.num_rays} != "
                f"{self.lidar_mapper.cfg.points_per_frame}"
            )
        self.terrain_bank_metadata = {
            name: torch.as_tensor(value, device=self.device)
            for name, value in self.scene.terrain.bank_metadata.items()
        }
        self.terrain_bank_levels = int(
            self.scene.terrain.terrain_origins.shape[0]
        )
        self.terrain_variants_per_type = int(
            terrain_parameters["mesh_variants_per_type"]
        )
        print(
            "[INFO] Local map observation source: Isaac Lab RayCaster Warp mesh "
            f"backend with authoritative poses ({self.lidar_sensor.num_rays} independent rays)",
            flush=True,
        )
        self.temporal_buffer = TemporalGridBuffer(
            self.num_envs,
            cfg.motion_history_length + 1,
            cfg.map_channels,
            cfg.map_size,
            self.device,
            float(map_parameters["max_abs_relative_height_m"]),
        )
        self.sensor_fusion_buffer = TemporalGridBuffer(
            self.num_envs,
            int(map_parameters["observation_fusion_length"]),
            cfg.map_channels,
            cfg.map_size,
            self.device,
            float(map_parameters["max_abs_relative_height_m"]),
        )
        action_parameters = project_config["action"]
        self.command_adapter = VelocityCommandAdapter(
            ActionLimits(
                linear_min_mps=float(action_parameters["linear_min_mps"]),
                linear_max_mps=float(action_parameters["linear_max_mps"]),
                angular_min_radps=float(action_parameters["angular_min_radps"]),
                angular_max_radps=float(action_parameters["angular_max_radps"]),
            )
        )
        self.maximum_command_speed_mps = max(
            abs(self.command_adapter.limits.linear_min_mps),
            abs(self.command_adapter.limits.linear_max_mps),
        )
        self.command_scale = torch.tensor(
            (
                self.maximum_command_speed_mps,
                max(
                    abs(self.command_adapter.limits.angular_min_radps),
                    abs(self.command_adapter.limits.angular_max_radps),
                ),
            ),
            dtype=torch.float32,
            device=self.device,
        )
        execution_parameters = project_config["execution_model"]
        self.execution_model = VelocityExecutionModel(
            self.num_envs,
            self.device,
            ExecutionModelConfig(
                linear_time_constant_s=float(execution_parameters["linear_time_constant_s"]),
                angular_time_constant_s=float(execution_parameters["angular_time_constant_s"]),
                linear_acceleration_limit_mps2=float(action_parameters["linear_acceleration_limit_mps2"]),
                angular_acceleration_limit_radps2=float(action_parameters["angular_acceleration_limit_radps2"]),
                maximum_terrain_speed_loss=float(execution_parameters["maximum_terrain_speed_loss"]),
                tracking_noise_std=float(execution_parameters["tracking_noise_std"]),
                entry_angle_speed_penalty=float(execution_parameters["entry_angle_speed_penalty"]),
            ),
        )
        self.poor_entry_alignment_threshold = float(
            execution_parameters["poor_entry_alignment_threshold"]
        )
        self.stuck_height_range_min_m = float(execution_parameters["stuck_height_range_min_m"])
        self.stuck_friction_threshold = float(execution_parameters["stuck_friction_threshold"])
        self.terrain_tilt_gain_rad = float(execution_parameters["terrain_tilt_gain_rad"])
        self.entry_tilt_gain = float(execution_parameters["entry_tilt_gain"])
        self.reward_weights = RewardWeights(**project_config["reward"])
        self.reward_term_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in (
                "progress",
                "regression",
                "heading",
                "forward_to_goal",
                "goal_reached",
                "collision",
                "unstable",
                "stuck",
                "timeout",
                "out_of_bounds",
                "observation_failure",
                "linear_action_rate",
                "angular_action_rate",
                "angular_speed",
                "spin",
                "path_length",
                "action_limit_violation",
                "unknown_risk",
                "terrain_speed_risk",
                "time",
            )
        }
        curriculum_parameters = project_config["curriculum"]
        self.frontier_sampling_probability = float(
            curriculum_parameters["frontier_sampling_probability"]
        )
        self.challenge_sampling_probability = float(
            curriculum_parameters["challenge_sampling_probability"]
        )
        self.challenge_level_span = int(
            curriculum_parameters["challenge_level_span"]
        )
        self.curriculum_initial_level = int(
            curriculum_parameters["initial_level"]
        )
        self.curriculum = CapabilityCurriculum(
            self.num_envs,
            self.device,
            initial_level=self.curriculum_initial_level,
            minimum_level=int(curriculum_parameters["minimum_level"]),
            maximum_level=int(curriculum_parameters["maximum_level"]),
            success_rate_up=float(curriculum_parameters["success_rate_up"]),
            success_rate_down=float(curriculum_parameters["success_rate_down"]),
            minimum_episodes_per_level=int(
                curriculum_parameters["minimum_episodes_per_level"]
            ),
            minimum_full_difficulty_episodes=int(
                curriculum_parameters[
                    "minimum_full_difficulty_episodes"
                ]
            ),
            full_difficulty_threshold=float(
                curriculum_parameters["full_difficulty_threshold"]
            ),
            smoothing=float(curriculum_parameters["success_rate_smoothing"]),
            allow_level_demotion=bool(
                curriculum_parameters["allow_level_demotion"]
            ),
        )
        self.curriculum_schedule = CurriculumStageSchedule(
            curriculum_parameters["stages"],
            TERRAIN_NAMES,
            self.device,
        )
        if (
            self.curriculum_schedule.minimum_stage
            != int(curriculum_parameters["minimum_level"])
            or self.curriculum_schedule.maximum_stage
            != int(curriculum_parameters["maximum_level"])
        ):
            raise ValueError("课程边界与九阶段定义不一致")
        self.current_curriculum_stage = torch.full(
            (self.num_envs,),
            self.curriculum_initial_level,
            dtype=torch.long,
            device=self.device,
        )
        self.current_curriculum_difficulty = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.current_goal_sampling_maximum_m = torch.full(
            (self.num_envs,),
            float(
                self.curriculum_schedule.goal_maximum_m[
                    self.curriculum_initial_level
                ].item()
            ),
            device=self.device,
        )
        self.current_map = torch.zeros(
            (self.num_envs, cfg.map_channels, cfg.map_size, cfg.map_size), device=self.device
        )
        self.current_actor_map = torch.zeros(
            (
                self.num_envs,
                cfg.actor_map_channels,
                cfg.map_size,
                cfg.map_size,
            ),
            device=self.device,
        )
        self.terrain_episode_totals = torch.zeros(
            len(TERRAIN_NAMES), dtype=torch.long, device=self.device
        )
        self.terrain_success_totals = torch.zeros_like(
            self.terrain_episode_totals
        )
        self._all_env_ids = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def _setup_scene(self) -> None:
        self.proxy_robot = self.scene.extras["robot"]
        self.lidar_sensor = self.scene.sensors["lidar"]
        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def _assign_physical_terrain(
        self,
        env_ids: torch.Tensor,
        requested_difficulty: torch.Tensor,
    ) -> None:
        """Select a static mesh tile and synchronize the analytic truth model."""
        env_ids = torch.as_tensor(
            env_ids, dtype=torch.long, device=self.device
        ).reshape(-1)
        requested_difficulty = torch.as_tensor(
            requested_difficulty, dtype=torch.float32, device=self.device
        )
        if requested_difficulty.shape != (env_ids.numel(),):
            raise ValueError("地形难度必须与env_ids等长")
        levels = torch.round(
            requested_difficulty.clamp(0.0, 1.0)
            * float(self.terrain_bank_levels - 1)
        ).long()
        variants = torch.randint(
            self.terrain_variants_per_type,
            (env_ids.numel(),),
            device=self.device,
        )
        columns = (
            self.terrain_truth.terrain_type[env_ids]
            * self.terrain_variants_per_type
            + variants
        )
        selected_origins = self.scene.terrain.terrain_origins[levels, columns]
        self.scene.terrain.env_origins[env_ids] = selected_origins
        if hasattr(self.scene.terrain, "terrain_levels"):
            self.scene.terrain.terrain_levels[env_ids] = levels
        if hasattr(self.scene.terrain, "terrain_types"):
            self.scene.terrain.terrain_types[env_ids] = columns
        parameters = {
            name: values[levels, columns]
            for name, values in self.terrain_bank_metadata.items()
        }
        self.terrain_truth.apply_physical_geometry(env_ids, parameters)

    def _generate_lidar_map(self, env_ids: torch.Tensor | None = None):
        """Ray-cast the physical terrain mesh and project hits into the local map."""
        if env_ids is None:
            env_ids = self._all_env_ids
        else:
            env_ids = torch.as_tensor(
                env_ids, dtype=torch.long, device=self.device
            ).reshape(-1)
        ground_reference_z = self.terrain_truth.ground_reference(
            self.pose[env_ids], env_ids
        )
        origins = self.scene.env_origins[env_ids]
        robot_world_position = torch.zeros(
            (env_ids.numel(), 3), dtype=torch.float32, device=self.device
        )
        robot_world_position[:, :2] = self.pose[env_ids, :2] + origins[:, :2]
        robot_world_position[:, 2] = (
            origins[:, 2] + ground_reference_z + 0.20
        )
        ray_starts_world, ray_directions_world = transform_lidar_rays_to_world(
            self.lidar_sensor.ray_starts[env_ids],
            self.lidar_sensor.ray_directions[env_ids],
            robot_world_position,
            self.pose[env_ids, 2],
        )
        mesh_prim_path = self.lidar_sensor.cfg.mesh_prim_paths[0]
        ray_hits_world = raycast_mesh(
            ray_starts_world,
            ray_directions_world,
            max_dist=float(self.lidar_sensor.cfg.max_distance),
            mesh=self.lidar_sensor.meshes[mesh_prim_path],
        )[0]
        # All XT16 beams have the same local start.  Keep the explicit batch
        # origin expected by the point-cloud corruption/projection pipeline.
        sensor_origins_world = ray_starts_world[:, 0]
        raw_map = self.lidar_mapper.project(
            ray_hits_world,
            sensor_origins_world,
            robot_world_position,
            self.pose[env_ids, 2],
            origins[:, 2] + ground_reference_z,
            self.terrain_truth.domain_randomization_scale[env_ids],
            extent_m=self.terrain_truth.cfg.extent_m,
            size=self.terrain_truth.cfg.size,
            maximum_relative_height_m=(
                self.terrain_truth.cfg.maximum_relative_height_m
            ),
            maximum_height_range_m=(
                self.terrain_truth.cfg.maximum_height_range_m
            ),
            ground_fill_value=self.terrain_truth.cfg.ground_fill_value,
            range_fill_value=self.terrain_truth.cfg.range_fill_value,
            normalize_heights=self.terrain_truth.cfg.normalize_heights,
            env_ids=env_ids,
        )
        return downsample_map_tensor(raw_map, self.cfg.map_size), ground_reference_z

    """接收网络动作"""
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        finite_action = torch.isfinite(actions).all(dim=-1)
        finite_goal = torch.isfinite(self.goal_xy).all(dim=-1)
        safe_to_move = finite_action & finite_goal & self.map_is_healthy
        self.safety_stop.copy_(~safe_to_move)
        self.safety_stop_steps += self.safety_stop.to(dtype=torch.long)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=2.0, neginf=-2.0)
        self.action_limit_violation = (
            torch.relu(torch.abs(actions) - 1.0).square().sum(dim=-1)
            + (~finite_action).to(dtype=actions.dtype)
        )
        self.previous_command.copy_(self.current_command)
        physical_command = self.command_adapter.to_physical(actions)
        physical_command = torch.where(
            safe_to_move[:, None],
            physical_command,
            torch.zeros_like(physical_command),
        )
        self.current_command.copy_(physical_command)
        self.path_increment.zero_()

    """ 执行运动 """
    def _apply_action(self) -> None:
        terrain_metrics = self.terrain_truth.true_motion_metrics(
            self.pose, self.current_command[:, 0]
        )
        self.true_height_range_m.copy_(terrain_metrics.hazard_height_m)
        self.terrain_risk.copy_(
            torch.clamp(
                terrain_metrics.hazard_height_m / self.cfg.collision_height_range_m,
                0.0,
                1.0,
            )
        )
        effective_entry_alignment = effective_terrain_entry_alignment(
            terrain_metrics.entry_alignment,
            self.terrain_truth.terrain_type,
        )
        blocked = (
            (terrain_metrics.maximum_discontinuity_m >= self.stuck_height_range_min_m)
            & (terrain_metrics.maximum_discontinuity_m < self.cfg.collision_height_range_m)
            & (effective_entry_alignment < self.poor_entry_alignment_threshold)
            & (self.terrain_truth.friction < self.stuck_friction_threshold)
        )
        velocity = self.execution_model.step(
            self.current_command,
            self.terrain_risk,
            self.terrain_truth.friction,
            self.physics_dt,
            effective_entry_alignment,
            blocked,
            self.terrain_truth.domain_randomization_scale,
        )
        yaw = self.pose[:, 2]
        dx = velocity[:, 0] * torch.cos(yaw) * self.physics_dt
        dy = velocity[:, 0] * torch.sin(yaw) * self.physics_dt
        self.pose[:, 0] += dx
        self.pose[:, 1] += dy
        self.pose[:, 2] = torch.atan2(
            torch.sin(self.pose[:, 2] + velocity[:, 1] * self.physics_dt),
            torch.cos(self.pose[:, 2] + velocity[:, 1] * self.physics_dt),
        )
        self.path_increment += torch.sqrt(dx.square() + dy.square())

    def _write_proxy_pose(self, env_ids: torch.Tensor | None = None) -> None:
        write_all = env_ids is None
        if write_all:
            env_ids = self._all_env_ids
        else:
            env_ids = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            ).reshape(-1)
        position = torch.zeros((env_ids.numel(), 3), device=self.device)
        position[:, :2] = self.pose[env_ids, :2] + self.scene.env_origins[env_ids, :2]
        supporting_ground = self.terrain_truth.ground_reference(
            self.pose[env_ids], env_ids
        )
        position[:, 2] = (
            self.scene.env_origins[env_ids, 2]
            + supporting_ground
            + 0.20
        )
        yaw = self.pose[env_ids, 2]
        quaternion = torch.zeros((env_ids.numel(), 4), device=self.device)
        quaternion[:, 0] = torch.cos(0.5 * yaw)
        quaternion[:, 3] = torch.sin(0.5 * yaw)
        require_finite(position, "proxy_robot位置")
        require_finite(quaternion, "proxy_robot姿态")
        self.proxy_robot.set_world_poses(
            positions=position,
            orientations=quaternion,
            indices=None if write_all else env_ids,
        )

    """更新地图与状态"""
    def _update_outcomes(self) -> None:
        raw_map, ground_reference_z = self._generate_lidar_map()
        self.current_ground_reference_z.copy_(ground_reference_z)
        previous_aligned_map = self.temporal_buffer.aligned_maps_to(
            self.pose,
            self.current_ground_reference_z,
            self.cfg.map_extent_m,
            history_count=1,
        )[:, 0]
        self.sensor_fusion_buffer.push(
            raw_map,
            self.pose,
            self.current_command,
            self.current_ground_reference_z,
        )
        aligned_observations = self.sensor_fusion_buffer.aligned_maps(self.cfg.map_extent_m)
        self.current_map = fuse_aligned_map_history(
            aligned_observations,
            maximum_ground_deviation=(
                self.sensor_fusion_maximum_ground_deviation
            ),
            span_percentile=self.sensor_fusion_span_percentile,
        )
        self.current_actor_map = build_compact_map_observation(
            self.current_map,
            previous_aligned_map,
            aligned_observations,
        )
        self.observed_terrain_risk, self.unknown_ratio = self.terrain_truth.forward_risk(
            self.current_map, self.execution_model.actual_velocity[:, 0]
        )
        terrain_metrics = self.terrain_truth.true_motion_metrics(
            self.pose, self.execution_model.actual_velocity[:, 0]
        )
        effective_entry_alignment = effective_terrain_entry_alignment(
            terrain_metrics.entry_alignment,
            self.terrain_truth.terrain_type,
        )
        self.true_height_range_m.copy_(terrain_metrics.hazard_height_m)
        (
            self.terrain_risk,
            self.tilt_angle,
            self.collision,
            self.fallen,
            self.unstable,
        ) = terrain_failure_state(
            terrain_metrics.collision_height_m,
            terrain_metrics.hazard_height_m,
            effective_entry_alignment,
            self.current_command[:, 0],
            self.execution_model.actual_velocity[:, 0],
            collision_height_threshold_m=self.cfg.collision_height_range_m,
            maximum_command_speed_mps=self.maximum_command_speed_mps,
            unstable_risk_threshold=self.cfg.unstable_risk_threshold,
            terrain_tilt_gain_rad=self.terrain_tilt_gain_rad,
            entry_tilt_gain=self.entry_tilt_gain,
            maximum_tilt_rad=self.cfg.maximum_tilt_rad,
            fall_tilt_rad=self.cfg.fall_tilt_rad,
        )
        commanded_motion = torch.abs(self.current_command[:, 0]) > 0.15
        insufficient_motion = torch.abs(self.execution_model.actual_velocity[:, 0]) < 0.03
        self.stuck_time = torch.where(
            commanded_motion & insufficient_motion,
            self.stuck_time + self.step_dt,
            torch.zeros_like(self.stuck_time),
        )
        self.current_distance = torch.linalg.vector_norm(self.goal_xy - self.pose[:, :2], dim=-1)
        navigation_guidance = self.terrain_truth.navigation_guidance(
            self.pose,
            self.goal_xy,
        )
        self.current_navigation_potential.copy_(
            navigation_guidance.potential_m
        )
        self.navigation_target_xy.copy_(navigation_guidance.target_xy)
        self.navigation_path_blocked.copy_(navigation_guidance.blocked)
        finite_map = torch.isfinite(self.current_map).flatten(1).all(dim=1)
        observed_ratio = self.current_map[:, 2].mean(dim=(-2, -1))
        height_valid_ratio = self.current_map[:, 3].mean(dim=(-2, -1))
        healthy_map = (
            finite_map
            & (observed_ratio >= self.cfg.minimum_observed_ratio)
            & (height_valid_ratio >= self.cfg.minimum_height_valid_ratio)
        )
        self.map_is_healthy.copy_(healthy_map)
        self.bad_observation_steps = torch.where(
            healthy_map,
            torch.zeros_like(self.bad_observation_steps),
            self.bad_observation_steps + 1,
        )
        (
            _,
            self.reached,
            self.stuck,
            self.out_of_bounds,
            self.observation_failure,
        ) = termination_flags(
            self.current_distance,
            self.collision,
            self.unstable,
            self.stuck_time,
            self.pose[:, :2],
            self.bad_observation_steps,
            goal_tolerance_m=self.cfg.goal_tolerance_m,
            stuck_timeout_s=self.cfg.stuck_timeout_s,
            maximum_distance_m=self.cfg.maximum_distance_m,
            maximum_bad_observation_steps=self.cfg.maximum_bad_observation_steps,
        )
        # This is a non-physical visualization Xform.  USD writes require a
        # CUDA-to-CPU synchronization, so keep them completely out of headless
        # training while still updating the proxy once per policy step in GUI.
        if self.sim.has_gui():
            self._write_proxy_pose()
    """区分是超时还是结束"""
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_outcomes()
        terminated = (
            self.reached
            | self.collision
            | self.unstable
            | self.stuck
            | self.out_of_bounds
            | self.observation_failure
        )
        self.time_out = (
            self.episode_length_buf >= self.max_episode_length - 1
        ) & ~terminated
        return terminated, self.time_out

    """计算奖励"""
    def _get_rewards(self) -> torch.Tensor:
        # 终止条件仍使用真实目标距离；航向奖励则跟随无碰撞引导点，避免
        # 在墙、深坑或立柱前继续奖励朝障碍物直行。
        goal_delta = self.navigation_target_xy - self.pose[:, :2]

        goal_bearing_world = torch.atan2(
            goal_delta[:, 1],
            goal_delta[:, 0],
        )

        goal_bearing = torch.atan2(
            torch.sin(goal_bearing_world - self.pose[:, 2]),
            torch.cos(goal_bearing_world - self.pose[:, 2]),
        )

        reward, reward_terms = navigation_reward(
            self.previous_navigation_potential,
            self.current_navigation_potential,
            self.current_command,
            self.previous_command,
            self.path_increment,
            self.action_limit_violation,
            self.unknown_ratio,
            self.reached,
            self.collision,
            self.unstable,
            self.stuck,
            goal_bearing,
            self.execution_model.actual_velocity,
            self.reward_weights,
            best_distance=self.best_navigation_potential,
            terrain_risk=self.terrain_risk,
            time_out=self.time_out,
            out_of_bounds=self.out_of_bounds,
            observation_failure=self.observation_failure,
            return_terms=True,
        )
        for name, term in reward_terms.items():
            self.reward_term_sums[name] += term

        self.best_navigation_potential.copy_(
            torch.minimum(
                self.best_navigation_potential,
                self.current_navigation_potential,
            )
        )
        self.previous_navigation_potential.copy_(
            self.current_navigation_potential
        )
        return reward

    """构建网络输入"""
    def _get_observations(self) -> dict[str, torch.Tensor]:
        self.temporal_buffer.push(
            self.current_map,
            self.pose,
            self.current_command,
            self.current_ground_reference_z,
        )
        goal = encode_local_goal(self.pose, self.goal_xy, self.cfg.local_goal_maximum_m)
        command_history = self.temporal_buffer.commands
        motion_history = self.temporal_buffer.motion_history()
        policy = assemble_policy_observation(
            self.current_actor_map,
            goal,
            self.execution_model.actual_velocity,
            command_history,
            motion_history,
            velocity_scale=self.command_scale,
            map_extent_m=self.cfg.map_extent_m,
            expected_dimension=self.cfg.observation_space,
        )
        tracking_error = self.current_command - self.execution_model.actual_velocity
        terrain_type = self.terrain_truth.terrain_type
        half_extent = 0.5 * self.cfg.map_extent_m
        feature_delta_x = self.terrain_truth.feature_x - self.pose[:, 0]
        feature_delta_y = self.terrain_truth.feature_y - self.pose[:, 1]
        pose_cosine = torch.cos(self.pose[:, 2])
        pose_sine = torch.sin(self.pose[:, 2])
        feature_local = torch.stack(
            (
                pose_cosine * feature_delta_x + pose_sine * feature_delta_y,
                -pose_sine * feature_delta_x + pose_cosine * feature_delta_y,
            ),
            dim=-1,
        ) / half_extent
        relative_feature_yaw = self.terrain_truth.feature_yaw - self.pose[:, 2]
        feature_heading = torch.stack(
            (
                torch.sin(relative_feature_yaw),
                torch.cos(relative_feature_yaw),
            ),
            dim=-1,
        )
        normalized_motion_history = motion_history.clone()
        normalized_motion_history[..., :2] /= half_extent
        normalized_motion_history[..., 2] /= torch.pi
        critic = torch.cat(
            (
                goal,
                (
                    self.current_navigation_potential
                    / self.cfg.local_goal_maximum_m
                )[:, None],
                self.execution_model.actual_velocity / self.command_scale,
                self.current_command / self.command_scale,
                tracking_error / self.command_scale,
                terrain_type[:, None].float()
                / max(1, len(TERRAIN_NAMES) - 1),
                self.terrain_truth.terrain_difficulty[:, None],
                (
                    self.terrain_truth.amplitude
                    / self.terrain_truth.cfg.maximum_height_range_m
                )[:, None],
                feature_local,
                feature_heading,
                (self.terrain_truth.feature_width / half_extent)[:, None],
                (self.terrain_truth.barrier_half_width / half_extent)[:, None],
                self.terrain_truth.friction[:, None],
                self.terrain_risk[:, None],
                self.observed_terrain_risk[:, None],
                self.unknown_ratio[:, None],
                torch.clamp(
                    self.stuck_time / self.cfg.stuck_timeout_s,
                    0.0,
                    1.0,
                )[:, None],
                (command_history / self.command_scale).flatten(1),
                normalized_motion_history.flatten(1),
            ),
            dim=-1,
        )
        if critic.shape[-1] != self.cfg.state_space:
            raise RuntimeError(
                f"观测维度错误：policy={policy.shape[-1]}, critic={critic.shape[-1]}"
            )
        require_finite(policy, "policy环境观测")
        require_finite(critic, "critic环境观测")
        return {"policy": policy, "critic": critic}

    """ 一个回合结束做什么 """
    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.extras.pop("log", None)
        completed = self.episode_length_buf[env_ids] > 0
        episode_log = None
        if completed.any():
            completed_ids = env_ids[completed]
            completed_terrain = self.terrain_truth.terrain_type[completed_ids]
            mastery_mask = (
                self.current_curriculum_stage[completed_ids]
                == self.curriculum.levels[completed_ids]
            )
            mastery_ids = completed_ids[mastery_mask]
            self.curriculum.update(
                mastery_ids,
                self.reached[mastery_ids],
                self.current_curriculum_difficulty[mastery_ids],
            )
            episode_log = {
                "Episode/success_rate": self.reached[completed_ids].float().mean(),
                "Episode/collision_rate": self.collision[completed_ids].float().mean(),
                "Episode/unstable_rate": self.unstable[completed_ids].float().mean(),
                "Episode/stuck_rate": self.stuck[completed_ids].float().mean(),
                "Episode/timeout_rate": self.time_out[completed_ids].float().mean(),
                "Episode/out_of_bounds_rate": (
                    self.out_of_bounds[completed_ids].float().mean()
                ),
                "Episode/observation_failure_rate": (
                    self.observation_failure[completed_ids].float().mean()
                ),
                "Curriculum/mean_level": self.curriculum.levels.float().mean(),
                "Curriculum/mean_sampled_stage": (
                    self.current_curriculum_stage[completed_ids].float().mean()
                ),
                "Curriculum/frontier_success_ema": (
                    self.curriculum.success_rate.mean()
                ),
                "Curriculum/full_difficulty_success_ema": (
                    self.curriculum.full_difficulty_success_rate.mean()
                ),
                "Curriculum/mean_full_difficulty_episode_count": (
                    self.curriculum.full_difficulty_episode_count.float().mean()
                ),
                "Curriculum/mean_frontier_difficulty": (
                    self.curriculum.frontier_difficulty().mean()
                ),
                "Curriculum/mean_sampled_difficulty": (
                    self.current_curriculum_difficulty[completed_ids].mean()
                ),
                "Curriculum/mean_goal_maximum_m": (
                    self.current_goal_sampling_maximum_m.mean()
                ),
                "Curriculum/mean_domain_randomization_scale": (
                    self.terrain_truth.domain_randomization_scale.mean()
                ),
                "Episode/completed_count": completed.float().sum(),
                "Episode/final_navigation_potential": (
                    self.current_navigation_potential[completed_ids].mean()
                ),
                "Navigation/blocked_path_rate": (
                    self.navigation_path_blocked[completed_ids].float().mean()
                ),
                "Sensor/observed_ratio": (
                    self.current_map[completed_ids, 2].mean()
                ),
                "Sensor/height_valid_ratio": (
                    self.current_map[completed_ids, 3].mean()
                ),
                "Safety/stop_step_count": (
                    self.safety_stop_steps[completed_ids].float().mean()
                ),
            }
            episode_log["Sensor/point_slots_per_frame"] = float(
                self.lidar_mapper.cfg.points_per_frame
            )
            episode_log["Sensor/mean_point_count"] = (
                self.lidar_mapper.last_hit_mask[completed_ids]
                .sum(dim=1)
                .float()
                .mean()
            )
            episode_log.update(
                {
                    f"Reward/{name}": values[completed_ids].mean()
                    for name, values in self.reward_term_sums.items()
                }
            )
            terrain_episode_counts = torch.bincount(
                completed_terrain, minlength=len(TERRAIN_NAMES)
            )
            terrain_success_counts = torch.bincount(
                completed_terrain[self.reached[completed_ids]],
                minlength=len(TERRAIN_NAMES),
            )
            self.terrain_episode_totals += terrain_episode_counts
            self.terrain_success_totals += terrain_success_counts
            for terrain_index, terrain_name in enumerate(TERRAIN_NAMES):
                terrain_mask = completed_terrain == terrain_index
                episode_log[f"Terrain/{terrain_name}_episode_count"] = (
                    terrain_mask.float().sum()
                )
                episode_log[f"Terrain/{terrain_name}_success_count"] = (
                    self.reached[completed_ids][terrain_mask].float().sum()
                )
                episode_log[f"Terrain/{terrain_name}_success_rate"] = (
                    self.terrain_success_totals[terrain_index].float()
                    / self.terrain_episode_totals[terrain_index].clamp(min=1).float()
                )
            completed_stages = self.current_curriculum_stage[completed_ids]
            for stage in range(
                self.curriculum_schedule.minimum_stage,
                self.curriculum_schedule.maximum_stage + 1,
            ):
                stage_mask = completed_stages == stage
                stage_name = self.curriculum_schedule.stage_names[stage]
                episode_log[f"Stage/{stage}_{stage_name}_episode_count"] = (
                    stage_mask.float().sum()
                )
                episode_log[f"Stage/{stage}_{stage_name}_success_count"] = (
                    self.reached[completed_ids][stage_mask].float().sum()
                )
            pit_mask = completed_terrain == TERRAIN_NAMES.index("pit")
            episode_log["Terrain/pit_mean_difficulty"] = (
                (
                    self.terrain_truth.terrain_difficulty[completed_ids][pit_mask]
                ).sum()
                / pit_mask.sum().clamp(min=1)
            )
        super()._reset_idx(env_ids)
        if episode_log is not None:
            self.extras["log"] = episode_log
        count = env_ids.numel()
        self.pose[env_ids] = 0.0
        self.current_command[env_ids] = 0.0
        self.previous_command[env_ids] = 0.0
        self.stuck_time[env_ids] = 0.0
        self.bad_observation_steps[env_ids] = 0
        self.safety_stop_steps[env_ids] = 0
        self.execution_model.reset(env_ids)
        fixed_stage_sampling = self.cfg.curriculum_sampling_maximum_stage >= 0
        task_batch = self.curriculum_schedule.sample(
            self.curriculum.levels[env_ids],
            self.curriculum.frontier_difficulty(env_ids),
            frontier_probability=self.frontier_sampling_probability,
            challenge_probability=self.challenge_sampling_probability,
            challenge_stage_span=self.challenge_level_span,
            fixed_minimum_stage=(
                self.cfg.curriculum_sampling_minimum_stage
                if fixed_stage_sampling
                else None
            ),
            fixed_maximum_stage=(
                self.cfg.curriculum_sampling_maximum_stage
                if fixed_stage_sampling
                else None
            ),
        )
        self.current_curriculum_stage[env_ids] = task_batch.stages
        self.current_curriculum_difficulty[env_ids] = (
            task_batch.curriculum_difficulty
        )
        self.terrain_truth.reset(
            env_ids=env_ids,
            terrain_probabilities=task_batch.terrain_probabilities,
            terrain_difficulty=task_batch.geometry_difficulty,
            domain_randomization_scale=(
                task_batch.domain_randomization_scale
            ),
        )
        # RayCaster只能缓存一个静态网格，因此所有地形预先合并成
        # 一张物理地形库；每个回合通过切换env origin选择真实网格块。
        self._assign_physical_terrain(env_ids, task_batch.geometry_difficulty)

        # 各阶段独立控制初始朝向难度：第一阶段近似正前方，
        # 第二阶段起可覆盖任意目标方向，楼梯阶段保持入口对齐。
        heading_error = task_batch.heading_half_range_rad * (
            2.0 * torch.rand(count, device=self.device) - 1.0
        )
        self.pose[env_ids, 2] = torch.atan2(
            torch.sin(
                self.terrain_truth.route_yaw[env_ids]
                + heading_error
            ),
            torch.cos(
                self.terrain_truth.route_yaw[env_ids]
                + heading_error
            ),
        )

        goal_sampling_maximum = task_batch.goal_maximum_m
        self.current_goal_sampling_maximum_m[env_ids] = goal_sampling_maximum
        self.goal_xy[env_ids] = self.terrain_truth.sample_task_goals(
            self.pose[env_ids],
            task_batch.goal_minimum_m,
            goal_sampling_maximum,
            env_ids,
        )
        # 可视化Xform与权威规划位姿同步；真实射线起点直接由该权威位姿
        # 计算，不再通过第二个PhysX刚体视图反读相同状态。无界面训练
        # 不写USD，避免每次reset发生GPU到CPU同步。
        if self.sim.has_gui():
            self._write_proxy_pose(env_ids)
        raw_map, ground_reference_z = self._generate_lidar_map(env_ids)
        self.current_map[env_ids] = raw_map
        reset_finite_map = torch.isfinite(raw_map).flatten(1).all(dim=1)
        reset_observed_ratio = raw_map[:, 2].mean(dim=(-2, -1))
        reset_height_valid_ratio = raw_map[:, 3].mean(dim=(-2, -1))
        self.map_is_healthy[env_ids] = (
            reset_finite_map
            & (reset_observed_ratio >= self.cfg.minimum_observed_ratio)
            & (reset_height_valid_ratio >= self.cfg.minimum_height_valid_ratio)
        )
        self.safety_stop[env_ids] = ~self.map_is_healthy[env_ids]
        self.current_ground_reference_z[env_ids] = ground_reference_z
        reset_observed_risk, reset_unknown = self.terrain_truth.forward_risk(
            self.current_map[env_ids], self.current_command[env_ids, 0]
        )
        reset_metrics = self.terrain_truth.true_motion_metrics(
            self.pose[env_ids], self.current_command[env_ids, 0], env_ids
        )
        self.true_height_range_m[env_ids] = reset_metrics.hazard_height_m
        self.terrain_risk[env_ids] = torch.clamp(
            reset_metrics.hazard_height_m / self.cfg.collision_height_range_m, 0.0, 1.0
        )
        self.observed_terrain_risk[env_ids] = reset_observed_risk
        self.unknown_ratio[env_ids] = reset_unknown
        self.current_distance[env_ids] = torch.linalg.vector_norm(
            self.goal_xy[env_ids] - self.pose[env_ids, :2], dim=-1
        )
        reset_guidance = self.terrain_truth.navigation_guidance(
            self.pose[env_ids],
            self.goal_xy[env_ids],
            env_ids,
        )
        self.current_navigation_potential[env_ids] = reset_guidance.potential_m
        self.navigation_target_xy[env_ids] = reset_guidance.target_xy
        self.navigation_path_blocked[env_ids] = reset_guidance.blocked
        self.previous_navigation_potential[env_ids] = (
            self.current_navigation_potential[env_ids]
        )
        self.best_navigation_potential[env_ids] = (
            self.current_navigation_potential[env_ids]
        )
        self.path_increment[env_ids] = 0.0
        self.action_limit_violation[env_ids] = 0.0
        self.collision[env_ids] = False
        self.unstable[env_ids] = False
        self.fallen[env_ids] = False
        self.tilt_angle[env_ids] = 0.0
        self.reached[env_ids] = False
        self.stuck[env_ids] = False
        self.out_of_bounds[env_ids] = False
        self.observation_failure[env_ids] = False
        self.time_out[env_ids] = False
        for values in self.reward_term_sums.values():
            values[env_ids] = 0.0
        self.sensor_fusion_buffer.reset(
            env_ids,
            self.current_map[env_ids],
            self.pose[env_ids],
            self.current_ground_reference_z[env_ids],
        )
        self.temporal_buffer.reset(
            env_ids,
            self.current_map[env_ids],
            self.pose[env_ids],
            self.current_ground_reference_z[env_ids],
        )
        repeated_observation = self.current_map[env_ids, None].expand(
            -1,
            self.sensor_fusion_buffer.history_length,
            -1,
            -1,
            -1,
        )
        self.current_actor_map[env_ids] = build_compact_map_observation(
            self.current_map[env_ids],
            self.current_map[env_ids],
            repeated_observation,
        )
