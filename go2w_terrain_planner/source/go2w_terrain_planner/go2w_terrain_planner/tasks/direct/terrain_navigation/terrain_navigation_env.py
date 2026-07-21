"""Direct Isaac Lab task for learning high-level linear and yaw-rate commands."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from go2w_terrain_planner.mapping.coordinate_transform import encode_local_goal
from go2w_terrain_planner.mapping.grid_preprocessor import downsample_map_tensor
from go2w_terrain_planner.mapping.simulated_local_map import (
    SimulatedLocalMap,
    SimulatedMapConfig,
    fuse_aligned_map_history,
)
from go2w_terrain_planner.mapping.temporal_grid_buffer import TemporalGridBuffer
from go2w_terrain_planner.robots.velocity_command_adapter import (
    ActionLimits,
    ExecutionModelConfig,
    VelocityCommandAdapter,
    VelocityExecutionModel,
)
from go2w_terrain_planner.utils.config_loader import load_project_config

from .rewards import RewardWeights, navigation_reward
from .curriculum import TerrainCurriculum
from .observations import assemble_policy_observation
from .terminations import terrain_failure_state, termination_flags
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
            int(map_parameters["history_length"]),
            int(history_parameters["command_length"]),
            int(history_parameters["motion_length"]),
        )
        actual = (
            cfg.map_size,
            cfg.map_channels,
            cfg.map_history_length,
            cfg.command_history_length,
            cfg.motion_history_length,
        )
        if actual != expected:
            raise ValueError(f"Isaac环境配置与项目YAML不一致：env={actual}, yaml={expected}")
        self.pose = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.current_command = torch.zeros((self.num_envs, 2), device=self.device)
        self.previous_command = torch.zeros_like(self.current_command)
        self.previous_distance = torch.zeros(self.num_envs, device=self.device)
        self.current_distance = torch.zeros(self.num_envs, device=self.device)
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
        self.terrain_risk = torch.zeros(self.num_envs, device=self.device)
        self.observed_terrain_risk = torch.zeros(self.num_envs, device=self.device)
        self.true_height_range_m = torch.zeros(self.num_envs, device=self.device)
        self.current_ground_reference_z = torch.zeros(self.num_envs, device=self.device)
        self.unknown_ratio = torch.zeros(self.num_envs, device=self.device)
        self.action_limit_violation = torch.zeros(self.num_envs, device=self.device)

        sensor_parameters = project_config["sensor"]
        terrain_parameters = project_config["terrain"]
        map_cfg = SimulatedMapConfig(
            extent_m=cfg.map_extent_m,
            size=int(map_parameters["source_size"]),
            maximum_relative_height_m=float(map_parameters["max_abs_relative_height_m"]),
            maximum_height_range_m=float(map_parameters["max_height_range_m"]),
            ground_fill_value=float(map_parameters["ground_fill_value"]),
            range_fill_value=float(map_parameters["range_fill_value"]),
            normalize_heights=bool(map_parameters["normalize_heights"]),
            height_noise_std_m=float(sensor_parameters["height_noise_std_m"]),
            range_noise_std_m=float(sensor_parameters["range_noise_std_m"]),
            missing_probability=float(sensor_parameters["random_missing_probability"]),
            ray_only_probability=float(sensor_parameters["ray_only_probability"]),
            occlusion_sector_probability=float(sensor_parameters["occlusion_sector_probability"]),
            occlusion_width_range_rad=tuple(sensor_parameters["occlusion_width_range_rad"]),
            pose_xy_noise_std_m=float(sensor_parameters["pose_xy_noise_std_m"]),
            pose_yaw_noise_std_rad=float(sensor_parameters["pose_yaw_noise_std_rad"]),
            time_jitter_std_s=float(sensor_parameters["time_jitter_std_s"]),
            enabled_terrain_names=tuple(terrain_parameters["enabled_types"]),
            ramp_slope_range=tuple(terrain_parameters["ramp_slope_range"]),
            step_height_range_m=tuple(terrain_parameters["step_height_range_m"]),
            step_width_range_m=tuple(terrain_parameters["step_width_range_m"]),
            rough_amplitude_range_m=tuple(terrain_parameters["rough_amplitude_range_m"]),
            pit_depth_range_m=tuple(terrain_parameters["pit_depth_range_m"]),
            obstacle_height_range_m=tuple(terrain_parameters["obstacle_height_range_m"]),
            friction_range=tuple(terrain_parameters["friction_range"]),
        )
        self.map_generator = SimulatedLocalMap(self.num_envs, self.device, map_cfg)
        self.temporal_buffer = TemporalGridBuffer(
            self.num_envs,
            cfg.map_history_length,
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
        curriculum_parameters = project_config["curriculum"]
        self.curriculum = TerrainCurriculum(
            self.num_envs,
            self.device,
            initial_level=int(curriculum_parameters["initial_level"]),
            maximum_level=int(curriculum_parameters["maximum_level"]),
            success_rate_up=float(curriculum_parameters["success_rate_up"]),
            success_rate_down=float(curriculum_parameters["success_rate_down"]),
            minimum_episodes_per_level=int(
                curriculum_parameters["minimum_episodes_per_level"]
            ),
        )
        self.current_map = torch.zeros(
            (self.num_envs, cfg.map_channels, cfg.map_size, cfg.map_size), device=self.device
        )
        self._reset_idx(torch.arange(self.num_envs, device=self.device))

    def _setup_scene(self) -> None:
        self.proxy_robot = RigidObject(self.cfg.proxy_robot)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.rigid_objects["robot"] = self.proxy_robot
        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = torch.nan_to_num(actions, nan=0.0, posinf=2.0, neginf=-2.0)
        self.action_limit_violation = torch.relu(torch.abs(actions) - 1.0).square().sum(dim=-1)
        self.previous_command.copy_(self.current_command)
        self.current_command.copy_(self.command_adapter.to_physical(actions))
        self.path_increment.zero_()

    def _apply_action(self) -> None:
        terrain_metrics = self.map_generator.true_motion_metrics(
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
        blocked = (
            (terrain_metrics.maximum_discontinuity_m >= self.stuck_height_range_min_m)
            & (terrain_metrics.maximum_discontinuity_m < self.cfg.collision_height_range_m)
            & (terrain_metrics.entry_alignment < self.poor_entry_alignment_threshold)
            & (self.map_generator.friction < self.stuck_friction_threshold)
        )
        velocity = self.execution_model.step(
            self.current_command,
            self.terrain_risk,
            self.map_generator.friction,
            self.physics_dt,
            terrain_metrics.entry_alignment,
            blocked,
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
        self._write_proxy_pose()

    def _write_proxy_pose(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        position = torch.zeros((env_ids.numel(), 3), device=self.device)
        position[:, :2] = self.pose[env_ids, :2] + self.scene.env_origins[env_ids, :2]
        position[:, 2] = 0.20
        yaw = self.pose[env_ids, 2]
        quaternion = torch.zeros((env_ids.numel(), 4), device=self.device)
        quaternion[:, 0] = torch.cos(0.5 * yaw)
        quaternion[:, 3] = torch.sin(0.5 * yaw)
        self.proxy_robot.write_root_pose_to_sim(torch.cat((position, quaternion), dim=-1), env_ids=env_ids)

    def _update_outcomes(self) -> None:
        raw_map, ground_reference_z = self.map_generator.generate(
            self.pose,
            self.execution_model.actual_velocity,
            return_ground_reference=True,
        )
        raw_map = downsample_map_tensor(raw_map, self.cfg.map_size)
        self.current_ground_reference_z.copy_(ground_reference_z)
        self.sensor_fusion_buffer.push(
            raw_map,
            self.pose,
            self.current_command,
            self.current_ground_reference_z,
        )
        aligned_observations = self.sensor_fusion_buffer.aligned_maps(self.cfg.map_extent_m)
        self.current_map = fuse_aligned_map_history(aligned_observations)
        self.observed_terrain_risk, self.unknown_ratio = self.map_generator.forward_risk(
            self.current_map, self.execution_model.actual_velocity[:, 0]
        )
        terrain_metrics = self.map_generator.true_motion_metrics(
            self.pose, self.execution_model.actual_velocity[:, 0]
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
            terrain_metrics.entry_alignment,
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
        finite_map = torch.isfinite(self.current_map).flatten(1).all(dim=1)
        observed_ratio = self.current_map[:, 2].mean(dim=(-2, -1))
        height_valid_ratio = self.current_map[:, 3].mean(dim=(-2, -1))
        healthy_map = (
            finite_map
            & (observed_ratio >= self.cfg.minimum_observed_ratio)
            & (height_valid_ratio >= self.cfg.minimum_height_valid_ratio)
        )
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
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        reward = navigation_reward(
            self.previous_distance,
            self.current_distance,
            self.current_command,
            self.previous_command,
            self.path_increment,
            self.action_limit_violation,
            self.unknown_ratio,
            self.reached,
            self.collision,
            self.unstable,
            self.stuck,
            self.reward_weights,
        )
        self.previous_distance.copy_(self.current_distance)
        return reward

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self.temporal_buffer.push(
            self.current_map,
            self.pose,
            self.current_command,
            self.current_ground_reference_z,
        )
        aligned_maps = self.temporal_buffer.aligned_maps(self.cfg.map_extent_m)
        goal = encode_local_goal(self.pose, self.goal_xy, self.cfg.local_goal_maximum_m)
        policy = assemble_policy_observation(
            aligned_maps,
            goal,
            self.execution_model.actual_velocity,
            self.temporal_buffer.commands,
            self.temporal_buffer.motion_history(),
            expected_dimension=self.cfg.observation_space,
        )
        goal_delta = self.goal_xy - self.pose[:, :2]
        tracking_error = self.current_command - self.execution_model.actual_velocity
        critic = torch.cat(
            (
                goal_delta,
                self.current_distance[:, None],
                self.execution_model.actual_velocity,
                self.map_generator.terrain_type[:, None].float()
                / max(1, self.cfg.curriculum_maximum_terrain_index),
                self.map_generator.amplitude[:, None],
                self.map_generator.friction[:, None],
                self.terrain_risk[:, None],
                self.unknown_ratio[:, None],
                self.stuck_time[:, None],
                tracking_error,
            ),
            dim=-1,
        )
        if critic.shape[-1] != self.cfg.state_space:
            raise RuntimeError(
                f"观测维度错误：policy={policy.shape[-1]}, critic={critic.shape[-1]}"
            )
        if not torch.isfinite(policy).all() or not torch.isfinite(critic).all():
            raise RuntimeError("环境观测包含NaN或Inf")
        return {"policy": policy, "critic": critic}

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.extras.pop("log", None)
        completed = self.episode_length_buf[env_ids] > 0
        episode_log = None
        if completed.any():
            completed_ids = env_ids[completed]
            episode_log = {
                "Episode/success_rate": self.reached[completed_ids].float().mean(),
                "Episode/collision_rate": self.collision[completed_ids].float().mean(),
                "Episode/unstable_rate": self.unstable[completed_ids].float().mean(),
                "Episode/stuck_rate": self.stuck[completed_ids].float().mean(),
                "Curriculum/mean_level": self.curriculum.levels.float().mean(),
            }
            mastery_mask = (
                self.map_generator.terrain_type[completed_ids]
                == self.curriculum.levels[completed_ids]
            )
            mastery_ids = completed_ids[mastery_mask]
            self.curriculum.update(mastery_ids, self.reached[mastery_ids])
        super()._reset_idx(env_ids)
        if episode_log is not None:
            self.extras["log"] = episode_log
        count = env_ids.numel()
        self.pose[env_ids] = 0.0
        self.current_command[env_ids] = 0.0
        self.previous_command[env_ids] = 0.0
        self.stuck_time[env_ids] = 0.0
        self.bad_observation_steps[env_ids] = 0
        self.execution_model.reset(env_ids)
        self.map_generator.reset(env_ids, self.curriculum.levels[env_ids])
        self.goal_xy[env_ids] = self.map_generator.sample_task_goals(
            self.pose[env_ids],
            self.cfg.local_goal_minimum_m,
            self.cfg.local_goal_maximum_m,
            env_ids,
        )
        heading_error = 0.7 * (torch.rand(count, device=self.device) - 0.5)
        self.pose[env_ids, 2] = self.map_generator.route_yaw[env_ids] + heading_error
        generated_map, ground_reference_z = self.map_generator.generate(
            self.pose[env_ids],
            self.execution_model.actual_velocity[env_ids],
            return_ground_reference=True,
            env_ids=env_ids,
        )
        raw_map = downsample_map_tensor(generated_map, self.cfg.map_size)
        self.current_map[env_ids] = raw_map
        self.current_ground_reference_z[env_ids] = ground_reference_z
        reset_observed_risk, reset_unknown = self.map_generator.forward_risk(
            self.current_map[env_ids], self.current_command[env_ids, 0]
        )
        reset_metrics = self.map_generator.true_motion_metrics(
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
        self.previous_distance[env_ids] = self.current_distance[env_ids]
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
        self._write_proxy_pose(env_ids)
