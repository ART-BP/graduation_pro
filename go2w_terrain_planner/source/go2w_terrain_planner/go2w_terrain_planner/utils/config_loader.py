"""Load and validate the editable project YAML files."""

from __future__ import annotations

import math
import os
from pathlib import Path

import yaml

from go2w_terrain_planner.mapping.simulated_local_map import TERRAIN_NAMES

from .observation_layout import ACTOR_MAP_CHANNELS, policy_observation_dimension


CONFIG_NAMES = ("observation", "action", "reward", "terrain", "sensor", "training")


def default_config_directory() -> Path:
    project_root = os.environ.get("GO2W_PROJECT_ROOT")
    if project_root:
        return Path(project_root) / "configs"
    return Path(__file__).resolve().parents[4] / "configs"


def load_project_config(config_directory: str | Path | None = None) -> dict:
    """ 读取全部参数，以字典返回 """
    directory = Path(config_directory) if config_directory is not None else default_config_directory()
    result: dict = {}
    for name in CONFIG_NAMES:
        path = directory / f"{name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"缺少配置文件：{path}")
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
        if not isinstance(value, dict):
            raise ValueError(f"配置文件必须包含YAML映射：{path}")
        overlap = result.keys() & value.keys()
        if overlap:
            raise ValueError(f"配置顶层键重复：{sorted(overlap)}")
        result.update(value)
    validate_project_config(result)
    return result


def validate_project_config(config: dict) -> None:
    """ 参数是否有效， 主要是参数大小范围等检查 """
    required_sections = {
        "map",
        "history",
        "goal",
        "action",
        "execution_model",
        "reward",
        "termination",
        "sensor",
        "terrain",
        "curriculum",
        "training",
        "runner",
        "ppo",
        "model",
    }
    missing = required_sections - config.keys()
    if missing:
        raise ValueError(f"缺少配置段：{sorted(missing)}")

    def require_positive(values: dict, keys: tuple[str, ...], section: str) -> None:
        """value[key]必须都是正数"""
        for key in keys:
            value = values[key]
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{section}.{key}必须为有限正数")

    def require_range(bounds, name: str, *, minimum: float | None = None) -> None:
        """检查bounds的上下限必须有效"""
        if (
            not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
            or not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in bounds
            )
            or bounds[0] > bounds[1]
            or (minimum is not None and bounds[0] < minimum)
        ):
            raise ValueError(f"{name}必须为有效递增范围")

    map_cfg = config["map"]
    history_cfg = config["history"]
    require_positive(
        map_cfg,
        (
            "extent_m",
            "source_resolution_m",
            "source_size",
            "output_size",
            "max_abs_relative_height_m",
            "max_height_range_m",
        ),
        "map",
    )
    if not math.isclose(
        map_cfg["source_size"] * map_cfg["source_resolution_m"],
        map_cfg["extent_m"],
        rel_tol=1.0e-6,
    ):
        raise ValueError("source_size、source_resolution_m与extent_m不一致")
    if map_cfg["channels"] != 4:
        raise ValueError("地图通道数必须为4")
    if map_cfg["actor_channels"] != ACTOR_MAP_CHANNELS:
        raise ValueError(
            f"Actor紧凑地图通道数必须为{ACTOR_MAP_CHANNELS}"
        )
    if not map_cfg["normalize_heights"]:
        raise ValueError("时序高度补偿要求normalize_heights=true")
    for key in ("minimum_observed_ratio", "minimum_height_valid_ratio"):
        if not 0.0 <= map_cfg[key] <= 1.0:
            raise ValueError(f"{key}必须位于[0,1]")
    if (
        map_cfg["observation_fusion_length"] < 2
        or map_cfg["observation_fusion_length"] > 20
    ):
        raise ValueError("单张地图快照的有效观测融合长度必须为2到20")
    if (
        history_cfg["command_length"] <= 0
        or history_cfg["motion_length"] != history_cfg["command_length"]
    ):
        raise ValueError("GRU要求指令历史与运动历史等长且大于0")
    action_cfg = config["action"]
    require_positive(
        action_cfg,
        ("linear_acceleration_limit_mps2", "angular_acceleration_limit_radps2"),
        "action",
    )
    if action_cfg["linear_min_mps"] >= action_cfg["linear_max_mps"]:
        raise ValueError("线速度范围无效")
    if action_cfg["angular_min_radps"] >= action_cfg["angular_max_radps"]:
        raise ValueError("角速度范围无效")
    if not action_cfg["linear_min_mps"] <= 0.0 <= action_cfg["linear_max_mps"]:
        raise ValueError("零中心动作映射要求线速度范围包含0")
    if not action_cfg["angular_min_radps"] <= 0.0 <= action_cfg["angular_max_radps"]:
        raise ValueError("零中心动作映射要求角速度范围包含0")
    execution_cfg = config["execution_model"]
    require_positive(
        execution_cfg,
        (
            "linear_time_constant_s",
            "angular_time_constant_s",
            "terrain_tilt_gain_rad",
            "poor_entry_alignment_threshold",
            "stuck_height_range_min_m",
            "stuck_friction_threshold",
        ),
        "execution_model",
    )
    for key in ("tracking_noise_std", "entry_angle_speed_penalty", "entry_tilt_gain"):
        if execution_cfg[key] < 0.0:
            raise ValueError(f"execution_model.{key}不能为负数")
    if not 0.0 <= execution_cfg["maximum_terrain_speed_loss"] < 1.0:
        raise ValueError("maximum_terrain_speed_loss必须位于[0,1)")

    goal_cfg = config["goal"]
    require_positive(goal_cfg, ("maximum_distance_m",), "goal")
    if (
        not isinstance(goal_cfg["minimum_distance_m"], (int, float))
        or not math.isfinite(goal_cfg["minimum_distance_m"])
        or goal_cfg["minimum_distance_m"] < 0.0
        or goal_cfg["minimum_distance_m"] >= goal_cfg["maximum_distance_m"]
    ):
        raise ValueError("局部目标距离范围无效")

    termination_cfg = config["termination"]
    require_positive(
        termination_cfg,
        (
            "goal_tolerance_m",
            "collision_height_range_m",
            "unstable_risk_threshold",
            "maximum_tilt_rad",
            "fall_tilt_rad",
            "stuck_timeout_s",
            "maximum_episode_s",
            "maximum_distance_from_origin_m",
            "maximum_bad_observation_steps",
        ),
        "termination",
    )
    if termination_cfg["fall_tilt_rad"] <= termination_cfg["maximum_tilt_rad"]:
        raise ValueError("跌倒角阈值必须大于失稳角阈值")
    if (
        termination_cfg["maximum_distance_from_origin_m"]
        <= goal_cfg["maximum_distance_m"]
    ):
        raise ValueError("最大活动半径必须大于局部目标最大距离")

    sensor_cfg = config["sensor"]
    if sensor_cfg["observation_source"] not in {"analytic", "raycast"}:
        raise ValueError("sensor.observation_source必须为analytic或raycast")
    for key in (
        "height_noise_std_m",
        "range_noise_std_m",
        "pose_xy_noise_std_m",
        "pose_yaw_noise_std_rad",
        "time_jitter_std_s",
    ):
        if sensor_cfg[key] < 0.0:
            raise ValueError(f"sensor.{key}不能为负数")
    for key in ("random_missing_probability", "ray_only_probability"):
        if not 0.0 <= sensor_cfg[key] <= 1.0:
            raise ValueError(f"{key}必须位于[0,1]")
    if sensor_cfg["random_missing_probability"] + sensor_cfg["ray_only_probability"] > 1.0:
        raise ValueError("缺失率与仅射线观测率之和不能超过1")
    if not 0.0 <= sensor_cfg["occlusion_sector_probability"] <= 1.0:
        raise ValueError("occlusion_sector_probability必须位于[0,1]")
    require_range(
        sensor_cfg["occlusion_width_range_rad"],
        "sensor.occlusion_width_range_rad",
        minimum=0.0,
    )
    lidar_cfg = sensor_cfg["lidar"]
    if not isinstance(lidar_cfg, dict):
        raise ValueError("sensor.lidar必须为YAML映射")
    channels = lidar_cfg["channels"]
    vertical_angles = lidar_cfg["vertical_angles_deg"]
    if (
        not isinstance(channels, int)
        or channels <= 0
        or not isinstance(vertical_angles, (list, tuple))
        or len(vertical_angles) != channels
        or not all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            and -90.0 < value < 90.0
            for value in vertical_angles
        )
    ):
        raise ValueError("sensor.lidar垂直角数量或取值无效")
    require_positive(
        lidar_cfg,
        (
            "points_per_frame",
            "horizontal_resolution_deg",
            "maximum_interpolation_range_difference_m",
            "maximum_range_m",
            "ray_step_m",
            "mount_height_m",
            "scan_frequency_hz",
            "ray_chunk_size",
        ),
        "sensor.lidar",
    )
    if (
        not isinstance(lidar_cfg["minimum_range_m"], (int, float))
        or not math.isfinite(lidar_cfg["minimum_range_m"])
        or lidar_cfg["minimum_range_m"] < 0.0
        or lidar_cfg["minimum_range_m"] >= lidar_cfg["maximum_range_m"]
        or lidar_cfg["ray_step_m"] > lidar_cfg["maximum_range_m"]
        or lidar_cfg["horizontal_resolution_deg"] > 20.0
        or not isinstance(lidar_cfg["points_per_frame"], int)
        or lidar_cfg["points_per_frame"] % channels != 0
        or not isinstance(lidar_cfg["ray_chunk_size"], int)
        or not isinstance(lidar_cfg["motion_distortion"], bool)
    ):
        raise ValueError("sensor.lidar量程、分辨率或扫描配置无效")
    projection_cfg = lidar_cfg["projection"]
    if not isinstance(projection_cfg, dict):
        raise ValueError("sensor.lidar.projection必须为YAML映射")
    require_positive(
        projection_cfg,
        (
            "input_crop_length_x_m",
            "input_crop_length_y_m",
            "body_height_m",
            "minimum_points_per_cell",
            "height_quantization_m",
            "maximum_ground_deviation_m",
        ),
        "sensor.lidar.projection",
    )
    if (
        projection_cfg["vertical_max_offset_m"]
        <= projection_cfg["vertical_min_offset_m"]
        or not isinstance(projection_cfg["minimum_points_per_cell"], int)
        or not 0.0 <= projection_cfg["ground_percentile"] <= 1.0
        or not 0.0
        <= projection_cfg["span_lower_percentile"]
        <= projection_cfg["span_upper_percentile"]
        <= 1.0
        or not 0.0 <= projection_cfg["fusion_span_percentile"] <= 1.0
    ):
        raise ValueError("sensor.lidar.projection裁剪或分位数配置无效")

    terrain_cfg = config["terrain"]
    for key in (
        "ramp_slope_range",
        "step_height_range_m",
        "step_width_range_m",
        "rough_amplitude_range_m",
        "pit_depth_range_m",
        "pit_half_width_range_m",
        "obstacle_height_range_m",
        "low_obstacle_height_range_m",
        "barrier_half_width_range_m",
        "friction_range",
    ):
        require_range(terrain_cfg[key], f"terrain.{key}", minimum=0.0)
    if not terrain_cfg["enabled_types"]:
        raise ValueError("terrain.enabled_types不能为空")
    if len(set(terrain_cfg["enabled_types"])) != len(terrain_cfg["enabled_types"]):
        raise ValueError("terrain.enabled_types不能包含重复项")
    unknown_terrains = set(terrain_cfg["enabled_types"]) - set(TERRAIN_NAMES)
    if unknown_terrains:
        raise ValueError(f"未知地形类型：{sorted(unknown_terrains)}")
    if not 0.0 < terrain_cfg["challenge_barrier_width_scale"] <= 1.0:
        raise ValueError("terrain.challenge_barrier_width_scale必须位于(0,1]")
    if terrain_cfg["barrier_navigation_clearance_m"] <= 0.0:
        raise ValueError("terrain.barrier_navigation_clearance_m必须大于0")
    pit_depth_minimum, pit_depth_maximum = terrain_cfg["pit_depth_range_m"]
    pit_width_minimum, pit_width_maximum = terrain_cfg["pit_half_width_range_m"]
    if not (
        pit_depth_minimum
        <= terrain_cfg["pit_curriculum_start_depth_max_m"]
        <= pit_depth_maximum
        and pit_width_minimum
        <= terrain_cfg["pit_curriculum_start_half_width_max_m"]
        <= pit_width_maximum
        and pit_depth_minimum
        <= terrain_cfg["pit_navigation_avoidance_depth_m"]
        <= pit_depth_maximum
        and terrain_cfg["pit_goal_clearance_m"] > 0.0
    ):
        raise ValueError("terrain中的pit渐进课程参数无效")

    curriculum_cfg = config["curriculum"]
    if (
        curriculum_cfg["minimum_episodes_per_level"] <= 0
        or curriculum_cfg["minimum_full_difficulty_episodes"] <= 0
    ):
        raise ValueError("课程回合数门槛必须大于0")
    if not (
        1
        <= curriculum_cfg["minimum_level"]
        <= curriculum_cfg["initial_level"]
        <= curriculum_cfg["maximum_level"]
    ):
        raise ValueError("课程初始等级与最高等级无效")
    if curriculum_cfg["maximum_level"] != 9:
        raise ValueError("当前能力课程必须完整定义1至9阶段")
    if not (
        0.0
        <= curriculum_cfg["success_rate_down"]
        < curriculum_cfg["success_rate_up"]
        <= 1.0
    ):
        raise ValueError("课程成功率阈值无效")
    if not 0.0 < curriculum_cfg["success_rate_smoothing"] <= 1.0:
        raise ValueError("curriculum.success_rate_smoothing必须位于(0,1]")
    if not 0.0 < curriculum_cfg["full_difficulty_threshold"] <= 1.0:
        raise ValueError("curriculum.full_difficulty_threshold必须位于(0,1]")
    if not isinstance(curriculum_cfg["allow_level_demotion"], bool):
        raise ValueError("curriculum.allow_level_demotion必须为布尔值")
    if not 0.0 <= curriculum_cfg["frontier_sampling_probability"] <= 1.0:
        raise ValueError("curriculum.frontier_sampling_probability必须位于[0,1]")
    if not 0.0 <= curriculum_cfg["challenge_sampling_probability"] <= 1.0:
        raise ValueError("curriculum.challenge_sampling_probability必须位于[0,1]")
    if (
        curriculum_cfg["frontier_sampling_probability"]
        + curriculum_cfg["challenge_sampling_probability"]
        > 1.0
    ):
        raise ValueError("课程前沿与挑战采样概率之和不能超过1")
    if (
        not isinstance(curriculum_cfg["challenge_level_span"], int)
        or curriculum_cfg["challenge_level_span"] <= 0
    ):
        raise ValueError("curriculum.challenge_level_span必须为正整数")
    stages = curriculum_cfg.get("stages")
    if not isinstance(stages, dict) or {int(key) for key in stages} != set(
        range(1, 10)
    ):
        raise ValueError("curriculum.stages必须完整定义1至9阶段")
    enabled_terrains = set(terrain_cfg["enabled_types"])
    for stage in range(1, 10):
        values = stages.get(stage, stages.get(str(stage)))
        if not isinstance(values, dict) or not isinstance(values.get("name"), str):
            raise ValueError(f"curriculum.stages.{stage}缺少阶段名称")
        terrain_weights = values.get("terrain_weights")
        if (
            not isinstance(terrain_weights, dict)
            or not terrain_weights
            or not set(terrain_weights).issubset(enabled_terrains)
            or any(
                not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight < 0.0
                for weight in terrain_weights.values()
            )
            or sum(terrain_weights.values()) <= 0.0
        ):
            raise ValueError(
                f"curriculum.stages.{stage}.terrain_weights无效"
            )
        goal_bounds = values.get("goal_distance_m")
        require_range(
            goal_bounds,
            f"curriculum.stages.{stage}.goal_distance_m",
            minimum=0.0,
        )
        if (
            goal_bounds[0] <= termination_cfg["goal_tolerance_m"]
            or goal_bounds[0] < goal_cfg["minimum_distance_m"]
            or goal_bounds[1] > goal_cfg["maximum_distance_m"]
            or goal_bounds[0] >= goal_bounds[1]
        ):
            raise ValueError(
                f"curriculum.stages.{stage}.goal_distance_m超出局部目标接口"
            )
        heading_range = values.get("heading_half_range_rad")
        if (
            not isinstance(heading_range, (int, float))
            or not math.isfinite(heading_range)
            or not 0.0 <= heading_range <= math.pi
        ):
            raise ValueError(
                f"curriculum.stages.{stage}.heading_half_range_rad必须位于[0,pi]"
            )
        randomization_scale = values.get("domain_randomization_scale")
        if (
            not isinstance(randomization_scale, (int, float))
            or not math.isfinite(randomization_scale)
            or not 0.0 <= randomization_scale <= 1.0
        ):
            raise ValueError(
                f"curriculum.stages.{stage}.domain_randomization_scale必须位于[0,1]"
            )
        geometry_range = values.get("geometry_difficulty_range")
        require_range(
            geometry_range,
            f"curriculum.stages.{stage}.geometry_difficulty_range",
            minimum=0.0,
        )
        if geometry_range[1] > 1.0:
            raise ValueError(
                f"curriculum.stages.{stage}.geometry_difficulty_range必须位于[0,1]"
            )

    reward_cfg = config["reward"]
    for key in (
        "regression",
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
    ):
        if reward_cfg[key] > 0.0:
            raise ValueError(f"reward.{key}必须为非正惩罚项")
    for key in ("progress", "goal_reached", "heading", "forward_to_goal"):
        if reward_cfg[key] < 0.0:
            raise ValueError(f"reward.{key}必须为非负奖励项")

    training_cfg = config["training"]
    require_positive(training_cfg, ("num_envs", "maximum_iterations"), "training")
    runner_cfg = config["runner"]
    require_positive(runner_cfg, ("num_steps_per_env", "save_interval"), "runner")
    if runner_cfg["clip_actions"] is not None and runner_cfg["clip_actions"] <= 0.0:
        raise ValueError("runner.clip_actions必须为null或正数")

    ppo_cfg = config["ppo"]
    require_positive(
        ppo_cfg,
        (
            "value_loss_coef",
            "clip_param",
            "num_learning_epochs",
            "num_mini_batches",
            "learning_rate",
            "gamma",
            "lam",
            "desired_kl",
            "max_grad_norm",
        ),
        "ppo",
    )
    if not 0.0 < ppo_cfg["gamma"] <= 1.0 or not 0.0 < ppo_cfg["lam"] <= 1.0:
        raise ValueError("ppo.gamma与ppo.lam必须位于(0,1]")
    if ppo_cfg["entropy_coef"] < 0.0:
        raise ValueError("ppo.entropy_coef不能为负数")

    model_cfg = config["model"]
    if model_cfg.get("architecture") != "compact_map_motion_gru_v6":
        raise ValueError("model.architecture必须为compact_map_motion_gru_v6")
    require_positive(
        model_cfg,
        (
            "map_feature_dim",
            "motion_gru_hidden_dim",
            "auxiliary_hidden_dim",
            "fusion_hidden_dim",
            "map_pool_size",
            "initial_action_std",
            "minimum_action_std",
            "maximum_action_std",
            "maximum_pre_tanh_mean",
        ),
        "model",
    )
    if (
        not isinstance(model_cfg["map_encoder_channels"], (list, tuple))
        or len(model_cfg["map_encoder_channels"]) != 4
        or any(
            not isinstance(value, int) or value <= 0
            for value in model_cfg["map_encoder_channels"]
        )
    ):
        raise ValueError("model.map_encoder_channels必须包含四个正整数")
    if model_cfg["map_encoder_channels"][-1] % 4 != 0:
        raise ValueError(
            "compact_map_motion_gru_v6最后一级通道数必须为4的倍数"
        )
    if not (
        model_cfg["minimum_action_std"]
        < model_cfg["initial_action_std"]
        < model_cfg["maximum_action_std"]
    ):
        raise ValueError(
            "model动作标准差必须满足minimum < initial < maximum"
        )
    if not model_cfg["critic_hidden_dims"] or any(
        value <= 0 for value in model_cfg["critic_hidden_dims"]
    ):
        raise ValueError("model.critic_hidden_dims必须包含正整数")


def apply_environment_config(env_cfg, config: dict, config_directory: str | Path | None = None) -> None:
    """Apply editable project YAML values to an Isaac Lab environment config."""
    map_cfg = config["map"]
    goal_cfg = config["goal"]
    history_cfg = config["history"]
    termination_cfg = config["termination"]
    curriculum_cfg = config["curriculum"]
    training_cfg = config["training"]

    env_cfg.map_extent_m = float(map_cfg["extent_m"])
    env_cfg.map_size = int(map_cfg["output_size"])
    env_cfg.map_channels = int(map_cfg["channels"])
    env_cfg.actor_map_channels = int(map_cfg["actor_channels"])
    env_cfg.command_history_length = int(history_cfg["command_length"])
    env_cfg.motion_history_length = int(history_cfg["motion_length"])
    env_cfg.minimum_observed_ratio = float(map_cfg["minimum_observed_ratio"])
    env_cfg.minimum_height_valid_ratio = float(map_cfg["minimum_height_valid_ratio"])
    env_cfg.observation_space = policy_observation_dimension(
        map_channels=env_cfg.actor_map_channels,
        map_size=env_cfg.map_size,
        command_history_length=env_cfg.command_history_length,
        motion_history_length=env_cfg.motion_history_length,
    )
    env_cfg.local_goal_minimum_m = float(goal_cfg["minimum_distance_m"])
    env_cfg.local_goal_maximum_m = float(goal_cfg["maximum_distance_m"])
    env_cfg.goal_tolerance_m = float(termination_cfg["goal_tolerance_m"])
    env_cfg.episode_length_s = float(termination_cfg["maximum_episode_s"])
    env_cfg.maximum_distance_m = float(termination_cfg["maximum_distance_from_origin_m"])
    env_cfg.collision_height_range_m = float(termination_cfg["collision_height_range_m"])
    env_cfg.unstable_risk_threshold = float(termination_cfg["unstable_risk_threshold"])
    env_cfg.maximum_tilt_rad = float(termination_cfg["maximum_tilt_rad"])
    env_cfg.fall_tilt_rad = float(termination_cfg["fall_tilt_rad"])
    env_cfg.stuck_timeout_s = float(termination_cfg["stuck_timeout_s"])
    env_cfg.maximum_bad_observation_steps = int(termination_cfg["maximum_bad_observation_steps"])
    env_cfg.curriculum_maximum_stage = int(curriculum_cfg["maximum_level"])
    if config_directory is not None:
        env_cfg.project_config_directory = str(Path(config_directory).resolve())

    env_cfg.scene.num_envs = int(training_cfg["num_envs"])
    env_cfg.sim.device = str(training_cfg["device"])


def apply_curriculum_stage_sampling_range(
    env_cfg,
    minimum_stage: int | None,
    maximum_stage: int | None,
) -> None:
    """Override capability-stage sampling for evaluation or stress tests."""
    if minimum_stage is None and maximum_stage is None:
        return
    minimum = 1 if minimum_stage is None else int(minimum_stage)
    maximum = (
        int(env_cfg.curriculum_maximum_stage)
        if maximum_stage is None
        else int(maximum_stage)
    )
    if not 1 <= minimum <= maximum <= int(env_cfg.curriculum_maximum_stage):
        raise ValueError(
            "课程阶段采样范围必须满足1 <= minimum <= maximum <= "
            f"{env_cfg.curriculum_maximum_stage}"
        )
    env_cfg.curriculum_sampling_minimum_stage = minimum
    env_cfg.curriculum_sampling_maximum_stage = maximum


def apply_project_config(env_cfg, agent_cfg, config: dict, config_directory: str | Path | None = None) -> None:
    """Apply editable project YAML values to Isaac Lab and RSL-RL config objects."""
    apply_environment_config(env_cfg, config, config_directory)
    training_cfg = config["training"]
    model_cfg = config["model"]
    runner_cfg = config["runner"]
    ppo_cfg = config["ppo"]
    agent_cfg.seed = int(training_cfg["seed"])
    agent_cfg.device = str(training_cfg["device"])
    agent_cfg.max_iterations = int(training_cfg["maximum_iterations"])
    agent_cfg.num_steps_per_env = int(runner_cfg["num_steps_per_env"])
    agent_cfg.save_interval = int(runner_cfg["save_interval"])
    agent_cfg.experiment_name = str(runner_cfg["experiment_name"])
    agent_cfg.run_name = str(runner_cfg["run_name"])
    agent_cfg.clip_actions = runner_cfg["clip_actions"]
    agent_cfg.policy.map_channels = env_cfg.actor_map_channels
    agent_cfg.policy.map_size = env_cfg.map_size
    agent_cfg.policy.command_history_length = env_cfg.command_history_length
    agent_cfg.policy.motion_history_length = env_cfg.motion_history_length
    agent_cfg.policy.architecture = str(model_cfg["architecture"])
    agent_cfg.policy.map_encoder_channels = list(
        model_cfg["map_encoder_channels"]
    )
    agent_cfg.policy.map_pool_size = int(model_cfg["map_pool_size"])
    agent_cfg.policy.map_feature_dim = int(model_cfg["map_feature_dim"])
    agent_cfg.policy.motion_gru_hidden_dim = int(
        model_cfg["motion_gru_hidden_dim"]
    )
    agent_cfg.policy.auxiliary_hidden_dim = int(model_cfg["auxiliary_hidden_dim"])
    agent_cfg.policy.fusion_hidden_dim = int(model_cfg["fusion_hidden_dim"])
    agent_cfg.policy.init_noise_std = float(model_cfg["initial_action_std"])
    agent_cfg.policy.minimum_action_std = float(model_cfg["minimum_action_std"])
    agent_cfg.policy.maximum_action_std = float(model_cfg["maximum_action_std"])
    agent_cfg.policy.maximum_pre_tanh_mean = float(
        model_cfg["maximum_pre_tanh_mean"]
    )
    agent_cfg.policy.runtime_finite_checks = bool(
        model_cfg.get("runtime_finite_checks", False)
    )
    agent_cfg.policy.critic_hidden_dims = list(model_cfg["critic_hidden_dims"])
    for key, value in ppo_cfg.items():
        setattr(agent_cfg.algorithm, key, value)
