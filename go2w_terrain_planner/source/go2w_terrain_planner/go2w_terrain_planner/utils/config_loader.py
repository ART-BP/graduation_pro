"""Load and validate the editable project YAML files."""

from __future__ import annotations

import math
import os
from pathlib import Path

import yaml


CONFIG_NAMES = ("observation", "action", "reward", "terrain", "sensor", "training")

DEFAULT_LIDAR_CONFIG = {
    "channels": 16,
    "vertical_angles_deg": [
        -15.0,
        -13.0,
        -11.0,
        -9.0,
        -7.0,
        -5.0,
        -3.0,
        -1.0,
        1.0,
        3.0,
        5.0,
        7.0,
        9.0,
        11.0,
        13.0,
        15.0,
    ],
    "horizontal_resolution_deg": 2.0,
    "minimum_range_m": 0.20,
    "maximum_range_m": 12.0,
    "ray_step_m": 0.08,
    "mount_height_m": 0.45,
    "scan_frequency_hz": 10.0,
    "motion_distortion": True,
    "ray_chunk_size": 256,
}


def _apply_backward_compatible_defaults(config: dict) -> None:
    """Keep pre-lidar run snapshots loadable for evaluation and export."""

    sensor_cfg = config.get("sensor")
    if not isinstance(sensor_cfg, dict):
        return
    # A historical snapshot without this key was trained with the dense
    # analytic observation model. Never silently reinterpret it as lidar.
    sensor_cfg.setdefault("observation_source", "analytic")
    lidar_cfg = sensor_cfg.setdefault("lidar", {})
    if isinstance(lidar_cfg, dict):
        for key, value in DEFAULT_LIDAR_CONFIG.items():
            lidar_cfg.setdefault(key, value.copy() if isinstance(value, list) else value)


def default_config_directory() -> Path:
    project_root = os.environ.get("GO2W_PROJECT_ROOT")
    if project_root:
        return Path(project_root) / "configs"
    return Path(__file__).resolve().parents[4] / "configs"


def load_project_config(config_directory: str | Path | None = None) -> dict:
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
    _apply_backward_compatible_defaults(result)
    validate_project_config(result)
    return result


def validate_project_config(config: dict) -> None:
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
        for key in keys:
            value = values[key]
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{section}.{key}必须为有限正数")

    def require_range(bounds, name: str, *, minimum: float | None = None) -> None:
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
    if map_cfg["history_length"] < 3 or map_cfg["history_length"] > 5:
        raise ValueError("地图历史长度必须为3到5")
    if map_cfg["channels"] != 4:
        raise ValueError("第一版地图通道数必须为4")
    if not map_cfg["normalize_heights"]:
        raise ValueError("第一版时序高度补偿要求normalize_heights=true")
    for key in ("minimum_observed_ratio", "minimum_height_valid_ratio"):
        if not 0.0 <= map_cfg[key] <= 1.0:
            raise ValueError(f"{key}必须位于[0,1]")
    if (
        map_cfg["observation_fusion_length"] < 2
        or map_cfg["observation_fusion_length"] > 20
    ):
        raise ValueError("单张地图快照的有效观测融合长度必须为2到20")
    if history_cfg["command_length"] != map_cfg["history_length"] - 1:
        raise ValueError("指令历史长度必须等于地图历史长度减1")
    if history_cfg["motion_length"] != map_cfg["history_length"] - 1:
        raise ValueError("运动历史长度必须等于地图历史长度减1")
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
    require_positive(
        goal_cfg,
        (
            "maximum_distance_m",
            "reset_minimum_distance_m",
            "curriculum_start_maximum_m",
        ),
        "goal",
    )
    if (
        not isinstance(goal_cfg["minimum_distance_m"], (int, float))
        or not math.isfinite(goal_cfg["minimum_distance_m"])
        or goal_cfg["minimum_distance_m"] < 0.0
        or goal_cfg["minimum_distance_m"] >= goal_cfg["maximum_distance_m"]
        or not (
            goal_cfg["minimum_distance_m"]
            <= goal_cfg["reset_minimum_distance_m"]
            < goal_cfg["curriculum_start_maximum_m"]
            <= goal_cfg["maximum_distance_m"]
        )
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
    if goal_cfg["reset_minimum_distance_m"] <= termination_cfg["goal_tolerance_m"]:
        raise ValueError("goal.reset_minimum_distance_m必须大于终点容差")
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
            "horizontal_resolution_deg",
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
        or not isinstance(lidar_cfg["ray_chunk_size"], int)
        or not isinstance(lidar_cfg["motion_distortion"], bool)
    ):
        raise ValueError("sensor.lidar量程、分辨率或扫描配置无效")

    terrain_cfg = config["terrain"]
    for key in (
        "ramp_slope_range",
        "step_height_range_m",
        "step_width_range_m",
        "rough_amplitude_range_m",
        "pit_depth_range_m",
        "pit_half_width_range_m",
        "obstacle_height_range_m",
        "barrier_half_width_range_m",
        "friction_range",
    ):
        require_range(terrain_cfg[key], f"terrain.{key}", minimum=0.0)
    if not terrain_cfg["enabled_types"]:
        raise ValueError("terrain.enabled_types不能为空")
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
        0
        <= curriculum_cfg["minimum_level"]
        <= curriculum_cfg["initial_level"]
        <= curriculum_cfg["maximum_level"]
    ):
        raise ValueError("课程初始等级与最高等级无效")
    if curriculum_cfg["maximum_level"] > 9:
        raise ValueError("课程最高等级不能超过当前十类地形的索引范围")
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
    require_positive(
        model_cfg,
        (
            "map_feature_dim",
            "temporal_hidden_dim",
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
    env_cfg.map_history_length = int(map_cfg["history_length"])
    env_cfg.command_history_length = int(history_cfg["command_length"])
    env_cfg.motion_history_length = int(history_cfg["motion_length"])
    env_cfg.minimum_observed_ratio = float(map_cfg["minimum_observed_ratio"])
    env_cfg.minimum_height_valid_ratio = float(map_cfg["minimum_height_valid_ratio"])
    env_cfg.observation_space = (
        env_cfg.map_history_length * env_cfg.map_channels * env_cfg.map_size * env_cfg.map_size
        + 3
        + 2
        + 2 * env_cfg.command_history_length
        + 3 * env_cfg.motion_history_length
    )
    env_cfg.local_goal_minimum_m = float(goal_cfg["minimum_distance_m"])
    env_cfg.local_goal_maximum_m = float(goal_cfg["maximum_distance_m"])
    env_cfg.local_goal_reset_minimum_m = float(
        goal_cfg["reset_minimum_distance_m"]
    )
    env_cfg.local_goal_curriculum_start_maximum_m = float(
        goal_cfg["curriculum_start_maximum_m"]
    )
    env_cfg.goal_tolerance_m = float(termination_cfg["goal_tolerance_m"])
    env_cfg.episode_length_s = float(termination_cfg["maximum_episode_s"])
    env_cfg.maximum_distance_m = float(termination_cfg["maximum_distance_from_origin_m"])
    env_cfg.collision_height_range_m = float(termination_cfg["collision_height_range_m"])
    env_cfg.unstable_risk_threshold = float(termination_cfg["unstable_risk_threshold"])
    env_cfg.maximum_tilt_rad = float(termination_cfg["maximum_tilt_rad"])
    env_cfg.fall_tilt_rad = float(termination_cfg["fall_tilt_rad"])
    env_cfg.stuck_timeout_s = float(termination_cfg["stuck_timeout_s"])
    env_cfg.maximum_bad_observation_steps = int(termination_cfg["maximum_bad_observation_steps"])
    env_cfg.curriculum_maximum_terrain_index = int(curriculum_cfg["maximum_level"])
    if config_directory is not None:
        env_cfg.project_config_directory = str(Path(config_directory).resolve())

    env_cfg.scene.num_envs = int(training_cfg["num_envs"])
    env_cfg.sim.device = str(training_cfg["device"])


def apply_terrain_sampling_range(
    env_cfg,
    minimum_index: int | None,
    maximum_index: int | None,
) -> None:
    """Override curriculum sampling for evaluation or short stress tests."""
    if minimum_index is None and maximum_index is None:
        return
    minimum = 0 if minimum_index is None else int(minimum_index)
    maximum = (
        int(env_cfg.curriculum_maximum_terrain_index)
        if maximum_index is None
        else int(maximum_index)
    )
    if not 0 <= minimum <= maximum <= int(env_cfg.curriculum_maximum_terrain_index):
        raise ValueError(
            "地形采样范围必须满足0 <= minimum <= maximum <= "
            f"{env_cfg.curriculum_maximum_terrain_index}"
        )
    env_cfg.terrain_sampling_minimum_index = minimum
    env_cfg.terrain_sampling_maximum_index = maximum


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
    agent_cfg.policy.map_history_length = env_cfg.map_history_length
    agent_cfg.policy.map_channels = env_cfg.map_channels
    agent_cfg.policy.map_size = env_cfg.map_size
    agent_cfg.policy.map_encoder_channels = list(
        model_cfg["map_encoder_channels"]
    )
    agent_cfg.policy.map_pool_size = int(model_cfg["map_pool_size"])
    agent_cfg.policy.map_feature_dim = int(model_cfg["map_feature_dim"])
    agent_cfg.policy.temporal_hidden_dim = int(model_cfg["temporal_hidden_dim"])
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
