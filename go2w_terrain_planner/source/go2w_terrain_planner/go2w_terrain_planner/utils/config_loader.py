"""Load and validate the editable project YAML files."""

from __future__ import annotations

import math
import os
from pathlib import Path

import yaml


CONFIG_NAMES = ("observation", "action", "reward", "terrain", "sensor", "training")


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
    validate_project_config(result)
    return result


def validate_project_config(config: dict) -> None:
    map_cfg = config["map"]
    history_cfg = config["history"]
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
    if map_cfg["source_size"] <= 0 or map_cfg["output_size"] <= 0:
        raise ValueError("地图尺寸必须大于0")
    if map_cfg["observation_fusion_length"] < 2 or map_cfg["observation_fusion_length"] > 5:
        raise ValueError("单帧地图的有效观测融合长度必须为2到5")
    if history_cfg["command_length"] != map_cfg["history_length"] - 1:
        raise ValueError("指令历史长度必须等于地图历史长度减1")
    if history_cfg["motion_length"] != map_cfg["history_length"] - 1:
        raise ValueError("运动历史长度必须等于地图历史长度减1")
    action_cfg = config["action"]
    if action_cfg["linear_min_mps"] >= action_cfg["linear_max_mps"]:
        raise ValueError("线速度范围无效")
    if action_cfg["angular_min_radps"] >= action_cfg["angular_max_radps"]:
        raise ValueError("角速度范围无效")
    execution_cfg = config["execution_model"]
    if not 0.0 <= execution_cfg["maximum_terrain_speed_loss"] < 1.0:
        raise ValueError("maximum_terrain_speed_loss必须位于[0,1)")
    if execution_cfg["terrain_tilt_gain_rad"] <= 0.0 or execution_cfg["entry_tilt_gain"] < 0.0:
        raise ValueError("地形倾斜增益无效")
    termination_cfg = config["termination"]
    if termination_cfg["fall_tilt_rad"] <= termination_cfg["maximum_tilt_rad"]:
        raise ValueError("跌倒角阈值必须大于失稳角阈值")
    sensor_cfg = config["sensor"]
    for key in ("random_missing_probability", "ray_only_probability"):
        if not 0.0 <= sensor_cfg[key] <= 1.0:
            raise ValueError(f"{key}必须位于[0,1]")
    if sensor_cfg["random_missing_probability"] + sensor_cfg["ray_only_probability"] > 1.0:
        raise ValueError("缺失率与仅射线观测率之和不能超过1")
    if not 0.0 <= sensor_cfg["occlusion_sector_probability"] <= 1.0:
        raise ValueError("occlusion_sector_probability必须位于[0,1]")
    if config["curriculum"]["minimum_episodes_per_level"] <= 0:
        raise ValueError("minimum_episodes_per_level必须大于0")


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
    agent_cfg.policy.map_feature_dim = int(model_cfg["map_feature_dim"])
    agent_cfg.policy.temporal_hidden_dim = int(model_cfg["temporal_hidden_dim"])
    agent_cfg.policy.auxiliary_hidden_dim = int(model_cfg["auxiliary_hidden_dim"])
    agent_cfg.policy.fusion_hidden_dim = int(model_cfg["fusion_hidden_dim"])
    agent_cfg.policy.init_noise_std = float(model_cfg["initial_action_std"])
    agent_cfg.policy.critic_hidden_dims = list(model_cfg["critic_hidden_dims"])
    for key, value in ppo_cfg.items():
        setattr(agent_cfg.algorithm, key, value)
