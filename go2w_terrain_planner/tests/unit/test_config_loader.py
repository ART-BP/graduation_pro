import shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from go2w_terrain_planner.utils.config_loader import (
    apply_curriculum_stage_sampling_range,
    load_project_config,
    validate_project_config,
)


def test_project_configuration_is_consistent() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    config = load_project_config(config_dir)
    assert config["map"]["source_size"] == 200
    assert config["map"]["output_size"] == 200
    assert config["map"]["max_abs_relative_height_m"] == 1.0
    assert config["map"]["actor_channels"] == 7
    assert config["map"]["observation_fusion_length"] == 12
    assert config["history"]["command_length"] == 4
    assert config["history"]["motion_length"] == 4
    assert config["goal"]["minimum_distance_m"] == 0.0
    assert config["goal"]["maximum_distance_m"] == 10.0
    assert config["curriculum"]["minimum_level"] == 1
    assert config["curriculum"]["initial_level"] == 1
    assert config["curriculum"]["stages"][1]["terrain_weights"] == {"flat": 1.0}
    assert config["curriculum"]["stages"][9]["goal_distance_m"] == [5.0, 10.0]
    assert config["model"]["initial_action_std"] == 0.4
    assert config["model"]["minimum_action_std"] == 0.08
    assert config["model"]["maximum_action_std"] == 0.6
    assert config["model"]["maximum_pre_tanh_mean"] == 1.5
    assert config["model"]["architecture"] == "compact_map_motion_gru_v6"
    assert config["model"]["map_encoder_channels"] == [32, 64, 96, 128]
    assert config["model"]["map_pool_size"] == 8
    assert config["model"]["map_feature_dim"] == 384
    assert config["model"]["motion_gru_hidden_dim"] == 128
    assert config["model"]["fusion_hidden_dim"] == 384
    assert config["ppo"]["entropy_coef"] == 0.001
    assert config["ppo"]["schedule"] == "fixed"
    assert config["curriculum"]["frontier_sampling_probability"] == 0.60
    assert config["curriculum"]["challenge_sampling_probability"] == 0.05
    assert config["curriculum"]["allow_level_demotion"] is False
    assert config["curriculum"]["success_rate_up"] == 0.85
    assert config["curriculum"]["minimum_episodes_per_level"] == 3200
    assert config["curriculum"]["full_difficulty_threshold"] == 0.90
    assert config["curriculum"]["minimum_full_difficulty_episodes"] == 960
    assert config["curriculum"]["success_rate_smoothing"] == 0.005
    assert config["terrain"]["pit_half_width_range_m"] == [0.25, 0.60]
    assert config["terrain"]["pit_curriculum_start_depth_max_m"] == 0.12
    assert config["terrain"]["barrier_navigation_clearance_m"] == 0.35
    assert config["runner"]["num_steps_per_env"] == 96
    assert config["ppo"]["num_learning_epochs"] == 3
    assert config["ppo"]["value_loss_coef"] == 0.5
    assert config["sensor"]["observation_source"] == "raycast"
    assert config["sensor"]["lidar"]["channels"] == 16
    assert config["sensor"]["lidar"]["points_per_frame"] == 32000
    assert len(config["sensor"]["lidar"]["vertical_angles_deg"]) == 16
    assert config["sensor"]["lidar"]["projection"]["ground_percentile"] == 0.10


def test_invalid_training_and_reward_values_are_rejected() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    config = load_project_config(config_dir)

    invalid_training = deepcopy(config)
    invalid_training["ppo"]["learning_rate"] = 0.0
    with pytest.raises(ValueError, match="learning_rate"):
        validate_project_config(invalid_training)

    invalid_reward = deepcopy(config)
    invalid_reward["reward"]["timeout"] = 1.0
    with pytest.raises(ValueError, match="reward.timeout"):
        validate_project_config(invalid_reward)

    invalid_curriculum = deepcopy(config)
    invalid_curriculum["curriculum"]["frontier_sampling_probability"] = 1.1
    with pytest.raises(ValueError, match="frontier_sampling_probability"):
        validate_project_config(invalid_curriculum)

    invalid_demotion = deepcopy(config)
    invalid_demotion["curriculum"]["allow_level_demotion"] = 0
    with pytest.raises(ValueError, match="allow_level_demotion"):
        validate_project_config(invalid_demotion)

    invalid_full_gate = deepcopy(config)
    invalid_full_gate["curriculum"]["minimum_full_difficulty_episodes"] = 0
    with pytest.raises(ValueError, match="课程回合数门槛"):
        validate_project_config(invalid_full_gate)

    invalid_probability_sum = deepcopy(config)
    invalid_probability_sum["curriculum"]["frontier_sampling_probability"] = 0.9
    invalid_probability_sum["curriculum"]["challenge_sampling_probability"] = 0.2
    with pytest.raises(ValueError, match="概率之和"):
        validate_project_config(invalid_probability_sum)

    invalid_stage_goal = deepcopy(config)
    invalid_stage_goal["curriculum"]["stages"][1]["goal_distance_m"] = [0.2, 2.5]
    with pytest.raises(ValueError, match="goal_distance_m"):
        validate_project_config(invalid_stage_goal)

    invalid_stage_terrain = deepcopy(config)
    invalid_stage_terrain["curriculum"]["stages"][4]["terrain_weights"] = {
        "unknown": 1.0
    }
    with pytest.raises(ValueError, match="terrain_weights"):
        validate_project_config(invalid_stage_terrain)

    invalid_action_std = deepcopy(config)
    invalid_action_std["model"]["minimum_action_std"] = 0.4
    with pytest.raises(ValueError, match="动作标准差"):
        validate_project_config(invalid_action_std)

    invalid_encoder = deepcopy(config)
    invalid_encoder["model"]["map_encoder_channels"] = [24, 48, 96]
    with pytest.raises(ValueError, match="map_encoder_channels"):
        validate_project_config(invalid_encoder)

    invalid_architecture = deepcopy(config)
    invalid_architecture["model"]["architecture"] = "flat_mlp"
    with pytest.raises(ValueError, match="model.architecture"):
        validate_project_config(invalid_architecture)

    invalid_history = deepcopy(config)
    invalid_history["history"]["motion_length"] = 3
    with pytest.raises(ValueError, match="GRU"):
        validate_project_config(invalid_history)

    invalid_lidar = deepcopy(config)
    invalid_lidar["sensor"]["lidar"]["vertical_angles_deg"] = [-10.0]
    with pytest.raises(ValueError, match="垂直角"):
        validate_project_config(invalid_lidar)

    invalid_point_count = deepcopy(config)
    invalid_point_count["sensor"]["lidar"]["points_per_frame"] = 32001
    with pytest.raises(ValueError, match="扫描配置"):
        validate_project_config(invalid_point_count)


def test_incomplete_sensor_config_is_rejected(tmp_path: Path) -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    for source in config_dir.glob("*.yaml"):
        shutil.copy2(source, tmp_path / source.name)
    sensor_path = tmp_path / "sensor.yaml"
    sensor_document = yaml.safe_load(sensor_path.read_text(encoding="utf-8"))
    sensor_document["sensor"].pop("observation_source")
    sensor_document["sensor"].pop("lidar")
    sensor_path.write_text(
        yaml.safe_dump(sensor_document, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(KeyError):
        load_project_config(tmp_path)


def test_missing_architecture_is_rejected(tmp_path: Path) -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    for source in config_dir.glob("*.yaml"):
        shutil.copy2(source, tmp_path / source.name)
    training_path = tmp_path / "training.yaml"
    training_document = yaml.safe_load(training_path.read_text(encoding="utf-8"))
    training_document["model"].pop("architecture")
    training_path.write_text(
        yaml.safe_dump(training_document, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model.architecture"):
        load_project_config(tmp_path)


def test_evaluation_terrain_sampling_override_is_validated() -> None:
    env_cfg = SimpleNamespace(
        curriculum_maximum_stage=9,
        curriculum_sampling_minimum_stage=-1,
        curriculum_sampling_maximum_stage=-1,
    )
    apply_curriculum_stage_sampling_range(env_cfg, 2, 9)
    assert env_cfg.curriculum_sampling_minimum_stage == 2
    assert env_cfg.curriculum_sampling_maximum_stage == 9

    with pytest.raises(ValueError, match="课程阶段采样范围"):
        apply_curriculum_stage_sampling_range(env_cfg, 7, 3)
