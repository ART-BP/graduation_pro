from pathlib import Path

from go2w_terrain_planner.utils.config_loader import load_project_config


def test_project_configuration_is_consistent() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    config = load_project_config(config_dir)
    assert config["map"]["source_size"] == 200
    assert config["map"]["output_size"] == 100
    assert config["map"]["history_length"] == 5
