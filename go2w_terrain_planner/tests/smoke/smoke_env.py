"""Headless GPU smoke test for the actual Go2W task."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--project-config-dir", "--project_config_dir", dest="project_config_dir", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import go2w_terrain_planner.tasks  # noqa: F401
from go2w_terrain_planner.utils.config_loader import apply_environment_config, load_project_config


def assert_finite_observation(observation) -> None:
    values = observation.values() if hasattr(observation, "values") else [observation]
    for value in values:
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise RuntimeError("smoke observation contains NaN or Inf")


def main() -> None:
    if args_cli.steps <= 0 or args_cli.num_envs <= 0:
        raise ValueError("steps和num_envs必须大于0")
    if not torch.cuda.is_available():
        raise RuntimeError("容器未获得CUDA GPU")
    task = "Go2W-Terrain-Navigation-Direct-v0"
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
    project_config = load_project_config(args_cli.project_config_dir)
    apply_environment_config(env_cfg, project_config, args_cli.project_config_dir)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    env = gym.make(task, cfg=env_cfg)
    observation, _ = env.reset()
    assert_finite_observation(observation)
    reset_count = 1
    for step in range(args_cli.steps):
        actions = 2.0 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1.0
        observation, reward, terminated, truncated, _ = env.step(actions)
        assert_finite_observation(observation)
        if not torch.isfinite(reward).all():
            raise RuntimeError("smoke reward contains NaN or Inf")
        if step == args_cli.steps // 2:
            observation, _ = env.reset()
            assert_finite_observation(observation)
            reset_count += 1
        if not torch.isfinite(actions).all() or terminated.shape != truncated.shape:
            raise RuntimeError("smoke action or done signal invalid")
    env.close()
    output = Path(os.environ.get("GO2W_DATA_ROOT", "/workspace/data")) / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "task": task,
        "steps": args_cli.steps,
        "num_envs": args_cli.num_envs,
        "resets": reset_count,
        "gpu": torch.cuda.get_device_name(0),
        "status": "passed",
    }
    (output / "smoke_test.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
