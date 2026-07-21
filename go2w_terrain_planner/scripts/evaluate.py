"""Headless checkpoint evaluation with persisted aggregate metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


RSL_SCRIPT_DIRECTORY = Path(__file__).resolve().parent / "rsl_rl"
sys.path.insert(0, str(RSL_SCRIPT_DIRECTORY))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Evaluate the Go2W high-level planner.")
parser.add_argument("--task", default="Go2W-Terrain-Navigation-Direct-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--project-config-dir", "--project_config_dir", dest="project_config_dir", default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import go2w_terrain_planner.tasks  # noqa: F401
from go2w_terrain_planner.models import Go2wActorCritic
from go2w_terrain_planner.utils.config_loader import apply_project_config, load_project_config

rsl_on_policy_runner.Go2wActorCritic = Go2wActorCritic


def main() -> None:
    if args_cli.num_envs <= 0 or args_cli.steps <= 0:
        raise ValueError("num_envs和steps必须大于0")
    project_config = load_project_config(args_cli.project_config_dir)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    apply_project_config(env_cfg, agent_cfg, project_config, args_cli.project_config_dir)
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    env_cfg.seed = agent_cfg.seed

    data_root = Path(os.environ.get("GO2W_DATA_ROOT", "/workspace/data"))
    runs_root = (
        data_root / "runs"
        if os.environ.get("GO2W_DATA_ROOT")
        else Path(project_config["training"]["output_root"])
    )
    experiment_root = runs_root / "rsl_rl" / agent_cfg.experiment_name
    checkpoint = (
        Path(args_cli.checkpoint)
        if args_cli.checkpoint
        else Path(get_checkpoint_path(str(experiment_root), agent_cfg.load_run, agent_cfg.load_checkpoint))
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint不存在：{checkpoint}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    observation = env.get_observations()
    reward_sum = torch.zeros(args_cli.num_envs, device=env.unwrapped.device)
    termination_count = 0
    estimated_successes = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            action = policy(observation)
            observation, reward, done, extras = env.step(action)
            if not torch.isfinite(reward).all() or not torch.isfinite(action).all():
                raise RuntimeError("评估期间出现NaN或Inf")
            reward_sum += reward
            completed_now = int(done.sum().item())
            termination_count += completed_now
            episode_log = extras.get("log", {}) if hasattr(extras, "get") else {}
            if completed_now and "Episode/success_rate" in episode_log:
                success_rate = episode_log["Episode/success_rate"]
                if isinstance(success_rate, torch.Tensor):
                    success_rate = float(success_rate.item())
                estimated_successes += float(success_rate) * completed_now
    elapsed = time.perf_counter() - started
    env.close()

    output = data_root / "evaluations" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "task": args_cli.task,
        "checkpoint": str(checkpoint),
        "seed": int(agent_cfg.seed),
        "num_envs": args_cli.num_envs,
        "steps_per_env": args_cli.steps,
        "mean_return_over_window": float(reward_sum.mean().item()),
        "terminations": termination_count,
        "estimated_success_rate": estimated_successes / max(termination_count, 1),
        "environment_steps_per_second": args_cli.num_envs * args_cli.steps / max(elapsed, 1.0e-9),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
