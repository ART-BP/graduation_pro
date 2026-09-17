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
parser.add_argument(
    "--curriculum-min-stage",
    type=int,
    default=1,
    help="Minimum sampled capability stage.",
)
parser.add_argument(
    "--curriculum-max-stage",
    type=int,
    default=9,
    help="Maximum sampled capability stage.",
)
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
from go2w_terrain_planner.mapping.simulated_local_map import TERRAIN_NAMES
from go2w_terrain_planner.utils.config_loader import (
    apply_project_config,
    apply_curriculum_stage_sampling_range,
    load_project_config,
)
from go2w_terrain_planner.utils.logging_utils import validate_checkpoint

rsl_on_policy_runner.Go2wActorCritic = Go2wActorCritic


def main() -> None:
    if args_cli.num_envs <= 0 or args_cli.steps <= 0:
        raise ValueError("num_envs和steps必须大于0")
    project_config = load_project_config(args_cli.project_config_dir)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    apply_project_config(env_cfg, agent_cfg, project_config, args_cli.project_config_dir)
    apply_curriculum_stage_sampling_range(
        env_cfg,
        args_cli.curriculum_min_stage,
        args_cli.curriculum_max_stage,
    )
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
    validate_checkpoint(
        checkpoint,
        include_optimizer=False,
        required_policy_architecture_version=6,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    observation = env.get_observations()
    reward_sum = torch.zeros(args_cli.num_envs, device=env.unwrapped.device)
    episode_return = torch.zeros_like(reward_sum)
    episode_length = torch.zeros(
        args_cli.num_envs, dtype=torch.long, device=env.unwrapped.device
    )
    completed_return_sum = 0.0
    completed_length_sum = 0
    termination_count = 0
    event_names = (
        "success",
        "collision",
        "unstable",
        "stuck",
        "timeout",
        "out_of_bounds",
        "observation_failure",
    )
    event_counts = {name: 0.0 for name in event_names}
    terrain_step_counts = torch.zeros(
        len(TERRAIN_NAMES), dtype=torch.long, device=env.unwrapped.device
    )
    terrain_episode_counts = torch.zeros_like(terrain_step_counts)
    terrain_success_counts = torch.zeros_like(terrain_step_counts)
    stage_step_counts = torch.zeros(
        10, dtype=torch.long, device=env.unwrapped.device
    )
    stage_episode_counts = torch.zeros_like(stage_step_counts)
    stage_success_counts = torch.zeros_like(stage_step_counts)
    stage_names = dict(env.unwrapped.curriculum_schedule.stage_names)
    active_terrain = env.unwrapped.map_generator.terrain_type.clone()
    active_stage = env.unwrapped.current_curriculum_stage.clone()
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            terrain_step_counts += torch.bincount(
                active_terrain, minlength=len(TERRAIN_NAMES)
            )
            stage_step_counts += torch.bincount(active_stage, minlength=10)
            action = policy(observation)
            observation, reward, done, extras = env.step(action)
            if not torch.isfinite(reward).all() or not torch.isfinite(action).all():
                raise RuntimeError("评估期间出现NaN或Inf")
            reward_sum += reward
            episode_return += reward
            episode_length += 1
            completed_now = int(done.sum().item())
            termination_count += completed_now
            if completed_now:
                terrain_episode_counts += torch.bincount(
                    active_terrain[done], minlength=len(TERRAIN_NAMES)
                )
                stage_episode_counts += torch.bincount(
                    active_stage[done], minlength=10
                )
                completed_return_sum += float(episode_return[done].sum().item())
                completed_length_sum += int(episode_length[done].sum().item())
                episode_return[done] = 0.0
                episode_length[done] = 0
            episode_log = extras.get("log", {}) if hasattr(extras, "get") else {}
            if completed_now:
                for name in event_names:
                    key = f"Episode/{name}_rate"
                    if key not in episode_log:
                        continue
                    rate = episode_log[key]
                    if isinstance(rate, torch.Tensor):
                        rate = float(rate.item())
                    event_counts[name] += float(rate) * completed_now
                for terrain_index, terrain_name in enumerate(TERRAIN_NAMES):
                    key = f"Terrain/{terrain_name}_success_count"
                    if key not in episode_log:
                        continue
                    success_count = episode_log[key]
                    if isinstance(success_count, torch.Tensor):
                        success_count = float(success_count.item())
                    terrain_success_counts[terrain_index] += int(
                        round(float(success_count))
                    )
                for stage, stage_name in stage_names.items():
                    key = f"Stage/{stage}_{stage_name}_success_count"
                    if key not in episode_log:
                        continue
                    success_count = episode_log[key]
                    if isinstance(success_count, torch.Tensor):
                        success_count = float(success_count.item())
                    stage_success_counts[stage] += int(
                        round(float(success_count))
                    )
            active_terrain.copy_(env.unwrapped.map_generator.terrain_type)
            active_stage.copy_(env.unwrapped.current_curriculum_stage)
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
        "curriculum_stage_sampling_range": {
            "minimum_stage": args_cli.curriculum_min_stage,
            "maximum_stage": args_cli.curriculum_max_stage,
        },
        "terrain_step_counts": {
            name: int(terrain_step_counts[index].item())
            for index, name in enumerate(TERRAIN_NAMES)
        },
        "terrain_completed_episode_counts": {
            name: int(terrain_episode_counts[index].item())
            for index, name in enumerate(TERRAIN_NAMES)
        },
        "terrain_success_rates": {
            name: (
                float(terrain_success_counts[index].item())
                / int(terrain_episode_counts[index].item())
                if terrain_episode_counts[index].item() > 0
                else None
            )
            for index, name in enumerate(TERRAIN_NAMES)
        },
        "curriculum_stage_step_counts": {
            f"{stage}_{name}": int(stage_step_counts[stage].item())
            for stage, name in stage_names.items()
        },
        "curriculum_stage_completed_episode_counts": {
            f"{stage}_{name}": int(stage_episode_counts[stage].item())
            for stage, name in stage_names.items()
        },
        "curriculum_stage_success_rates": {
            f"{stage}_{name}": (
                float(stage_success_counts[stage].item())
                / int(stage_episode_counts[stage].item())
                if stage_episode_counts[stage].item() > 0
                else None
            )
            for stage, name in stage_names.items()
        },
        "mean_return_over_window": float(reward_sum.mean().item()),
        "mean_completed_episode_return": (
            completed_return_sum / termination_count if termination_count else None
        ),
        "mean_completed_episode_length_steps": (
            completed_length_sum / termination_count if termination_count else None
        ),
        "terminations": termination_count,
        "event_rates": {
            name: count / termination_count if termination_count else None
            for name, count in event_counts.items()
        },
        "environment_steps_per_second": args_cli.num_envs * args_cli.steps / max(elapsed, 1.0e-9),
    }
    metrics["estimated_success_rate"] = metrics["event_rates"]["success"]
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
