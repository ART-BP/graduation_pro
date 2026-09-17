# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--curriculum-min-stage",
    type=int,
    default=None,
    help="Optional minimum capability stage for a short stress-training run.",
)
parser.add_argument(
    "--curriculum-max-stage",
    type=int,
    default=None,
    help="Optional maximum capability stage; setting either bound bypasses curriculum sampling.",
)
parser.add_argument(
    "--finetune",
    action="store_true",
    default=False,
    help="Load model weights from --checkpoint but start a new optimizer and iteration count.",
)
parser.add_argument(
    "--project-config-dir",
    "--project_config_dir",
    dest="project_config_dir",
    type=str,
    default=None,
    help="Directory containing observation/action/reward/terrain/sensor/training YAML files.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
original_argv = sys.argv.copy()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.1.2"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) != version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import go2w_terrain_planner.tasks  # noqa: F401
from go2w_terrain_planner.models import Go2wActorCritic
from go2w_terrain_planner.utils.config_loader import (
    apply_project_config,
    apply_curriculum_stage_sampling_range,
    default_config_directory,
    load_project_config,
)
from go2w_terrain_planner.utils.logging_utils import (
    mirror_latest_checkpoint,
    prepare_run_directory,
    validate_checkpoint,
    validate_module_parameters,
    validate_ppo_loss_dict,
)

# RSL-RL 3.1.2 resolves policy class names in on_policy_runner.py globals.
# Isaac Lab 2.3.2 pins that release, so register the external model explicitly.
rsl_on_policy_runner.Go2wActorCritic = Go2wActorCritic

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    project_config = load_project_config(args_cli.project_config_dir)
    apply_project_config(env_cfg, agent_cfg, project_config, args_cli.project_config_dir)
    apply_curriculum_stage_sampling_range(
        env_cfg,
        args_cli.curriculum_min_stage,
        args_cli.curriculum_max_stage,
    )
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if agent_cfg.resume and args_cli.finetune:
        raise ValueError("--resume与--finetune不能同时使用")
    if args_cli.finetune and args_cli.checkpoint is None:
        raise ValueError("--finetune必须显式提供--checkpoint")
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    data_root = os.environ.get("GO2W_DATA_ROOT")
    runs_root = (
        os.path.join(data_root, "runs")
        if data_root
        else project_config["training"]["output_root"]
    )
    log_root_path = os.path.join(runs_root, "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    load_requested = bool(
        agent_cfg.resume
        or args_cli.finetune
        or agent_cfg.algorithm.class_name == "Distillation"
    )
    if load_requested:
        checkpoint_value = str(agent_cfg.load_checkpoint)
        checkpoint_path = Path(checkpoint_value).expanduser()
        if checkpoint_path.is_absolute():
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")
            resume_path = str(checkpoint_path)
        else:
            resume_path = get_checkpoint_path(
                log_root_path,
                agent_cfg.load_run,
                agent_cfg.load_checkpoint,
            )
        load_optimizer = bool(
            agent_cfg.resume
            and not args_cli.finetune
            and agent_cfg.algorithm.class_name != "Distillation"
        )
        validate_checkpoint(
            resume_path,
            include_optimizer=load_optimizer,
            required_action_std_parameterization_version=(
                2 if load_optimizer else None
            ),
            required_policy_architecture_version=6,
        )

    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    prepare_run_directory(
        log_dir,
        args_cli.project_config_dir or default_config_directory(),
        seed=agent_cfg.seed,
        command_line=original_argv,
    )

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    task_environment = env.unwrapped
    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if load_requested:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        checkpoint_infos = runner.load(resume_path, load_optimizer=load_optimizer)
        if args_cli.finetune:
            runner.current_learning_iteration = 0
        elif agent_cfg.resume and hasattr(task_environment, "curriculum"):
            curriculum_state = (
                checkpoint_infos.get("capability_curriculum")
                if isinstance(checkpoint_infos, dict)
                else None
            )
            if curriculum_state is None:
                raise RuntimeError(
                    "严格--resume要求checkpoint包含当前版本的课程状态"
                )
            task_environment.curriculum.load_state_dict(curriculum_state)
            # 环境在runner构造时已经按初始课程生成过任务；恢复课程后立即
            # 全量重置，避免续训的第一批rollout仍停留在初始难度。
            env.reset()
            print(
                "[INFO] Restored capability curriculum: "
                f"mean_level={task_environment.curriculum.levels.float().mean().item():.2f}",
                flush=True,
            )
        validate_module_parameters(runner.alg.policy, label="载入后的policy")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    # Checkpoint不包含并行环境的完整物理状态。随机化初始episode相位可避免
    # 续训后所有环境同步超时，因而从头训练、续训和微调均保持启用。
    init_at_random_ep_len = True

    print(
        f"[INFO] init_at_random_ep_len={init_at_random_ep_len}",
        flush=True,
    )

    original_save = runner.save

    def checked_save(path, infos=None):
        validate_module_parameters(runner.alg.policy, label="待保存policy")
        checkpoint_infos = {} if infos is None else dict(infos)
        if hasattr(task_environment, "curriculum"):
            checkpoint_infos["capability_curriculum"] = (
                task_environment.curriculum.state_dict()
            )
        return original_save(path, infos=checkpoint_infos)

    runner.save = checked_save
    original_update = runner.alg.update

    def checked_update(*args, **kwargs):
        losses = original_update(*args, **kwargs)
        validate_ppo_loss_dict(losses)
        losses["diagnostic_learning_rate"] = float(runner.alg.learning_rate)
        losses["diagnostic_action_mean_abs"] = float(
            runner.alg.policy.action_mean.detach().abs().mean().item()
        )
        losses["diagnostic_action_std"] = float(
            runner.alg.policy.action_std.detach().mean().item()
        )
        return losses

    runner.alg.update = checked_update

    # 保存基于训练回合成功率EMA的候选最佳模型。它不能替代独立的确定性
    # evaluate，但能防止后期策略退化时只留下最新的坏checkpoint。
    original_log = runner.log
    success_ema = None
    best_curriculum_progress = float("-inf")
    best_frontier_success = float("-inf")
    completed_log_batches = 0
    last_best_save_iteration = -50

    def monitored_log(locs, *args, **kwargs):
        nonlocal success_ema
        nonlocal best_curriculum_progress
        nonlocal best_frontier_success
        nonlocal completed_log_batches
        nonlocal last_best_save_iteration

        success_values = []
        for episode_info in locs.get("ep_infos", []):
            if "Episode/success_rate" in episode_info:
                value = torch.as_tensor(
                    episode_info["Episode/success_rate"],
                    dtype=torch.float32,
                )
                success_values.append(float(value.mean().item()))
        if success_values:
            batch_success = sum(success_values) / len(success_values)
            success_ema = (
                batch_success
                if success_ema is None
                else 0.9 * success_ema + 0.1 * batch_success
            )
            completed_log_batches += 1
            locs["loss_dict"]["diagnostic_success_ema"] = success_ema

        result = original_log(locs, *args, **kwargs)

        iteration = int(locs["it"])
        if success_ema is not None and completed_log_batches >= 20:
            curriculum = task_environment.curriculum
            frontier_difficulty = curriculum.frontier_difficulty()
            mean_level = float(curriculum.levels.float().mean().item())
            mean_frontier_difficulty = float(
                frontier_difficulty.mean().item()
            )
            frontier_success = float(curriculum.success_rate.mean().item())
            # ``level + intra-level difficulty``在升级前后近似连续：例如
            # 5+1与6+0代表相同课程进度。优先保存更高课程进度，只有在
            # 进度相近时才比较前沿成功率，避免低等级高成功率覆盖高等级模型。
            curriculum_progress = mean_level + mean_frontier_difficulty
            enough_interval = iteration - last_best_save_iteration >= 50
            progress_improved = (
                curriculum_progress > best_curriculum_progress + 0.01
            )
            same_progress_better = (
                curriculum_progress >= best_curriculum_progress - 0.01
                and frontier_success > best_frontier_success + 0.02
            )
            if (
                (progress_improved or same_progress_better)
                and enough_interval
            ):
                best_curriculum_progress = curriculum_progress
                best_frontier_success = frontier_success
                last_best_save_iteration = iteration
                runner.save(
                    os.path.join(log_dir, "model_best.pt"),
                    infos={
                        "best_training_success_ema": success_ema,
                        "best_training_mean_level": mean_level,
                        "best_training_frontier_success_ema": frontier_success,
                        "best_training_frontier_difficulty": (
                            mean_frontier_difficulty
                        ),
                        "best_training_curriculum_progress": curriculum_progress,
                        "best_training_iteration": iteration,
                    },
                )
                print(
                    "[INFO] Updated model_best.pt: "
                    f"iteration={iteration}, "
                    f"curriculum_progress={curriculum_progress:.3f}, "
                    f"frontier_success={frontier_success:.3f}, "
                    f"success_ema={success_ema:.3f}",
                    flush=True,
                )
        return result

    runner.log = monitored_log
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=init_at_random_ep_len,
    )
    final_iteration = getattr(runner, "current_learning_iteration", agent_cfg.max_iterations)
    runner.save(os.path.join(log_dir, f"model_{final_iteration}.pt"))
    latest_checkpoint = mirror_latest_checkpoint(log_dir)
    print(f"[INFO] Latest checkpoint copied to: {latest_checkpoint}")

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
