"""Export the deterministic actor with stable deployment tensor names."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from go2w_terrain_planner.models import ActorExportWrapper, Go2wActorCritic
from go2w_terrain_planner.utils.config_loader import load_project_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="/workspace/data/export/planner.onnx")
    parser.add_argument("--map-size", type=int, default=None)
    parser.add_argument("--project-config-dir", "--project_config_dir", dest="project_config_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.project_config_dir)
    map_config = config["map"]
    history_config = config["history"]
    map_size = args.map_size if args.map_size is not None else int(map_config["output_size"])
    history = int(map_config["history_length"])
    policy_dim = (
        history * int(map_config["channels"]) * map_size * map_size
        + 3
        + 2
        + int(history_config["command_length"]) * 2
        + int(history_config["motion_length"]) * 3
    )
    observations = {
        "policy": torch.zeros((1, policy_dim), dtype=torch.float32),
        "critic": torch.zeros((1, 13), dtype=torch.float32),
    }
    model = Go2wActorCritic(
        observations,
        {"policy": ["policy"], "critic": ["critic"]},
        2,
        map_history_length=history,
        map_channels=int(map_config["channels"]),
        map_size=map_size,
        map_feature_dim=int(config["model"]["map_feature_dim"]),
        temporal_hidden_dim=int(config["model"]["temporal_hidden_dim"]),
        auxiliary_hidden_dim=int(config["model"]["auxiliary_hidden_dim"]),
        fusion_hidden_dim=int(config["model"]["fusion_hidden_dim"]),
        critic_hidden_dims=list(config["model"]["critic_hidden_dims"]),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("model"))
    if state is None:
        raise KeyError("checkpoint中不存在model_state_dict或model")
    model.load_state_dict(state)
    action_config = config["action"]
    wrapper = ActorExportWrapper(
        model,
        (float(action_config["linear_min_mps"]), float(action_config["angular_min_radps"])),
        (float(action_config["linear_max_mps"]), float(action_config["angular_max_radps"])),
    ).eval()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, policy_dim), dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy,
        output,
        input_names=["policy_observation"],
        output_names=["velocity_command"],
        dynamic_axes={"policy_observation": {0: "batch"}, "velocity_command": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX exported to {output}")


if __name__ == "__main__":
    main()
