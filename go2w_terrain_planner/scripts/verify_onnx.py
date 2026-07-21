"""Check exported ONNX names, shapes, and finite outputs."""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort

from go2w_terrain_planner.utils.config_loader import load_project_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/data/export/planner.onnx")
    parser.add_argument("--map-size", type=int, default=None)
    parser.add_argument("--project-config-dir", "--project_config_dir", dest="project_config_dir", default=None)
    args = parser.parse_args()
    config = load_project_config(args.project_config_dir)
    map_config = config["map"]
    history_config = config["history"]
    map_size = args.map_size if args.map_size is not None else int(map_config["output_size"])
    policy_dim = (
        int(map_config["history_length"])
        * int(map_config["channels"])
        * map_size
        * map_size
        + 3
        + int(history_config["command_length"]) * 2
        + int(history_config["motion_length"]) * 3
    )
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if [item.name for item in inputs] != ["policy_observation"]:
        raise RuntimeError(f"ONNX输入名称错误：{[item.name for item in inputs]}")
    if [item.name for item in outputs] != ["velocity_command"]:
        raise RuntimeError(f"ONNX输出名称错误：{[item.name for item in outputs]}")
    observation = np.zeros((2, policy_dim), dtype=np.float32)
    action = session.run(None, {"policy_observation": observation})[0]
    if action.shape != (2, 2) or not np.isfinite(action).all():
        raise RuntimeError(f"ONNX动作输出异常：shape={action.shape}")
    action_config = config["action"]
    lower = np.array(
        [action_config["linear_min_mps"], action_config["angular_min_radps"]], dtype=np.float32
    )
    upper = np.array(
        [action_config["linear_max_mps"], action_config["angular_max_radps"]], dtype=np.float32
    )
    if np.any(action < lower - 1.0e-4) or np.any(action > upper + 1.0e-4):
        raise RuntimeError("ONNX速度指令超出配置范围")
    print("ONNX verification passed")
