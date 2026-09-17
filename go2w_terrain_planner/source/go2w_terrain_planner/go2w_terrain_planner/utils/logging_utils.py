"""Create reproducible run metadata inside the mounted data directory."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path


def _package_version(name: str, fallback: str = "unknown") -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return fallback


def _git_state(project_root: Path) -> tuple[str, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(project_root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None


def prepare_run_directory(
    run_directory: str | Path,
    config_directory: str | Path,
    *,
    seed: int,
    command_line: list[str] | None = None,
) -> Path:
    """Create run folders, copy all YAML files, and write ``metadata.json``."""
    run_path = Path(run_directory)
    run_path.mkdir(parents=True, exist_ok=True)
    for child in ("configs", "checkpoints", "videos", "metrics"):
        (run_path / child).mkdir(exist_ok=True)
    config_path = Path(config_directory)
    for source in sorted(config_path.glob("*.yaml")):
        shutil.copy2(source, run_path / "configs" / source.name)

    project_root = Path(os.environ.get("GO2W_PROJECT_ROOT", Path(__file__).resolve().parents[4]))
    commit, dirty = _git_state(project_root)
    if commit == "unknown":
        commit = os.environ.get("GO2W_GIT_COMMIT", "unknown")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "git_commit": commit,
        "git_dirty": dirty,
        "docker_image_version": os.environ.get("GO2W_IMAGE_VERSION", "unknown"),
        "isaac_lab_version": _package_version(
            "isaaclab", os.environ.get("ISAAC_LAB_VERSION", "unknown")
        ),
        "isaac_sim_version": os.environ.get("ISAAC_SIM_VERSION", "5.1.0"),
        "rsl_rl_version": _package_version("rsl-rl-lib"),
        "command_line": command_line or [],
    }
    (run_path / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_path


def mirror_latest_checkpoint(run_directory: str | Path) -> Path:
    """Copy the newest RSL-RL checkpoint to a stable deployment path."""
    run_path = Path(run_directory)
    checkpoints = list(run_path.glob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"训练目录中没有RSL-RL checkpoint：{run_path}")
    latest = max(checkpoints, key=lambda path: (path.stat().st_mtime_ns, path.name))
    destination = run_path / "checkpoints" / "model_latest.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, destination)
    return destination


def _non_finite_tensor_paths(value, prefix: str = "checkpoint") -> list[str]:
    """Return paths of floating-point tensors containing NaN or Inf."""
    import torch

    bad: list[str] = []
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            bad.append(prefix)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            bad.extend(_non_finite_tensor_paths(child, f"{prefix}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            bad.extend(_non_finite_tensor_paths(child, f"{prefix}[{index}]"))
    return bad


def validate_checkpoint(
    checkpoint_path: str | Path,
    *,
    include_optimizer: bool,
    required_action_std_parameterization_version: int | None = None,
    required_policy_architecture_version: int | None = None,
) -> None:
    """Reject a corrupt checkpoint before it can contaminate a training run."""
    import torch

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint文件不存在: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    value = checkpoint
    model_state = None
    if not include_optimizer and isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "model", "policy_state_dict"):
            if key in checkpoint:
                value = checkpoint[key]
                model_state = value
                break
    elif isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "model", "policy_state_dict"):
            if key in checkpoint:
                model_state = checkpoint[key]
                break
    bad_paths = _non_finite_tensor_paths(value)
    if bad_paths:
        preview = ", ".join(bad_paths[:12])
        suffix = "" if len(bad_paths) <= 12 else f"（另有{len(bad_paths) - 12}项）"
        raise RuntimeError(f"Checkpoint包含NaN或Inf: {preview}{suffix}")
    if required_action_std_parameterization_version is not None:
        version_values = []
        if isinstance(model_state, Mapping):
            for key, tensor in model_state.items():
                if str(key).endswith("_action_std_parameterization_version"):
                    try:
                        version_values.append(int(tensor.item()))
                    except (AttributeError, TypeError, ValueError):
                        pass
        if version_values != [required_action_std_parameterization_version]:
            raise RuntimeError(
                "checkpoint动作分布参数化版本与当前训练接口不一致，"
                "不能严格--resume"
            )
    if required_policy_architecture_version is not None:
        version_values = []
        if isinstance(model_state, Mapping):
            for key, tensor in model_state.items():
                if str(key).endswith("_policy_architecture_version"):
                    try:
                        version_values.append(int(tensor.item()))
                    except (AttributeError, TypeError, ValueError):
                        pass
        if version_values != [required_policy_architecture_version]:
            raise RuntimeError(
                "checkpoint网络结构与当前地图编码器不兼容；"
                "请使用对应run目录中的配置评估，或从头训练新网络"
            )


def validate_module_parameters(module, *, label: str = "policy") -> None:
    """Validate model parameters and buffers immediately before persistence."""
    import torch

    bad: list[str] = []
    for name, tensor in module.state_dict().items():
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all().item()
        ):
            bad.append(name)
    if bad:
        raise RuntimeError(
            f"{label}包含NaN或Inf，拒绝保存checkpoint: {', '.join(bad[:12])}"
        )


def validate_ppo_loss_dict(
    losses: Mapping,
    *,
    maximum_abs_surrogate_loss: float = 10.0,
) -> None:
    """Abort a run before an invalid PPO update can be checkpointed."""
    invalid = {
        str(name): value
        for name, value in losses.items()
        if not isinstance(value, (int, float)) or not math.isfinite(float(value))
    }
    if invalid:
        raise RuntimeError(f"PPO损失出现NaN或Inf: {invalid}")
    surrogate = float(losses.get("surrogate", 0.0))
    if abs(surrogate) > maximum_abs_surrogate_loss:
        raise RuntimeError(
            "PPO surrogate loss异常增大，训练已中止以防止保存损坏模型: "
            f"surrogate={surrogate:.6g}, "
            f"threshold={maximum_abs_surrogate_loss:.6g}"
        )
