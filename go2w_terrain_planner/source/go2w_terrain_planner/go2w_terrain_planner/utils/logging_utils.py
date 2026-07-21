"""Create reproducible run metadata inside the mounted data directory."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
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
