import json
from pathlib import Path

import pytest

from go2w_terrain_planner.utils.logging_utils import (
    mirror_latest_checkpoint,
    prepare_run_directory,
    validate_checkpoint,
    validate_module_parameters,
    validate_ppo_loss_dict,
)


def test_run_metadata_and_checkpoint_mirror(tmp_path: Path) -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    run_dir = prepare_run_directory(tmp_path / "run", config_dir, seed=7, command_line=["train"])
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["seed"] == 7
    assert len(list((run_dir / "configs").glob("*.yaml"))) == 6
    first = run_dir / "model_1.pt"
    second = run_dir / "model_2.pt"
    first.write_bytes(b"old")
    second.write_bytes(b"new")
    latest = mirror_latest_checkpoint(run_dir)
    assert latest.read_bytes() == b"new"


def test_checkpoint_finite_value_guards(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    valid = tmp_path / "valid.pt"
    invalid = tmp_path / "invalid.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {"moment": torch.zeros(1)},
        },
        valid,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {"moment": torch.tensor([torch.nan])},
        },
        invalid,
    )

    validate_checkpoint(valid, include_optimizer=True)
    validate_module_parameters(model)
    with pytest.raises(RuntimeError, match="NaN或Inf"):
        validate_checkpoint(invalid, include_optimizer=True)
    # 仅权重微调可以忽略旧优化器，但仍会检查模型本身。
    validate_checkpoint(invalid, include_optimizer=False)


def test_ppo_loss_validation_rejects_probability_ratio_explosion() -> None:
    validate_ppo_loss_dict(
        {"value_function": 5.0, "surrogate": -0.01, "entropy": -3.0}
    )

    with pytest.raises(RuntimeError, match="surrogate loss异常"):
        validate_ppo_loss_dict(
            {"value_function": 5.0, "surrogate": 55481.0, "entropy": -11.0}
        )


def test_strict_resume_rejects_legacy_action_std_checkpoint(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": {"std": torch.tensor([1.0, 1.0])},
            "optimizer_state_dict": {},
        },
        checkpoint,
    )

    with pytest.raises(RuntimeError, match="不能严格--resume"):
        validate_checkpoint(
            checkpoint,
            include_optimizer=True,
            required_action_std_parameterization_version=2,
        )


def test_checkpoint_rejects_incompatible_policy_architecture(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "old_architecture.pt"
    torch.save(
        {
            "model_state_dict": {
                "_action_std_parameterization_version": torch.tensor(2),
                "std": torch.zeros(2),
            },
            "optimizer_state_dict": {},
        },
        checkpoint,
    )

    with pytest.raises(RuntimeError, match="网络结构"):
        validate_checkpoint(
            checkpoint,
            include_optimizer=False,
            required_policy_architecture_version=2,
        )
