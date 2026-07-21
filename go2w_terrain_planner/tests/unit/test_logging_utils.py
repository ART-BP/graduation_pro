import json
from pathlib import Path

from go2w_terrain_planner.utils.logging_utils import mirror_latest_checkpoint, prepare_run_directory


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
