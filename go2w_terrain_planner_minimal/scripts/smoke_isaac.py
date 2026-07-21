from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Isaac Lab headless GPU smoke test"
)
parser.add_argument(
    "--steps",
    type=int,
    default=100,
    help="物理仿真步数",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

from isaaclab.sim import SimulationCfg, SimulationContext


def main() -> None:
    if args_cli.steps <= 0:
        raise ValueError("--steps必须大于0")

    if not torch.cuda.is_available():
        raise RuntimeError("当前容器无法访问CUDA GPU")

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    sim_cfg = SimulationCfg(
        dt=0.01,
        device="cuda:0",
    )
    sim = SimulationContext(sim_cfg)
    sim.reset()

    for _ in range(args_cli.steps):
        sim.step()

    output_dir = Path("/workspace/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    result_file = output_dir / "smoke_test_passed.txt"
    result_file.write_text(
        f"Isaac Lab smoke test passed: {args_cli.steps} steps\n",
        encoding="utf-8",
    )

    print(f"Smoke test passed: {args_cli.steps} steps")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
