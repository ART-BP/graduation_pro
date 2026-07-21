import pytest

torch = pytest.importorskip("torch")

from go2w_terrain_planner.robots.velocity_command_adapter import (
    ExecutionModelConfig,
    VelocityCommandAdapter,
    VelocityExecutionModel,
)
from go2w_terrain_planner.tasks.direct.terrain_navigation.terminations import (
    terrain_failure_state,
)


def test_action_mapping_and_acceleration_limit() -> None:
    adapter = VelocityCommandAdapter()
    physical = adapter.to_physical(torch.tensor([[-1.0, -1.0], [1.0, 1.0]]))
    assert physical[0].tolist() == pytest.approx([-0.2, -1.0])
    assert physical[1].tolist() == pytest.approx([0.8, 1.0])
    model = VelocityExecutionModel(2, "cpu", ExecutionModelConfig(tracking_noise_std=0.0))
    velocity = model.step(physical, torch.zeros(2), torch.ones(2), dt=0.1)
    assert torch.abs(velocity[:, 0]).max().item() <= 0.080001
    assert torch.isfinite(velocity).all()


def test_terrain_speed_loss_is_bounded() -> None:
    model = VelocityExecutionModel(
        1,
        "cpu",
        ExecutionModelConfig(
            maximum_terrain_speed_loss=0.25,
            entry_angle_speed_penalty=0.0,
            tracking_noise_std=0.0,
        ),
    )
    command = torch.tensor([[0.8, 0.0]])
    for _ in range(200):
        model.step(command, torch.ones(1), torch.ones(1), dt=0.02)
    assert model.actual_velocity[0, 0].item() == pytest.approx(0.6, abs=1.0e-4)


def test_terrain_failure_is_monotonic_with_hazard() -> None:
    risk, tilt, collision, fallen, unstable = terrain_failure_state(
        torch.zeros(2),
        torch.tensor([0.2, 0.8]) * 0.45,
        torch.ones(2),
        torch.full((2,), 0.8),
        torch.full((2,), 0.6),
        collision_height_threshold_m=0.45,
        maximum_command_speed_mps=0.8,
        unstable_risk_threshold=0.2,
        terrain_tilt_gain_rad=1.0,
        entry_tilt_gain=0.5,
        maximum_tilt_rad=0.65,
        fall_tilt_rad=1.0,
    )
    assert risk.tolist() == pytest.approx([0.2, 0.8])
    assert tilt[1] > tilt[0]
    assert collision.tolist() == [False, False]
    assert fallen.tolist() == [False, False]
    assert unstable.tolist() == [False, True]
