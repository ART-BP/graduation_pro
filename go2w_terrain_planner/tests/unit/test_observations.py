import pytest

torch = pytest.importorskip("torch")

from go2w_terrain_planner.tasks.direct.terrain_navigation.events import sample_local_goals
from go2w_terrain_planner.tasks.direct.terrain_navigation.observations import (
    assemble_policy_observation,
    build_compact_map_observation,
)


def test_policy_observation_shape_and_goal_sampling() -> None:
    batch, history, size = 2, 5, 12
    maps = torch.zeros((batch, 7, size, size))
    poses = torch.zeros((batch, 3))
    goals_world = sample_local_goals(poses, 1.5, 4.0)
    distance = torch.linalg.vector_norm(goals_world, dim=-1)
    assert torch.all(distance >= 1.5) and torch.all(distance <= 4.0)
    goal = torch.zeros((batch, 3))
    current_velocity = torch.zeros((batch, 2))
    commands = torch.zeros((batch, history - 1, 2))
    motion = torch.zeros((batch, history - 1, 3))
    expected = 7 * size * size + 3 + 2 + (history - 1) * 2 + (history - 1) * 3
    observation = assemble_policy_observation(
        maps,
        goal,
        current_velocity,
        commands,
        motion,
        velocity_scale=(1.2, 1.0),
        map_extent_m=10.0,
        expected_dimension=expected,
    )
    assert observation.shape == (batch, expected)
    assert torch.isfinite(observation).all()


def test_policy_observation_rejects_nan() -> None:
    maps = torch.zeros((1, 7, 4, 4))
    maps[0, 0, 0, 0] = torch.nan
    with pytest.raises(RuntimeError):
        assemble_policy_observation(
            maps,
            torch.zeros((1, 3)),
            torch.zeros((1, 2)),
            torch.zeros((1, 2, 2)),
            torch.zeros((1, 2, 3)),
            velocity_scale=(1.2, 1.0),
            map_extent_m=10.0,
        )


def test_policy_auxiliary_inputs_are_physically_normalized() -> None:
    maps = torch.zeros((1, 7, 2, 2))
    goal = torch.tensor([[0.5, 0.0, 1.0]])
    velocity = torch.tensor([[1.2, -1.0]])
    commands = torch.tensor([[[0.6, 0.5], [-0.2, -1.0]]])
    motion = torch.tensor([[[5.0, -2.5, torch.pi], [1.0, 0.0, -0.5 * torch.pi]]])

    observation = assemble_policy_observation(
        maps,
        goal,
        velocity,
        commands,
        motion,
        velocity_scale=(1.2, 1.0),
        map_extent_m=10.0,
    )

    auxiliary = observation[:, 7 * 2 * 2 :]
    assert auxiliary[0, :5].tolist() == pytest.approx([0.5, 0.0, 1.0, 1.0, -1.0])
    assert auxiliary[0, 5:9].tolist() == pytest.approx([0.5, 0.5, -1.0 / 6.0, -1.0])
    assert auxiliary[0, 9:].tolist() == pytest.approx(
        [1.0, -0.5, 1.0, 0.2, 0.0, -0.5]
    )


def test_compact_map_encodes_change_age_and_confidence() -> None:
    current = torch.zeros((1, 4, 2, 2))
    previous = torch.zeros_like(current)
    current[:, 2:] = 1.0
    previous[:, 2:] = 1.0
    current[0, 0, 0, 0] = 0.8
    previous[0, 0, 0, 0] = 0.2
    history = torch.zeros((1, 4, 4, 2, 2))
    history[0, -1, 2, 0, 0] = 1.0
    history[0, 1, 2, 0, 1] = 1.0

    compact = build_compact_map_observation(current, previous, history)

    assert compact.shape == (1, 7, 2, 2)
    assert compact[0, 4, 0, 0].item() == pytest.approx(0.3)
    assert compact[0, 5, 0, 0].item() == pytest.approx(0.0)
    assert compact[0, 5, 0, 1].item() == pytest.approx(2.0 / 3.0)
    assert compact[0, 5, 1, 1].item() == pytest.approx(1.0)
    assert compact[0, 6, 0, 0].item() > compact[0, 6, 0, 1].item()
