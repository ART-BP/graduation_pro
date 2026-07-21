import pytest

torch = pytest.importorskip("torch")

from go2w_terrain_planner.tasks.direct.terrain_navigation.events import sample_local_goals
from go2w_terrain_planner.tasks.direct.terrain_navigation.observations import assemble_policy_observation


def test_policy_observation_shape_and_goal_sampling() -> None:
    batch, history, size = 2, 5, 12
    maps = torch.zeros((batch, history, 4, size, size))
    poses = torch.zeros((batch, 3))
    goals_world = sample_local_goals(poses, 1.5, 4.0)
    distance = torch.linalg.vector_norm(goals_world, dim=-1)
    assert torch.all(distance >= 1.5) and torch.all(distance <= 4.0)
    goal = torch.zeros((batch, 3))
    current_velocity = torch.zeros((batch, 2))
    commands = torch.zeros((batch, history - 1, 2))
    motion = torch.zeros((batch, history - 1, 3))
    expected = history * 4 * size * size + 3 + 2 + (history - 1) * 2 + (history - 1) * 3
    observation = assemble_policy_observation(
        maps, goal, current_velocity, commands, motion, expected_dimension=expected
    )
    assert observation.shape == (batch, expected)
    assert torch.isfinite(observation).all()


def test_policy_observation_rejects_nan() -> None:
    maps = torch.zeros((1, 3, 4, 4, 4))
    maps[0, 0, 0, 0, 0] = torch.nan
    with pytest.raises(RuntimeError):
        assemble_policy_observation(
            maps,
            torch.zeros((1, 3)),
            torch.zeros((1, 2)),
            torch.zeros((1, 2, 2)),
            torch.zeros((1, 2, 3)),
        )
