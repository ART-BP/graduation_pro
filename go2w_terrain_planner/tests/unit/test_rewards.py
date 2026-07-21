import pytest

torch = pytest.importorskip("torch")

from go2w_terrain_planner.tasks.direct.terrain_navigation.rewards import RewardWeights, navigation_reward
from go2w_terrain_planner.tasks.direct.terrain_navigation.terminations import termination_flags


def test_progress_is_better_and_collision_is_penalized() -> None:
    zeros = torch.zeros(2)
    flags = torch.zeros(2, dtype=torch.bool)
    reward = navigation_reward(
        torch.tensor([2.0, 2.0]),
        torch.tensor([1.5, 2.0]),
        torch.zeros((2, 2)),
        torch.zeros((2, 2)),
        zeros,
        zeros,
        zeros,
        flags,
        torch.tensor([False, True]),
        flags,
        flags,
        RewardWeights(),
    )
    assert reward[0] > reward[1]
    assert torch.isfinite(reward).all()


def test_termination_reached() -> None:
    terminated, reached, *_ = termination_flags(
        torch.tensor([0.2, 1.0]),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2),
        torch.zeros((2, 2)),
        torch.zeros(2, dtype=torch.long),
        goal_tolerance_m=0.35,
        stuck_timeout_s=2.0,
        maximum_distance_m=8.0,
        maximum_bad_observation_steps=3,
    )
    assert reached.tolist() == [True, False]
    assert terminated.tolist() == [True, False]


def test_action_limit_violation_is_penalized() -> None:
    zeros = torch.zeros(2)
    flags = torch.zeros(2, dtype=torch.bool)
    reward = navigation_reward(
        zeros,
        zeros,
        torch.zeros((2, 2)),
        torch.zeros((2, 2)),
        zeros,
        torch.tensor([0.0, 2.0]),
        zeros,
        flags,
        flags,
        flags,
        flags,
        RewardWeights(),
    )
    assert reward[1] < reward[0]
