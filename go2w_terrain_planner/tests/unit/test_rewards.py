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
        zeros,
        torch.zeros((2, 2)),
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
        zeros,
        torch.zeros((2, 2)),
        RewardWeights(),
    )
    assert reward[1] < reward[0]


def test_stationary_heading_cannot_outweigh_time_penalty() -> None:
    zeros = torch.zeros(2)
    flags = torch.zeros(2, dtype=torch.bool)
    reward = navigation_reward(
        zeros,
        zeros,
        torch.zeros((2, 2)),
        torch.zeros((2, 2)),
        zeros,
        zeros,
        zeros,
        flags,
        flags,
        flags,
        flags,
        torch.tensor([0.0, torch.pi]),
        torch.zeros((2, 2)),
        RewardWeights(),
    )
    assert torch.all(reward < 0.0)


def test_short_regression_is_penalized_less_than_equal_progress_is_rewarded() -> None:
    zeros = torch.zeros(2)
    flags = torch.zeros(2, dtype=torch.bool)
    _, terms = navigation_reward(
        torch.tensor([2.0, 2.0]),
        torch.tensor([1.9, 2.1]),
        torch.zeros((2, 2)),
        torch.zeros((2, 2)),
        zeros,
        zeros,
        zeros,
        flags,
        flags,
        flags,
        flags,
        zeros,
        torch.zeros((2, 2)),
        RewardWeights(),
        return_terms=True,
    )

    assert terms["progress"][0].item() == pytest.approx(1.0)
    assert terms["progress"][1].item() == pytest.approx(0.0)
    assert terms["regression"][0].item() == pytest.approx(0.0)
    assert terms["regression"][1].item() == pytest.approx(-0.1)


def test_progress_cannot_be_collected_repeatedly_by_oscillation() -> None:
    zeros = torch.zeros(3)
    flags = torch.zeros(3, dtype=torch.bool)
    _, terms = navigation_reward(
        torch.tensor([3.0, 2.0, 3.0]),
        torch.tensor([2.0, 3.0, 2.0]),
        torch.zeros((3, 2)),
        torch.zeros((3, 2)),
        zeros,
        zeros,
        zeros,
        flags,
        flags,
        flags,
        flags,
        zeros,
        torch.zeros((3, 2)),
        RewardWeights(),
        best_distance=torch.tensor([3.0, 2.0, 2.0]),
        return_terms=True,
    )

    assert terms["progress"].tolist() == pytest.approx([10.0, 0.0, 0.0])
    assert terms["regression"].tolist() == pytest.approx([0.0, -1.0, 0.0])


def test_timeout_and_failure_terminations_are_penalized() -> None:
    zeros = torch.zeros(3)
    flags = torch.zeros(3, dtype=torch.bool)
    reward = navigation_reward(
        zeros,
        zeros,
        torch.zeros((3, 2)),
        torch.zeros((3, 2)),
        zeros,
        zeros,
        zeros,
        flags,
        flags,
        flags,
        flags,
        zeros,
        torch.zeros((3, 2)),
        RewardWeights(),
        time_out=torch.tensor([True, False, False]),
        out_of_bounds=torch.tensor([False, True, False]),
        observation_failure=torch.tensor([False, False, True]),
    )
    assert torch.all(reward < RewardWeights().time)


def test_terrain_risk_penalizes_fast_motion_but_not_stopping() -> None:
    zeros = torch.zeros(2)
    flags = torch.zeros(2, dtype=torch.bool)
    _, terms = navigation_reward(
        zeros,
        zeros,
        torch.zeros((2, 2)),
        torch.zeros((2, 2)),
        zeros,
        zeros,
        zeros,
        flags,
        flags,
        flags,
        flags,
        zeros,
        torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        RewardWeights(),
        terrain_risk=torch.ones(2),
        return_terms=True,
    )

    assert terms["terrain_speed_risk"][0].item() == pytest.approx(0.0)
    assert terms["terrain_speed_risk"][1].item() == pytest.approx(-0.75)
