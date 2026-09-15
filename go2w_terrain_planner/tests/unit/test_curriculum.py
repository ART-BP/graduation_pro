import pytest

torch = pytest.importorskip("torch")

from go2w_terrain_planner.tasks.direct.terrain_navigation.curriculum import (
    TerrainCurriculum,
    goal_distance_maximum_for_levels,
)


def test_curriculum_requires_minimum_episode_history() -> None:
    curriculum = TerrainCurriculum(
        1,
        "cpu",
        initial_level=1,
        maximum_level=3,
        success_rate_up=0.8,
        success_rate_down=0.45,
        minimum_episodes_per_level=3,
        smoothing=1.0,
        allow_level_demotion=True,
    )
    curriculum.update(torch.tensor([0]), torch.tensor([False]))
    assert curriculum.levels.item() == 1
    curriculum.update(torch.tensor([0]), torch.tensor([False]))
    assert curriculum.levels.item() == 1
    curriculum.update(torch.tensor([0]), torch.tensor([False]))
    assert curriculum.levels.item() == 0


def test_curriculum_state_survives_resume_and_environment_resize() -> None:
    curriculum = TerrainCurriculum(
        2,
        "cpu",
        initial_level=2,
        maximum_level=9,
        success_rate_up=0.8,
        success_rate_down=0.45,
    )
    curriculum.levels[:] = torch.tensor([4, 7])
    curriculum.goal_levels[:] = torch.tensor([3, 6])
    curriculum.success_rate[:] = torch.tensor([0.65, 0.75])
    curriculum.episode_count[:] = torch.tensor([3, 5])
    curriculum.full_difficulty_success_rate[:] = torch.tensor([0.55, 0.70])
    curriculum.full_difficulty_episode_count[:] = torch.tensor([1, 4])

    restored = TerrainCurriculum(
        4,
        "cpu",
        initial_level=2,
        maximum_level=9,
        success_rate_up=0.8,
        success_rate_down=0.45,
    )
    restored.load_state_dict(curriculum.state_dict())

    assert restored.levels.tolist() == [4, 7, 4, 7]
    assert restored.goal_levels.tolist() == [3, 6, 3, 6]
    assert restored.success_rate.tolist() == pytest.approx(
        [0.65, 0.75, 0.65, 0.75]
    )
    assert restored.episode_count.tolist() == [3, 5, 3, 5]
    assert restored.full_difficulty_success_rate.tolist() == pytest.approx(
        [0.55, 0.70, 0.55, 0.70]
    )
    assert restored.full_difficulty_episode_count.tolist() == [1, 4, 1, 4]


def test_curriculum_cannot_demote_below_configured_floor() -> None:
    curriculum = TerrainCurriculum(
        1,
        "cpu",
        initial_level=2,
        minimum_level=2,
        maximum_level=9,
        success_rate_up=0.65,
        success_rate_down=0.30,
        minimum_episodes_per_level=1,
        smoothing=1.0,
        allow_level_demotion=True,
    )

    curriculum.update(torch.tensor([0]), torch.tensor([False]))

    assert curriculum.levels.item() == 2


def test_curriculum_does_not_regress_when_demotion_is_disabled() -> None:
    curriculum = TerrainCurriculum(
        1,
        "cpu",
        initial_level=4,
        minimum_level=2,
        maximum_level=9,
        success_rate_up=0.8,
        success_rate_down=0.2,
        minimum_episodes_per_level=3,
        smoothing=1.0,
        allow_level_demotion=False,
    )

    for _ in range(10):
        curriculum.update(torch.tensor([0]), torch.tensor([False]))

    assert curriculum.levels.item() == 4
    assert curriculum.goal_levels.item() == 4
    assert curriculum.success_rate.item() == pytest.approx(0.0)


def test_goal_curriculum_advances_only_after_mastering_frontier() -> None:
    curriculum = TerrainCurriculum(
        1,
        "cpu",
        initial_level=2,
        minimum_level=2,
        maximum_level=9,
        success_rate_up=0.8,
        success_rate_down=0.2,
        minimum_episodes_per_level=1,
        smoothing=1.0,
    )

    curriculum.update(torch.tensor([0]), torch.tensor([True]))
    assert curriculum.levels.item() == 3
    assert curriculum.goal_levels.item() == 2

    curriculum.update(torch.tensor([0]), torch.tensor([True]))
    assert curriculum.levels.item() == 4
    assert curriculum.goal_levels.item() == 3


def test_legacy_curriculum_state_restores_with_safe_goal_level() -> None:
    curriculum = TerrainCurriculum(
        2,
        "cpu",
        initial_level=2,
        minimum_level=2,
        maximum_level=9,
        success_rate_up=0.8,
        success_rate_down=0.2,
    )
    curriculum.load_state_dict(
        {
            "version": 1,
            "minimum_level": 2,
            "maximum_level": 9,
            "levels": torch.tensor([2, 5]),
            "success_rate": torch.tensor([0.5, 0.1]),
            "episode_count": torch.tensor([0, 30]),
        }
    )

    assert curriculum.levels.tolist() == [2, 5]
    assert curriculum.goal_levels.tolist() == [2, 4]


def test_goal_distance_ceiling_grows_with_curriculum_level() -> None:
    maximum = goal_distance_maximum_for_levels(
        torch.tensor([2, 5, 9]),
        initial_level=2,
        maximum_level=9,
        start_maximum_m=4.0,
        final_maximum_m=10.0,
    )

    assert maximum.tolist() == pytest.approx([4.0, 46.0 / 7.0, 10.0])


def test_frontier_geometry_difficulty_requires_history_and_success() -> None:
    curriculum = TerrainCurriculum(
        1,
        "cpu",
        initial_level=5,
        minimum_level=2,
        maximum_level=9,
        success_rate_up=0.8,
        success_rate_down=0.2,
        minimum_episodes_per_level=30,
    )

    assert curriculum.frontier_difficulty().item() == pytest.approx(0.0)
    curriculum.episode_count[:] = 15
    curriculum.success_rate[:] = 0.8
    assert curriculum.frontier_difficulty().item() == pytest.approx(0.5)
    curriculum.episode_count[:] = 30
    assert curriculum.frontier_difficulty().item() == pytest.approx(1.0)


def test_easy_progressive_episodes_cannot_trigger_promotion() -> None:
    curriculum = TerrainCurriculum(
        1,
        "cpu",
        initial_level=5,
        minimum_level=2,
        maximum_level=9,
        success_rate_up=0.8,
        success_rate_down=0.2,
        minimum_episodes_per_level=3,
        minimum_full_difficulty_episodes=2,
        full_difficulty_threshold=0.9,
        smoothing=1.0,
    )

    for _ in range(5):
        curriculum.update(
            torch.tensor([0]),
            torch.tensor([True]),
            torch.tensor([0.8]),
        )
    assert curriculum.levels.item() == 5
    assert curriculum.full_difficulty_episode_count.item() == 0

    curriculum.update(
        torch.tensor([0]),
        torch.tensor([True]),
        torch.tensor([0.95]),
    )
    assert curriculum.levels.item() == 5

    curriculum.update(
        torch.tensor([0]),
        torch.tensor([True]),
        torch.tensor([1.0]),
    )
    assert curriculum.levels.item() == 6
    assert curriculum.goal_levels.item() == 5
