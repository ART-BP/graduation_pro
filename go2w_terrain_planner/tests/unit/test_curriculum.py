from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from go2w_terrain_planner.tasks.direct.terrain_navigation.curriculum import (
    CurriculumStageSchedule,
    CapabilityCurriculum,
)
from go2w_terrain_planner.mapping.terrain_truth_model import TERRAIN_NAMES
from go2w_terrain_planner.utils.config_loader import load_project_config


def test_curriculum_requires_minimum_episode_history() -> None:
    curriculum = CapabilityCurriculum(
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
    curriculum = CapabilityCurriculum(
        2,
        "cpu",
        initial_level=2,
        maximum_level=9,
        success_rate_up=0.8,
        success_rate_down=0.45,
    )
    curriculum.levels.fill_(4)
    curriculum.success_rate.fill_(0.65)
    curriculum.episode_count.fill_(5)
    curriculum.full_difficulty_success_rate.fill_(0.55)
    curriculum.full_difficulty_episode_count.fill_(4)

    restored = CapabilityCurriculum(
        4,
        "cpu",
        initial_level=2,
        maximum_level=9,
        success_rate_up=0.8,
        success_rate_down=0.45,
    )
    restored.load_state_dict(curriculum.state_dict())

    assert restored.levels.tolist() == [4, 4, 4, 4]
    assert restored.success_rate.tolist() == pytest.approx([0.65] * 4)
    assert restored.episode_count.tolist() == [5] * 4
    assert restored.full_difficulty_success_rate.tolist() == pytest.approx(
        [0.55] * 4
    )
    assert restored.full_difficulty_episode_count.tolist() == [4] * 4


def test_parallel_outcomes_update_one_shared_curriculum_state() -> None:
    curriculum = CapabilityCurriculum(
        4,
        "cpu",
        initial_level=2,
        minimum_level=2,
        maximum_level=9,
        success_rate_up=0.9,
        success_rate_down=0.2,
        minimum_episodes_per_level=10,
        smoothing=0.2,
    )

    curriculum.update(
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([True, False, True, True]),
    )

    # alpha_batch=1-(1-0.2)^4=0.5904, batch success=0.75.
    expected_rate = 0.5 + 0.5904 * (0.75 - 0.5)
    assert curriculum.success_rate.tolist() == pytest.approx(
        [expected_rate] * 4
    )
    assert curriculum.episode_count.tolist() == [4] * 4
    assert curriculum.levels.tolist() == [2] * 4


def test_global_frontier_promotes_all_workers_together() -> None:
    curriculum = CapabilityCurriculum(
        4,
        "cpu",
        initial_level=2,
        minimum_level=2,
        maximum_level=9,
        success_rate_up=0.9,
        success_rate_down=0.2,
        minimum_episodes_per_level=4,
        minimum_full_difficulty_episodes=4,
        smoothing=1.0,
    )

    curriculum.update(
        torch.arange(4),
        torch.ones(4, dtype=torch.bool),
        torch.ones(4),
    )

    assert curriculum.levels.tolist() == [3] * 4
    assert curriculum.episode_count.tolist() == [0] * 4
    assert curriculum.success_rate.tolist() == pytest.approx([0.5] * 4)


def test_curriculum_cannot_demote_below_configured_floor() -> None:
    curriculum = CapabilityCurriculum(
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
    curriculum = CapabilityCurriculum(
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
    assert curriculum.success_rate.item() == pytest.approx(0.0)


def test_capability_curriculum_advances_only_after_mastering_frontier() -> None:
    curriculum = CapabilityCurriculum(
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

    curriculum.update(torch.tensor([0]), torch.tensor([True]))
    assert curriculum.levels.item() == 4


def test_old_curriculum_state_is_rejected() -> None:
    curriculum = CapabilityCurriculum(
        2,
        "cpu",
        initial_level=2,
        minimum_level=2,
        maximum_level=9,
        success_rate_up=0.8,
        success_rate_down=0.2,
    )
    with pytest.raises(ValueError, match="课程checkpoint格式"):
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


def test_stage_schedule_uses_explicit_goal_ranges() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    config = load_project_config(config_dir)
    schedule = CurriculumStageSchedule(
        config["curriculum"]["stages"], TERRAIN_NAMES, "cpu"
    )
    batch = schedule.sample(
        torch.tensor([1, 5, 9]),
        torch.ones(3),
        frontier_probability=1.0,
        challenge_probability=0.0,
        challenge_stage_span=1,
    )

    assert batch.stages.tolist() == [1, 5, 9]
    assert batch.goal_minimum_m.tolist() == pytest.approx([1.0, 3.0, 5.0])
    assert batch.goal_maximum_m.tolist() == pytest.approx([2.5, 6.0, 10.0])
    assert batch.domain_randomization_scale.tolist() == pytest.approx(
        [0.0, 0.0, 1.0]
    )


def test_non_progressive_flat_stages_are_full_difficulty_immediately() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    config = load_project_config(config_dir)
    schedule = CurriculumStageSchedule(
        config["curriculum"]["stages"], TERRAIN_NAMES, "cpu"
    )
    batch = schedule.sample(
        torch.tensor([1, 2, 3]),
        torch.zeros(3),
        frontier_probability=1.0,
        challenge_probability=0.0,
        challenge_stage_span=1,
    )

    assert batch.stages.tolist() == [1, 2, 3]
    assert batch.curriculum_difficulty.tolist() == pytest.approx(
        [1.0, 1.0, 0.0]
    )
    assert batch.geometry_difficulty.tolist() == pytest.approx(
        [0.0, 0.0, 0.0]
    )


def test_stage_schedule_mixes_replay_frontier_and_future_stage() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    config = load_project_config(config_dir)
    schedule = CurriculumStageSchedule(
        config["curriculum"]["stages"], TERRAIN_NAMES, "cpu"
    )
    torch.manual_seed(17)
    count = 12000
    batch = schedule.sample(
        torch.full((count,), 5),
        torch.full((count,), 0.4),
        frontier_probability=0.60,
        challenge_probability=0.05,
        challenge_stage_span=1,
    )

    assert (batch.stages == 5).float().mean().item() == pytest.approx(0.60, abs=0.02)
    assert (batch.stages < 5).float().mean().item() == pytest.approx(0.35, abs=0.02)
    assert (batch.stages == 6).float().mean().item() == pytest.approx(0.05, abs=0.01)
    assert torch.all(batch.curriculum_difficulty[batch.stages < 5] == 1.0)
    assert torch.all(batch.curriculum_difficulty[batch.stages == 5] == 0.4)
    assert torch.all(batch.curriculum_difficulty[batch.stages == 6] == 0.0)


def test_stage_geometry_limits_are_independent_from_mastery_gate() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    config = load_project_config(config_dir)
    schedule = CurriculumStageSchedule(
        config["curriculum"]["stages"], TERRAIN_NAMES, "cpu"
    )
    batch = schedule.sample(
        torch.tensor([3, 4, 5, 7]),
        torch.ones(4),
        frontier_probability=1.0,
        challenge_probability=0.0,
        challenge_stage_span=1,
    )

    assert batch.curriculum_difficulty.tolist() == pytest.approx([1.0] * 4)
    assert batch.geometry_difficulty.tolist() == pytest.approx(
        [0.45, 0.45, 1.0, 0.65]
    )


def test_frontier_geometry_difficulty_requires_history_and_success() -> None:
    curriculum = CapabilityCurriculum(
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
    curriculum = CapabilityCurriculum(
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
