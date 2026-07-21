import pytest

torch = pytest.importorskip("torch")

from go2w_terrain_planner.tasks.direct.terrain_navigation.curriculum import TerrainCurriculum


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
    )
    curriculum.update(torch.tensor([0]), torch.tensor([False]))
    assert curriculum.levels.item() == 1
    curriculum.update(torch.tensor([0]), torch.tensor([False]))
    assert curriculum.levels.item() == 1
    curriculum.update(torch.tensor([0]), torch.tensor([False]))
    assert curriculum.levels.item() == 0
