import pytest

from go2w_terrain_planner.utils.observation_layout import (
    ACTOR_MAP_CHANNELS,
    CRITIC_OBSERVATION_COMPONENTS,
    CRITIC_OBSERVATION_DIMENSION,
    DEFAULT_POLICY_OBSERVATION_DIMENSION,
    policy_observation_dimension,
)


def test_default_actor_and_critic_dimensions_are_explicit() -> None:
    assert ACTOR_MAP_CHANNELS == 7
    assert DEFAULT_POLICY_OBSERVATION_DIMENSION == 280025
    assert CRITIC_OBSERVATION_DIMENSION == 44
    assert sum(CRITIC_OBSERVATION_COMPONENTS.values()) == 44


def test_policy_dimension_tracks_map_resolution() -> None:
    assert policy_observation_dimension(
        map_channels=7,
        map_size=100,
        command_history_length=4,
        motion_history_length=4,
    ) == 70025
    with pytest.raises(ValueError, match="正整数"):
        policy_observation_dimension(
            map_channels=7,
            map_size=0,
            command_history_length=4,
            motion_history_length=4,
        )
