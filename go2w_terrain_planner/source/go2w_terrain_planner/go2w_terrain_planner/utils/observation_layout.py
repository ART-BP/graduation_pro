"""Central dimensions for actor and privileged critic observations."""

from __future__ import annotations


CRITIC_OBSERVATION_COMPONENTS = {
    "local_goal": 3,
    "navigation_potential": 1,
    "actual_velocity": 2,
    "current_command": 2,
    "tracking_error": 2,
    "terrain_type": 1,
    "terrain_difficulty": 1,
    "terrain_amplitude": 1,
    "feature_position": 2,
    "feature_heading": 2,
    "feature_width": 1,
    "barrier_half_width": 1,
    "friction": 1,
    "terrain_risk": 1,
    "observed_terrain_risk": 1,
    "unknown_ratio": 1,
    "stuck_fraction": 1,
    "command_history": 16,
    "motion_history": 24,
}
CRITIC_OBSERVATION_DIMENSION = sum(CRITIC_OBSERVATION_COMPONENTS.values())

FUSED_MAP_CHANNELS = 4
RECENT_CHANGE_CHANNELS = 1
MAP_AGE_CONFIDENCE_CHANNELS = 2
ACTOR_MAP_CHANNELS = (
    FUSED_MAP_CHANNELS
    + RECENT_CHANGE_CHANNELS
    + MAP_AGE_CONFIDENCE_CHANNELS
)


def policy_observation_dimension(
    *,
    map_channels: int,
    map_size: int,
    command_history_length: int,
    motion_history_length: int,
) -> int:
    """Return the flattened deployable Actor observation dimension."""

    values = (
        map_channels,
        map_size,
        command_history_length,
        motion_history_length,
    )
    if any(not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("Actor观测布局参数必须为正整数")
    return (
        map_channels * map_size * map_size
        + 3
        + 2
        + 2 * command_history_length
        + 3 * motion_history_length
    )


def critic_observation_dimension(
    *,
    command_history_length: int,
    motion_history_length: int,
) -> int:
    """Return the privileged Critic dimension for configurable histories."""

    if (
        not isinstance(command_history_length, int)
        or not isinstance(motion_history_length, int)
        or command_history_length <= 0
        or motion_history_length <= 0
    ):
        raise ValueError("Critic历史长度必须为正整数")
    history_dimension = (
        CRITIC_OBSERVATION_COMPONENTS["command_history"]
        + CRITIC_OBSERVATION_COMPONENTS["motion_history"]
    )
    fixed_dimension = CRITIC_OBSERVATION_DIMENSION - history_dimension
    return (
        fixed_dimension
        + 2 * command_history_length
        + 3 * motion_history_length
    )


DEFAULT_POLICY_OBSERVATION_DIMENSION = policy_observation_dimension(
    map_channels=ACTOR_MAP_CHANNELS,
    map_size=200,
    command_history_length=8,
    motion_history_length=8,
)
