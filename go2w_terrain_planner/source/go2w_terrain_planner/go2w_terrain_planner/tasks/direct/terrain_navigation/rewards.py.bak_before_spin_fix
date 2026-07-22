"""Vectorized reward composition independent of Isaac Sim."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardWeights:
    progress: float = 4.0
    goal_reached: float = 15.0
    collision: float = -20.0
    unstable: float = -15.0
    stuck: float = -5.0
    linear_action_rate: float = -0.05
    angular_action_rate: float = -0.03
    time: float = -0.01
    path_length: float = -0.02
    action_limit_violation: float = -0.25
    unknown_risk: float = -0.10


def navigation_reward(
    previous_distance,
    current_distance,
    action,
    previous_action,
    path_increment,
    action_limit_violation,
    unknown_ratio,
    reached,
    collision,
    unstable,
    stuck,
    weights: RewardWeights,
):
    reward = weights.progress * (previous_distance - current_distance)
    reward = reward + weights.goal_reached * reached.float()
    reward = reward + weights.collision * collision.float()
    reward = reward + weights.unstable * unstable.float()
    reward = reward + weights.stuck * stuck.float()
    action_delta = action - previous_action
    reward = reward + weights.linear_action_rate * action_delta[:, 0].square()
    reward = reward + weights.angular_action_rate * action_delta[:, 1].square()
    reward = reward + weights.path_length * path_increment
    reward = reward + weights.action_limit_violation * action_limit_violation
    reward = reward + weights.unknown_risk * unknown_ratio
    reward = reward + weights.time
    return reward
