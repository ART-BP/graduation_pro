"""Vectorized reward composition independent of Isaac Sim."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RewardWeights:
    # 目标趋近是最主要奖励。
    progress: float = 8.0
    goal_reached: float = 20.0

    # 轻量方向引导，避免策略只靠稀疏距离差探索。
    heading: float = 0.05
    forward_to_goal: float = 0.25

    collision: float = -20.0
    unstable: float = -15.0
    stuck: float = -5.0

    # 动作变化惩罚：抑制指令抖动。
    linear_action_rate: float = -0.02
    angular_action_rate: float = -0.02

    # 持续转动惩罚：解决恒定角速度绕圈。
    angular_speed: float = -0.08
    spin: float = -0.25

    time: float = -0.01

    # 第一阶段暂时不惩罚路径长度，避免压制必要前进。
    path_length: float = 0.0

    action_limit_violation: float = -0.25

    # 原来的-0.10可能接近或抵消每步前进收益。
    unknown_risk: float = -0.02


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
    goal_bearing,
    actual_velocity,
    weights: RewardWeights,
):
    # 沿目标距离方向产生的真实进展。
    progress = previous_distance - current_distance
    reward = weights.progress * progress

    # 目标位于正前方时为正，背后时为负。
    heading_cosine = torch.cos(goal_bearing)
    reward = reward + weights.heading * heading_cosine

    # 只有向前并且朝向目标时才奖励。
    forward_speed = torch.clamp(actual_velocity[:, 0], min=0.0)
    forward_alignment = torch.clamp(heading_cosine, min=0.0)
    reward = reward + (
        weights.forward_to_goal
        * forward_speed
        * forward_alignment
    )

    reward = reward + weights.goal_reached * reached.float()
    reward = reward + weights.collision * collision.float()
    reward = reward + weights.unstable * unstable.float()
    reward = reward + weights.stuck * stuck.float()

    # 动作变化率惩罚。
    action_delta = action - previous_action
    reward = reward + (
        weights.linear_action_rate
        * action_delta[:, 0].square()
    )
    reward = reward + (
        weights.angular_action_rate
        * action_delta[:, 1].square()
    )

    # 无论角速度是否变化，只要持续旋转就产生惩罚。
    actual_linear_speed = actual_velocity[:, 0]
    actual_angular_speed = actual_velocity[:, 1]

    reward = reward + (
        weights.angular_speed
        * actual_angular_speed.square()
    )

    # 线速度很低但角速度很高，视为原地绕圈。
    spin_mask = (
        (actual_angular_speed.abs() > 0.35)
        & (actual_linear_speed.abs() < 0.10)
    )

    reward = reward + weights.spin * spin_mask.float()

    reward = reward + weights.path_length * path_increment
    reward = reward + (
        weights.action_limit_violation
        * action_limit_violation
    )
    reward = reward + weights.unknown_risk * unknown_ratio
    reward = reward + weights.time

    return reward
