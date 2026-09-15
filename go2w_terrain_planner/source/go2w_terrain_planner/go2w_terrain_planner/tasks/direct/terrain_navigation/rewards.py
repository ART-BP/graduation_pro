"""Vectorized reward composition independent of Isaac Sim."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RewardWeights:
    # 目标趋近是最主要奖励。
    progress: float = 8.0
    # 绕障时允许短暂横移或退让，但仍用较小代价约束无效远离。
    regression: float = -1.0
    goal_reached: float = 20.0

    # 轻量方向引导，避免策略只靠稀疏距离差探索。
    heading: float = 0.05
    forward_to_goal: float = 0.25

    collision: float = -20.0
    unstable: float = -15.0
    stuck: float = -5.0
    timeout: float = -15.0
    out_of_bounds: float = -10.0
    observation_failure: float = -5.0

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
    # 高风险地形上的高速运动惩罚，鼓励提前减速或绕行。
    terrain_speed_risk: float = -0.15


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
    *,
    best_distance=None,
    terrain_risk=None,
    time_out=None,
    out_of_bounds=None,
    observation_failure=None,
    return_terms: bool = False,
):
    zeros = torch.zeros_like(previous_distance)
    time_out = zeros if time_out is None else time_out.float()
    out_of_bounds = zeros if out_of_bounds is None else out_of_bounds.float()
    observation_failure = (
        zeros if observation_failure is None else observation_failure.float()
    )
    terrain_risk = zeros if terrain_risk is None else terrain_risk

    # 进展只在刷新本episode历史最近距离时结算，避免策略通过反复靠近和
    # 远离目标重复领取正奖励。远离仍维持轻惩罚，以允许必要绕障。
    distance_delta = previous_distance - current_distance
    progress_reference = previous_distance if best_distance is None else best_distance
    progress_term = weights.progress * torch.clamp(
        progress_reference - current_distance, min=0.0
    )
    regression_term = weights.regression * torch.clamp(-distance_delta, min=0.0)

    # 方向奖励必须由真实运动门控，防止机器人原地朝向目标持续获得净正回报。
    heading_cosine = torch.cos(goal_bearing)
    actual_linear_speed = actual_velocity[:, 0]
    heading_motion_gate = torch.clamp(actual_linear_speed.abs() / 0.10, 0.0, 1.0)
    heading_term = weights.heading * heading_cosine * heading_motion_gate

    # 只有向前并且朝向目标时才奖励。
    forward_speed = torch.clamp(actual_linear_speed, min=0.0)
    forward_alignment = torch.clamp(heading_cosine, min=0.0)
    forward_term = (
        weights.forward_to_goal
        * forward_speed
        * forward_alignment
    )
    terrain_speed_risk_term = (
        weights.terrain_speed_risk
        * torch.clamp(terrain_risk, 0.0, 1.0)
        * actual_linear_speed.square()
    )

    # 动作变化率惩罚。
    action_delta = action - previous_action
    linear_action_rate_term = (
        weights.linear_action_rate
        * action_delta[:, 0].square()
    )
    angular_action_rate_term = (
        weights.angular_action_rate
        * action_delta[:, 1].square()
    )

    # 无论角速度是否变化，只要持续旋转就产生惩罚。
    actual_angular_speed = actual_velocity[:, 1]
    angular_speed_term = (
        weights.angular_speed
        * actual_angular_speed.square()
    )

    # 线速度很低但角速度很高，视为原地绕圈。
    spin_mask = (
        (actual_angular_speed.abs() > 0.35)
        & (actual_linear_speed.abs() < 0.10)
    )

    terms = {
        "progress": progress_term,
        "regression": regression_term,
        "heading": heading_term,
        "forward_to_goal": forward_term,
        "goal_reached": weights.goal_reached * reached.float(),
        "collision": weights.collision * collision.float(),
        "unstable": weights.unstable * unstable.float(),
        "stuck": weights.stuck * stuck.float(),
        "timeout": weights.timeout * time_out,
        "out_of_bounds": weights.out_of_bounds * out_of_bounds,
        "observation_failure": weights.observation_failure * observation_failure,
        "linear_action_rate": linear_action_rate_term,
        "angular_action_rate": angular_action_rate_term,
        "angular_speed": angular_speed_term,
        "spin": weights.spin * spin_mask.float(),
        "path_length": weights.path_length * path_increment,
        "action_limit_violation": (
            weights.action_limit_violation * action_limit_violation
        ),
        "unknown_risk": weights.unknown_risk * unknown_ratio,
        "terrain_speed_risk": terrain_speed_risk_term,
        "time": torch.full_like(previous_distance, weights.time),
    }
    reward = torch.stack(tuple(terms.values()), dim=0).sum(dim=0)
    return (reward, terms) if return_terms else reward
