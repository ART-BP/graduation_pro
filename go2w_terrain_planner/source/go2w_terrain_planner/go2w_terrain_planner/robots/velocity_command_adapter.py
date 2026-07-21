"""Map normalized policy actions to Go2W velocity commands and simulate tracking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActionLimits:
    linear_min_mps: float = -0.2
    linear_max_mps: float = 0.8
    angular_min_radps: float = -1.0
    angular_max_radps: float = 1.0


@dataclass
class ExecutionModelConfig:
    linear_time_constant_s: float = 0.25
    angular_time_constant_s: float = 0.20
    linear_acceleration_limit_mps2: float = 0.8
    angular_acceleration_limit_radps2: float = 2.0
    maximum_terrain_speed_loss: float = 0.25
    tracking_noise_std: float = 0.02
    entry_angle_speed_penalty: float = 0.40


class VelocityCommandAdapter:
    """Convert normalized actions in ``[-1,1]`` to configured physical limits."""

    def __init__(self, limits: ActionLimits | None = None) -> None:
        self.limits = limits or ActionLimits()
        if self.limits.linear_min_mps >= self.limits.linear_max_mps:
            raise ValueError("线速度上下限无效")
        if self.limits.angular_min_radps >= self.limits.angular_max_radps:
            raise ValueError("角速度上下限无效")

    def to_physical(self, normalized_action):
        import torch

        if normalized_action.shape[-1] != 2:
            raise ValueError("动作最后一维必须为2")
        if not torch.isfinite(normalized_action).all():
            raise RuntimeError("归一化动作包含NaN或Inf")
        action = torch.clamp(normalized_action, -1.0, 1.0)
        linear = self.limits.linear_min_mps + 0.5 * (action[..., 0] + 1.0) * (
            self.limits.linear_max_mps - self.limits.linear_min_mps
        )
        angular = self.limits.angular_min_radps + 0.5 * (action[..., 1] + 1.0) * (
            self.limits.angular_max_radps - self.limits.angular_min_radps
        )
        return torch.stack((linear, angular), dim=-1)


class VelocityExecutionModel:
    """First-order tracking model with terrain-dependent speed loss."""

    def __init__(self, num_envs: int, device, config: ExecutionModelConfig | None = None) -> None:
        import torch

        self.cfg = config or ExecutionModelConfig()
        if self.cfg.linear_time_constant_s <= 0.0 or self.cfg.angular_time_constant_s <= 0.0:
            raise ValueError("速度响应时间常数必须大于0")
        if (
            self.cfg.linear_acceleration_limit_mps2 <= 0.0
            or self.cfg.angular_acceleration_limit_radps2 <= 0.0
        ):
            raise ValueError("加速度限制必须大于0")
        if not 0.0 <= self.cfg.maximum_terrain_speed_loss < 1.0:
            raise ValueError("maximum_terrain_speed_loss必须位于[0,1)")
        self.actual_velocity = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
        self.time_constants = torch.tensor(
            [self.cfg.linear_time_constant_s, self.cfg.angular_time_constant_s],
            dtype=torch.float32,
            device=device,
        )
        self.acceleration_limits = torch.tensor(
            [
                self.cfg.linear_acceleration_limit_mps2,
                self.cfg.angular_acceleration_limit_radps2,
            ],
            dtype=torch.float32,
            device=device,
        )

    def reset(self, env_ids) -> None:
        self.actual_velocity[env_ids] = 0.0

    def step(self, command, terrain_resistance, friction, dt: float, entry_alignment=None, blocked=None):
        import torch

        if command.shape != self.actual_velocity.shape:
            raise ValueError("command尺寸与环境数量不一致")
        if dt <= 0.0:
            raise ValueError("dt必须大于0")
        friction_scale = torch.clamp(friction, 0.2, 1.0)
        resistance = torch.clamp(terrain_resistance, 0.0, 1.0)
        terrain_scale = 1.0 - self.cfg.maximum_terrain_speed_loss * resistance
        if entry_alignment is not None:
            alignment = torch.clamp(entry_alignment, 0.0, 1.0)
            terrain_scale *= 1.0 - resistance * self.cfg.entry_angle_speed_penalty * (1.0 - alignment)
        target = command.clone()
        target[:, 0] *= friction_scale * terrain_scale
        if blocked is not None:
            target[:, 0] = torch.where(blocked, torch.zeros_like(target[:, 0]), target[:, 0])

        response = torch.clamp(dt / self.time_constants.to(dtype=command.dtype), max=1.0)
        desired_delta = (target - self.actual_velocity) * response
        max_delta = self.acceleration_limits.to(dtype=command.dtype) * dt
        delta = torch.clamp(desired_delta, -max_delta, max_delta)
        self.actual_velocity += delta
        if self.cfg.tracking_noise_std > 0.0:
            self.actual_velocity += (
                self.cfg.tracking_noise_std
                * (dt**0.5)
                * torch.randn_like(self.actual_velocity)
            )
        if blocked is not None:
            self.actual_velocity[:, 0] = torch.where(
                blocked, 0.05 * self.actual_velocity[:, 0], self.actual_velocity[:, 0]
            )
        if not torch.isfinite(self.actual_velocity).all():
            raise RuntimeError("速度执行模型产生NaN或Inf")
        return self.actual_velocity
