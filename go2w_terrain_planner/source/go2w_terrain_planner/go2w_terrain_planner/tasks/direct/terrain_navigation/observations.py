"""Tensor-only observation assembly shared by training checks and the task."""

from __future__ import annotations


def assemble_policy_observation(
    aligned_maps,
    goal,
    current_velocity,
    command_history,
    motion_history,
    imu_history=None,
    *,
    expected_dimension: int | None = None,
):
    """Assemble the deployable actor observation without privileged state."""
    import torch

    if aligned_maps.ndim != 5 or aligned_maps.shape[2] != 4:
        raise ValueError("aligned_maps必须为[B,T,4,H,W]")
    batch, history = aligned_maps.shape[:2]
    if goal.shape != (batch, 3):
        raise ValueError("goal必须为[B,3]")
    if current_velocity.shape != (batch, 2):
        raise ValueError("current_velocity必须为[B,2]")
    if command_history.shape != (batch, history - 1, 2):
        raise ValueError("command_history必须为[B,T-1,2]")
    if motion_history.shape != (batch, history - 1, 3):
        raise ValueError("motion_history必须为[B,T-1,3]")
    values = [
        aligned_maps.flatten(1),
        goal,
        current_velocity,
        command_history.flatten(1),
        motion_history.flatten(1),
    ]
    if imu_history is not None:
        if imu_history.shape != (batch, history - 1, 5):
            raise ValueError("imu_history必须为[B,T-1,5]")
        values.append(imu_history.flatten(1))
    output = torch.cat(values, dim=-1)
    if expected_dimension is not None and output.shape[-1] != expected_dimension:
        raise ValueError(
            f"policy观测维度错误：actual={output.shape[-1]}, expected={expected_dimension}"
        )
    if not torch.isfinite(output).all():
        raise RuntimeError("policy观测包含NaN或Inf")
    return output
