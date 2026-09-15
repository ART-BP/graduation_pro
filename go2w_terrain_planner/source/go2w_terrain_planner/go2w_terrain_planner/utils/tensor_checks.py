"""Small runtime guards for batched observations."""

from __future__ import annotations

import os


def runtime_checks_enabled() -> bool:
    """Return whether expensive tensor-value guards are enabled.

    Shape and configuration validation always remain active. Full finite-value
    scans synchronize CUDA with the CPU, so the training entrypoint disables
    them after smoke tests have passed.
    """

    value = os.environ.get("GO2W_RUNTIME_CHECKS", "1")
    return value.strip().lower() not in {"0", "false", "no", "off"}


def require_finite(tensor, name: str, *, enabled: bool | None = None) -> None:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name}必须是torch.Tensor")
    if enabled is None:
        enabled = runtime_checks_enabled()
    if not enabled:
        return
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name}包含NaN或Inf")


def require_shape(tensor, expected: tuple[int | None, ...], name: str) -> None:
    if tensor.ndim != len(expected):
        raise ValueError(f"{name}维数错误：{tuple(tensor.shape)}，期望{expected}")
    for actual, wanted in zip(tensor.shape, expected, strict=True):
        if wanted is not None and actual != wanted:
            raise ValueError(f"{name}形状错误：{tuple(tensor.shape)}，期望{expected}")
