"""Small runtime guards for batched observations."""

from __future__ import annotations


def require_finite(tensor, name: str) -> None:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name}必须是torch.Tensor")
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name}包含NaN或Inf")


def require_shape(tensor, expected: tuple[int | None, ...], name: str) -> None:
    if tensor.ndim != len(expected):
        raise ValueError(f"{name}维数错误：{tuple(tensor.shape)}，期望{expected}")
    for actual, wanted in zip(tensor.shape, expected, strict=True):
        if wanted is not None and actual != wanted:
            raise ValueError(f"{name}形状错误：{tuple(tensor.shape)}，期望{expected}")
