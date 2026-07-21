"""Lightweight shared CNN for one four-channel local elevation map."""

from __future__ import annotations

import torch
from torch import nn


class MapEncoder(nn.Module):
    def __init__(self, input_channels: int = 4, feature_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            # Keep a coarse spatial layout. Global 1x1 pooling made obstacle
            # features almost translation invariant, so the actor could not
            # reliably tell left, right and forward terrain apart.
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, feature_dim),
            nn.ELU(),
        )

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        if maps.ndim != 4 or maps.shape[1] != 4:
            raise ValueError("MapEncoder输入必须为[B,4,H,W]")
        return self.network(maps)
