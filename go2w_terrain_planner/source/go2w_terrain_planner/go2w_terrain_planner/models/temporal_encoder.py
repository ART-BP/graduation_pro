"""GRU encoder over already pose-aligned map features."""

from __future__ import annotations

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    def __init__(self, feature_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden_dim, batch_first=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("TemporalEncoder输入必须为[B,T,F]")
        initial = torch.zeros(
            self.gru.num_layers,
            features.shape[0],
            self.gru.hidden_size,
            dtype=features.dtype,
            device=features.device,
        )
        _, hidden = self.gru(features, initial)
        return hidden[-1]
