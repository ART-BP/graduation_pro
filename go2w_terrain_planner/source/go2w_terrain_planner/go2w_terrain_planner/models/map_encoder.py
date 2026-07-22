"""Lightweight shared CNN for one four-channel local elevation map."""

from __future__ import annotations

import torch
from torch import nn


class OnnxFriendlyAdaptiveAvgPool2d(nn.Module):
    """Adaptive average pooling implemented with constant slices.

    The spatial input size is fixed for each exported model. During ONNX
    tracing, H and W are frozen as constants while the batch dimension remains
    dynamic. This avoids the legacy ONNX exporter's AdaptiveAvgPool2d shape
    limitation.
    """

    def __init__(self, output_size: tuple[int, int] = (4, 4)) -> None:
        super().__init__()
        self.output_height = int(output_size[0])
        self.output_width = int(output_size[1])

        if self.output_height <= 0 or self.output_width <= 0:
            raise ValueError("output_size必须为正整数")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only batch is dynamic in the exported model. Converting H and W to
        # Python integers deliberately freezes the spatial map size.
        input_height = int(x.shape[-2])
        input_width = int(x.shape[-1])

        rows: list[torch.Tensor] = []

        for output_y in range(self.output_height):
            y_start = (
                output_y * input_height // self.output_height
            )
            y_end = (
                (output_y + 1) * input_height
                + self.output_height
                - 1
            ) // self.output_height

            columns: list[torch.Tensor] = []

            for output_x in range(self.output_width):
                x_start = (
                    output_x * input_width // self.output_width
                )
                x_end = (
                    (output_x + 1) * input_width
                    + self.output_width
                    - 1
                ) // self.output_width

                pooled = x[
                    :,
                    :,
                    y_start:y_end,
                    x_start:x_end,
                ].mean(dim=(-2, -1), keepdim=True)

                columns.append(pooled)

            rows.append(torch.cat(columns, dim=3))

        return torch.cat(rows, dim=2)


class MapEncoder(nn.Module):
    def __init__(self, input_channels: int = 4, feature_dim: int = 128) -> None:
        super().__init__()

        self.input_channels = int(input_channels)

        self.network = nn.Sequential(
            nn.Conv2d(
                self.input_channels,
                16,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.ELU(),
            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ELU(),
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ELU(),
            nn.Conv2d(
                64,
                96,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ELU(),
            # Preserve a 4x4 coarse spatial layout while remaining compatible
            # with dynamic-batch ONNX export.
            OnnxFriendlyAdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, feature_dim),
            nn.ELU(),
        )

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        # Avoid tensor-to-Python-boolean warnings while tracing ONNX.
        if not torch.jit.is_tracing():
            if maps.ndim != 4 or maps.shape[1] != self.input_channels:
                raise ValueError(
                    f"MapEncoder输入必须为[B,{self.input_channels},H,W]"
                )

        return self.network(maps)
