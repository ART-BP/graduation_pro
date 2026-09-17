"""Spatial encoders for aligned local elevation-map histories."""

from __future__ import annotations

import torch
from torch import nn


def _normalization_groups(channels: int) -> int:
    """Choose a small GroupNorm divisor without depending on batch size."""

    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


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
    def __init__(
        self,
        input_channels: int = 4,
        feature_dim: int = 192,
        *,
        encoder_channels: tuple[int, int, int, int] | list[int] = (24, 48, 96, 128),
        spatial_pool_size: int = 6,
    ) -> None:
        super().__init__()

        self.input_channels = int(input_channels)
        self.encoder_channels = tuple(int(value) for value in encoder_channels)
        self.spatial_pool_size = int(spatial_pool_size)
        if (
            len(self.encoder_channels) != 4
            or any(value <= 0 for value in self.encoder_channels)
            or self.spatial_pool_size <= 0
            or feature_dim <= 0
        ):
            raise ValueError("地图编码器通道数、池化尺寸和特征维度必须为正数")
        channel_1, channel_2, channel_3, channel_4 = self.encoder_channels

        self.network = nn.Sequential(
            nn.Conv2d(
                self.input_channels,
                channel_1,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.ELU(),
            nn.Conv2d(
                channel_1,
                channel_2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ELU(),
            nn.Conv2d(
                channel_2,
                channel_3,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ELU(),
            nn.Conv2d(
                channel_3,
                channel_4,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ELU(),
            # Use the fused PyTorch kernel during training. It is replaced by
            # the slice-based equivalent only when preparing ONNX export.
            nn.AdaptiveAvgPool2d(
                (self.spatial_pool_size, self.spatial_pool_size)
            ),
            nn.Flatten(),
            nn.Linear(
                channel_4
                * self.spatial_pool_size
                * self.spatial_pool_size,
                feature_dim,
            ),
            nn.ELU(),
        )

    def prepare_for_onnx_export(self) -> None:
        """Replace unsupported adaptive pooling without changing weights."""

        for index, module in enumerate(self.network):
            if isinstance(module, nn.AdaptiveAvgPool2d):
                self.network[index] = OnnxFriendlyAdaptiveAvgPool2d(
                    (self.spatial_pool_size, self.spatial_pool_size)
                )

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        # Avoid tensor-to-Python-boolean warnings while tracing ONNX.
        if not torch.jit.is_tracing():
            if maps.ndim != 4 or maps.shape[1] != self.input_channels:
                raise ValueError(
                    f"MapEncoder输入必须为[B,{self.input_channels},H,W]"
                )

        return self.network(maps)


class ResidualSpatialBlock(nn.Module):
    """Two-convolution residual block for stable PPO optimization."""

    def __init__(self, channels: int, *, dilation: int = 1) -> None:
        super().__init__()
        groups = _normalization_groups(channels)
        self.network = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.network(features)


class MultiScaleContextBlock(nn.Module):
    """Keep local edges and wider terrain context at the 1/8 map scale."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        if output_channels % 4 != 0:
            raise ValueError("多尺度上下文输出通道数必须为4的倍数")
        branch_channels = output_channels // 4
        branches: list[nn.Module] = []
        for dilation in (1, 2, 4, 6):
            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        input_channels,
                        branch_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.GroupNorm(
                        _normalization_groups(branch_channels),
                        branch_channels,
                    ),
                    nn.SiLU(),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.shortcut = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=1,
            bias=False,
        )
        self.output = nn.Sequential(
            nn.GroupNorm(
                _normalization_groups(output_channels),
                output_channels,
            ),
            nn.SiLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        context = torch.cat(
            [branch(features) for branch in self.branches],
            dim=1,
        )
        return self.output(context + self.shortcut(features))


class CompactTerrainMapEncoder(nn.Module):
    """Encode one compact terrain map without stacking full historical maps.

    The input contains the current fused map, a recent-change channel, and
    explicit cell age/confidence. A residual feature pyramid retains fine
    terrain edges and wider context. Goal-conditioned attention selects
    evidence along the intended navigation direction.
    """

    def __init__(
        self,
        input_channels: int = 7,
        map_size: int = 200,
        feature_dim: int = 384,
        *,
        encoder_channels: tuple[int, int, int, int] | list[int] = (
            32,
            64,
            96,
            128,
        ),
        spatial_pool_size: int = 8,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.map_size = int(map_size)
        self.encoder_channels = tuple(int(value) for value in encoder_channels)
        self.spatial_pool_size = int(spatial_pool_size)
        self.feature_dim = int(feature_dim)
        if (
            self.input_channels <= 0
            or self.map_size <= 0
            or len(self.encoder_channels) != 4
            or any(value <= 0 for value in self.encoder_channels)
            or self.encoder_channels[3] % 4 != 0
            or self.spatial_pool_size <= 0
            or self.feature_dim <= 0
        ):
            raise ValueError("紧凑地图编码器参数无效")

        channel_1, channel_2, channel_3, channel_4 = self.encoder_channels
        # x/y/r坐标使卷积特征保留机器人前后、左右和距离信息。
        axis = torch.linspace(-1.0, 1.0, self.map_size)
        coordinate_x, coordinate_y = torch.meshgrid(axis, axis, indexing="ij")
        coordinates = torch.stack(
            (
                coordinate_x,
                coordinate_y,
                torch.sqrt(coordinate_x.square() + coordinate_y.square()),
            ),
            dim=0,
        )
        self.register_buffer("robot_coordinates", coordinates, persistent=True)

        # 地图与坐标分支在首次降采样后相加，避免为PPO大mini-batch
        # 额外物化坐标拼接张量。
        self.map_stem = nn.Conv2d(
            self.input_channels,
            channel_1,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.coordinate_stem = nn.Conv2d(
            3,
            channel_1,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.stem = nn.Sequential(
            nn.GroupNorm(_normalization_groups(channel_1), channel_1),
            nn.SiLU(),
            ResidualSpatialBlock(channel_1),
        )
        self.fine_stage = nn.Sequential(
            nn.Conv2d(
                channel_1,
                channel_2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            ResidualSpatialBlock(channel_2),
        )
        self.coarse_stage = nn.Sequential(
            nn.Conv2d(
                channel_2,
                channel_3,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            ResidualSpatialBlock(channel_3, dilation=2),
        )
        self.context_stage = MultiScaleContextBlock(channel_3, channel_4)
        self.fine_pool: nn.Module = nn.AdaptiveAvgPool2d(
            (self.spatial_pool_size, self.spatial_pool_size)
        )
        self.context_pool: nn.Module = nn.AdaptiveAvgPool2d(
            (self.spatial_pool_size, self.spatial_pool_size)
        )
        self.goal_query = nn.Sequential(
            nn.Linear(3, channel_4),
            nn.SiLU(),
            nn.Linear(channel_4, channel_4),
        )
        projected_input_dim = (
            (channel_2 + channel_4)
            * self.spatial_pool_size
            * self.spatial_pool_size
            + channel_4
        )
        self.projection = nn.Sequential(
            nn.Linear(projected_input_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.SiLU(),
        )
        self.attention_scale = channel_4 ** -0.5

    def prepare_for_onnx_export(self) -> None:
        """Replace adaptive pools with fixed-slice ONNX-friendly modules."""

        output_size = (self.spatial_pool_size, self.spatial_pool_size)
        if isinstance(self.fine_pool, nn.AdaptiveAvgPool2d):
            self.fine_pool = OnnxFriendlyAdaptiveAvgPool2d(output_size)
        if isinstance(self.context_pool, nn.AdaptiveAvgPool2d):
            self.context_pool = OnnxFriendlyAdaptiveAvgPool2d(output_size)

    def forward(self, maps: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing():
            expected = (
                maps.shape[0],
                self.input_channels,
                self.map_size,
                self.map_size,
            )
            if maps.ndim != 4 or tuple(maps.shape) != expected:
                raise ValueError(
                    "CompactTerrainMapEncoder输入地图必须为"
                    f"[B,{self.input_channels},"
                    f"{self.map_size},{self.map_size}]"
                )
            if goal.shape != (maps.shape[0], 3):
                raise ValueError("目标编码必须为[B,3]")

        coordinates = self.robot_coordinates.to(dtype=maps.dtype).unsqueeze(0)
        stem_input = (
            self.map_stem(maps)
            + self.coordinate_stem(coordinates)
        )
        stem = self.stem(stem_input)
        fine = self.fine_stage(stem)
        coarse = self.coarse_stage(fine)
        context = self.context_stage(coarse)

        query = self.goal_query(goal)
        attention_logits = (
            context * query[:, :, None, None]
        ).sum(dim=1) * self.attention_scale
        attention = torch.softmax(attention_logits.flatten(1), dim=1)
        attended = (
            context.flatten(2) * attention[:, None, :]
        ).sum(dim=-1)
        spatial = torch.cat(
            (
                self.fine_pool(fine).flatten(1),
                self.context_pool(context).flatten(1),
                attended,
            ),
            dim=-1,
        )
        return self.projection(spatial)
