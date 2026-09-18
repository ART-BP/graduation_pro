import pytest

torch = pytest.importorskip("torch")

from go2w_terrain_planner.mapping.grid_preprocessor import (
    downsample_map_tensor,
    preprocess_grid_map_torch,
)


def test_torch_preprocessing_normalizes_and_cleans_invalid_heights() -> None:
    ground = torch.tensor([[0.0, torch.nan], [0.3, 1.5]])
    height_range = torch.tensor([[0.0, torch.nan], [0.3, 4.0]])
    observed = torch.ones((2, 2))
    actual = preprocess_grid_map_torch(
        ground,
        height_range,
        observed,
        normalize=True,
    )
    expected = torch.tensor(
        [
            [[0.0, 0.0], [0.2, 1.0]],
            [[0.0, 0.0], [0.1, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[1.0, 0.0], [1.0, 1.0]],
        ]
    )
    assert torch.allclose(actual, expected)


def test_torch_preprocessing_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="尺寸必须一致"):
        preprocess_grid_map_torch(
            torch.zeros((10, 10)),
            torch.zeros((8, 10)),
            torch.zeros((10, 10)),
        )


def test_downsample_preserves_channel_contract() -> None:
    maps = torch.rand((2, 5, 4, 20, 20))
    maps[:, :, 2:] = (maps[:, :, 2:] > 0.5).float()
    result = downsample_map_tensor(maps, 10)
    assert result.shape == (2, 5, 4, 10, 10)
    assert set(torch.unique(result[:, :, 2:]).tolist()).issubset({0.0, 1.0})
