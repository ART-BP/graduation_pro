import numpy as np
import pytest

torch = pytest.importorskip("torch")

from go2w_terrain_planner.mapping.grid_preprocessor import (
    downsample_map_tensor,
    preprocess_grid_map,
    preprocess_grid_map_torch,
)


def test_numpy_and_torch_preprocessing_match() -> None:
    ground = np.array([[0.0, np.nan], [0.3, 1.5]], dtype=np.float32)
    height_range = np.array([[0.0, np.nan], [0.3, 4.0]], dtype=np.float32)
    observed = np.ones((2, 2), dtype=np.float32)
    expected = preprocess_grid_map(
        ground,
        height_range,
        observed,
        normalize=True,
    )
    actual = preprocess_grid_map_torch(
        ground,
        height_range,
        observed,
        normalize=True,
    )
    assert np.allclose(np.asarray(actual.tolist(), dtype=np.float32), expected)


def test_downsample_preserves_channel_contract() -> None:
    maps = torch.rand((2, 5, 4, 20, 20))
    maps[:, :, 2:] = (maps[:, :, 2:] > 0.5).float()
    result = downsample_map_tensor(maps, 10)
    assert result.shape == (2, 5, 4, 10, 10)
    assert set(torch.unique(result[:, :, 2:]).tolist()).issubset({0.0, 1.0})
