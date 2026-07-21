import numpy as np
import pytest

from go2w_terrain_planner.preprocessing import preprocess_grid_map


def test_preprocess_grid_map() -> None:
    ground = np.array(
        [
            [1.00, np.nan],
            [1.10, 1.20],
        ],
        dtype=np.float32,
    )

    height_range = np.array(
        [
            [0.00, np.nan],
            [0.05, 0.10],
        ],
        dtype=np.float32,
    )

    observed = np.array(
        [
            [1.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = preprocess_grid_map(
        ground,
        height_range,
        observed,
        robot_z=1.0,
    )

    assert result.shape == (4, 2, 2)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()

    # relative_ground_height
    assert result[0, 0, 0] == pytest.approx(0.0)
    assert result[0, 1, 0] == pytest.approx(0.1)

    # 射线经过但没有有效高程。
    assert result[2, 0, 1] == pytest.approx(1.0)
    assert result[3, 0, 1] == pytest.approx(0.0)

    # 有效高程。
    assert result[3, 1, 0] == pytest.approx(1.0)


def test_shape_mismatch() -> None:
    ground = np.zeros((10, 10), dtype=np.float32)
    height_range = np.zeros((8, 10), dtype=np.float32)
    observed = np.zeros((10, 10), dtype=np.float32)

    with pytest.raises(ValueError):
        preprocess_grid_map(
            ground,
            height_range,
            observed,
            robot_z=0.0,
        )


def test_invalid_robot_z() -> None:
    grid = np.zeros((10, 10), dtype=np.float32)

    with pytest.raises(ValueError):
        preprocess_grid_map(
            grid,
            grid,
            grid,
            robot_z=float("nan"),
        )


def test_height_clipping() -> None:
    ground = np.array([[10.0, -10.0]], dtype=np.float32)
    height_range = np.array([[5.0, -1.0]], dtype=np.float32)
    observed = np.ones((1, 2), dtype=np.float32)

    result = preprocess_grid_map(
        ground,
        height_range,
        observed,
        robot_z=0.0,
        max_abs_relative_height=1.5,
        max_height_range=3.0,
    )

    assert result[0, 0, 0] == pytest.approx(1.5)
    assert result[0, 0, 1] == pytest.approx(-1.5)
    assert result[1, 0, 0] == pytest.approx(3.0)
    assert result[1, 0, 1] == pytest.approx(0.0)
