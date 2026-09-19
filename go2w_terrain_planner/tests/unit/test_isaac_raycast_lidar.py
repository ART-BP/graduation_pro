import pytest

torch = pytest.importorskip("torch")

from go2w_terrain_planner.mapping.isaac_raycast_lidar import (
    IsaacRaycastLidarConfig,
    IsaacRaycastLidarMapper,
    build_spinning_lidar_pattern,
    transform_lidar_rays_to_world,
)
from go2w_terrain_planner.mapping.pointcloud_grid_projector import (
    PointCloudProjectionConfig,
    PointCloudGridProjector,
)


def native_lidar_config(**overrides) -> IsaacRaycastLidarConfig:
    values = {
        "channels": 2,
        "vertical_angles_deg": (-10.0, 10.0),
        "points_per_frame": 8,
        "minimum_range_m": 0.1,
        "maximum_range_m": 6.0,
        "range_noise_std_m": 0.0,
        "height_noise_std_m": 0.0,
        "pose_xy_noise_std_m": 0.0,
        "pose_yaw_noise_std_rad": 0.0,
        "beam_dropout_probability": 0.0,
        "return_dropout_probability": 0.0,
        "projection_config": PointCloudProjectionConfig(
            input_crop_length_x_m=6.0,
            input_crop_length_y_m=6.0,
            body_height_m=0.35,
            vertical_min_offset_m=-1.5,
            vertical_max_offset_m=1.5,
        ),
    }
    values.update(overrides)
    return IsaacRaycastLidarConfig(**values)


def test_xt16_pattern_has_one_native_ray_per_firing_slot() -> None:
    vertical_angles = tuple(-15.0 + 2.0 * index for index in range(16))
    starts, directions = build_spinning_lidar_pattern(
        vertical_angles,
        0.18,
        "cpu",
    )

    assert starts.shape == (32000, 3)
    assert directions.shape == (32000, 3)
    assert torch.count_nonzero(starts).item() == 0
    assert torch.allclose(
        torch.linalg.vector_norm(directions, dim=-1),
        torch.ones(32000),
        atol=1.0e-6,
    )
    channel_elevation = torch.rad2deg(
        torch.asin(directions.reshape(16, 2000, 3)[:, 0, 2])
    )
    assert channel_elevation.tolist() == pytest.approx(
        vertical_angles, abs=1.0e-5
    )


def test_authoritative_pose_transforms_lidar_rays_without_physx_view() -> None:
    starts_world, directions_world = transform_lidar_rays_to_world(
        torch.tensor([[1.0, 0.0, 0.35], [0.0, 0.0, 0.35]]),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        torch.tensor([[10.0, 20.0, 0.20]]),
        torch.tensor([torch.pi / 2.0]),
    )

    assert starts_world.shape == (1, 2, 3)
    assert directions_world.shape == (1, 2, 3)
    assert torch.allclose(
        starts_world[0, 0],
        torch.tensor([10.0, 21.0, 0.55]),
        atol=1.0e-6,
    )
    assert torch.allclose(
        directions_world[0],
        torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
        atol=1.0e-6,
    )


def test_native_lidar_requires_one_angle_per_channel() -> None:
    with pytest.raises(ValueError, match="垂直角数量"):
        IsaacRaycastLidarMapper(
            1,
            "cpu",
            native_lidar_config(vertical_angles_deg=(-10.0,)),
        )


def test_native_lidar_requires_complete_channel_cycles() -> None:
    with pytest.raises(ValueError, match="points_per_frame"):
        IsaacRaycastLidarMapper(
            1,
            "cpu",
            native_lidar_config(points_per_frame=9),
        )


def test_pointcloud_projection_uses_robust_height_percentiles() -> None:
    projector = PointCloudGridProjector(
        PointCloudProjectionConfig(
            body_height_m=0.35,
            vertical_min_offset_m=-1.5,
            vertical_max_offset_m=1.5,
            ground_percentile=0.10,
            span_lower_percentile=0.05,
            span_upper_percentile=0.80,
        )
    )
    heights = torch.tensor(
        [0.00, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 1.20]
    )
    points = torch.stack(
        (
            torch.full_like(heights, 0.025),
            torch.full_like(heights, 0.025),
            heights,
        ),
        dim=-1,
    )[None]
    observed = torch.zeros((1, 4, 4))
    observed[:, 2, 2] = 1.0

    result = projector.project(
        points,
        observed,
        extent_m=0.2,
        size=4,
        maximum_relative_height_m=1.0,
        maximum_height_range_m=2.0,
        ground_fill_value=0.0,
        range_fill_value=0.0,
        normalize_heights=False,
    )

    assert result[0, 0, 2, 2].item() == pytest.approx(0.0)
    assert result[0, 1, 2, 2].item() < 0.1
    assert result[0, 2, 2, 2].item() == 1.0
    assert result[0, 3, 2, 2].item() == 1.0


def test_native_hits_are_transformed_and_projected_without_densification() -> None:
    mapper = IsaacRaycastLidarMapper(1, "cpu", native_lidar_config())
    cardinal_hits = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    hits = cardinal_hits.repeat(2, 1)[None]

    result = mapper.project(
        hits,
        torch.tensor([[0.0, 0.0, 0.45]]),
        torch.tensor([[0.0, 0.0, 0.20]]),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        extent_m=4.0,
        size=20,
        maximum_relative_height_m=1.5,
        maximum_height_range_m=3.0,
        ground_fill_value=0.0,
        range_fill_value=0.0,
        normalize_heights=False,
    )

    assert result.shape == (1, 4, 20, 20)
    assert torch.isfinite(result).all()
    assert mapper.last_pointcloud.shape == (1, 8, 3)
    assert mapper.last_hit_mask.sum().item() == 8
    assert result[:, 3].sum().item() == 4
    assert result[:, 2].sum().item() > result[:, 3].sum().item()
    assert torch.allclose(
        mapper.last_pointcloud[0, 0],
        torch.tensor([1.0, 0.0, 0.0]),
        atol=1.0e-6,
    )


def test_invalid_native_returns_remain_unknown_points() -> None:
    mapper = IsaacRaycastLidarMapper(1, "cpu", native_lidar_config())
    hits = torch.full((1, 8, 3), torch.inf)

    result = mapper.project(
        hits,
        torch.tensor([[0.0, 0.0, 0.45]]),
        torch.tensor([[0.0, 0.0, 0.20]]),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        extent_m=4.0,
        size=20,
        maximum_relative_height_m=1.5,
        maximum_height_range_m=3.0,
        ground_fill_value=0.0,
        range_fill_value=0.0,
        normalize_heights=False,
    )

    assert torch.isfinite(result).all()
    assert mapper.last_hit_mask.sum().item() == 0
    assert torch.isnan(mapper.last_pointcloud).all()
    # Downward rays without a return still mark traversed free space, but no
    # cell may acquire a fabricated height measurement.
    assert result[:, 2].sum().item() > 0
    assert result[:, 3].sum().item() == 0
