import pytest

torch = pytest.importorskip("torch")

from go2w_terrain_planner.mapping.simulated_local_map import (
    SimulatedLocalMap,
    SimulatedMapConfig,
)
from go2w_terrain_planner.mapping.simulated_xt16_lidar import (
    SimulatedXt16Lidar,
    Xt16LidarConfig,
)


def lidar_config(**overrides) -> Xt16LidarConfig:
    values = {
        "horizontal_resolution_deg": 15.0,
        "maximum_range_m": 6.0,
        "ray_step_m": 0.08,
        "motion_distortion": False,
        "range_noise_std_m": 0.0,
        "height_noise_std_m": 0.0,
        "pose_xy_noise_std_m": 0.0,
        "pose_yaw_noise_std_rad": 0.0,
        "time_jitter_std_s": 0.0,
        "beam_dropout_probability": 0.0,
        "return_dropout_probability": 0.0,
    }
    values.update(overrides)
    return Xt16LidarConfig(**values)


def test_xt16_configuration_requires_one_angle_per_channel() -> None:
    with pytest.raises(ValueError, match="垂直角数量"):
        SimulatedXt16Lidar(
            1,
            "cpu",
            Xt16LidarConfig(channels=2, vertical_angles_deg=(-10.0,)),
        )


def test_raycast_flat_map_comes_from_sparse_returns_and_traversed_cells() -> None:
    cfg = SimulatedMapConfig(
        size=40,
        enabled_terrain_names=("flat",),
        observation_source="raycast",
        lidar_config=lidar_config(),
        occlusion_sector_probability=0.0,
    )
    generator = SimulatedLocalMap(2, "cpu", cfg)
    result, reference = generator.generate(
        torch.zeros((2, 3)),
        torch.zeros((2, 2)),
        return_ground_reference=True,
    )

    assert result.shape == (2, 4, 40, 40)
    assert torch.isfinite(result).all()
    assert torch.allclose(reference, torch.zeros_like(reference))
    assert torch.all(result[:, 3].sum(dim=(-2, -1)) > 0)
    assert torch.all(result[:, 2].sum(dim=(-2, -1)) > result[:, 3].sum(dim=(-2, -1)))
    valid_ground = result[:, 0][result[:, 3] > 0.5]
    assert torch.allclose(valid_ground, torch.zeros_like(valid_ground), atol=1.0e-5)
    assert generator.lidar_sensor is not None
    assert generator.lidar_sensor.last_pointcloud.shape[0] == 2
    assert torch.all(generator.lidar_sensor.last_hit_mask.sum(dim=1) > 0)


def test_raycast_wall_produces_vertical_height_spread() -> None:
    cfg = SimulatedMapConfig(
        size=100,
        enabled_terrain_names=("wall",),
        observation_source="raycast",
        lidar_config=lidar_config(
            horizontal_resolution_deg=5.0,
            ray_step_m=0.04,
        ),
        occlusion_sector_probability=0.0,
    )
    generator = SimulatedLocalMap(1, "cpu", cfg)
    generator.terrain_type[:] = 6
    generator.amplitude[:] = 1.5
    generator.feature_x[:] = 2.0
    generator.feature_y[:] = 0.0
    generator.feature_yaw[:] = 0.0
    generator.barrier_half_width[:] = 2.0

    result = generator.generate(torch.zeros((1, 3)), torch.zeros((1, 2)))

    assert result[:, 1].amax().item() > 0.05
    assert generator.lidar_sensor is not None
    points = generator.lidar_sensor.last_pointcloud[0]
    wall_points = points[torch.isfinite(points).all(dim=1) & (points[:, 0] > 1.7)]
    assert wall_points.shape[0] > 1
    assert wall_points[:, 2].amax() - wall_points[:, 2].amin() > 0.25
