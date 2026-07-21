import pytest

torch = pytest.importorskip("torch")

from go2w_terrain_planner.mapping.coordinate_transform import (
    encode_local_goal,
    relative_pose_deltas,
    warp_map_sequence,
    world_aligned_map_to_robot_frame,
)


def test_goal_encoding_in_robot_frame() -> None:
    pose = torch.tensor([[1.0, 2.0, torch.pi / 2]])
    goal = torch.tensor([[1.0, 4.0]])
    encoded = encode_local_goal(pose, goal, maximum_distance=4.0)
    assert encoded.shape == (1, 3)
    assert encoded[0, 0].item() == pytest.approx(0.5)
    assert encoded[0, 1].item() == pytest.approx(0.0, abs=1e-6)
    assert encoded[0, 2].item() == pytest.approx(1.0, abs=1e-6)


def test_relative_pose_delta_uses_previous_body_frame() -> None:
    poses = torch.tensor([[[0.0, 0.0, torch.pi / 2], [0.0, 1.0, torch.pi / 2]]])
    motion = relative_pose_deltas(poses)
    assert motion[0, 0, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert motion[0, 0, 1].item() == pytest.approx(0.0, abs=1e-6)


def test_identity_map_warp() -> None:
    maps = torch.rand((2, 3, 4, 12, 12))
    poses = torch.zeros((2, 3, 3))
    result = warp_map_sequence(maps, poses, poses[:, -1], extent_m=10.0)
    assert torch.allclose(result, maps, atol=1e-5)


def test_identity_warp_applies_vertical_reference_offset() -> None:
    maps = torch.zeros((1, 2, 4, 4, 4))
    maps[:, :, 2:] = 1.0
    poses = torch.zeros((1, 2, 3))
    result = warp_map_sequence(
        maps,
        poses,
        poses[:, -1],
        extent_m=4.0,
        source_ground_reference_z=torch.tensor([[0.0, 1.0]]),
        target_ground_reference_z=torch.tensor([1.0]),
        normalized_ground_height_scale_m=2.0,
    )
    assert torch.allclose(result[:, 0, 0], torch.full((1, 4, 4), -0.5))
    assert torch.allclose(result[:, 1, 0], torch.zeros((1, 4, 4)))


def test_height_range_warp_keeps_peak_value() -> None:
    maps = torch.zeros((1, 1, 4, 9, 9))
    maps[:, :, 1, 6, 4] = 1.0
    maps[:, :, 2:, 6, 4] = 1.0
    source_pose = torch.zeros((1, 1, 3))
    target_pose = torch.tensor([[0.0, 0.0, torch.pi / 2]])

    result = warp_map_sequence(maps, source_pose, target_pose, extent_m=9.0)

    assert result[:, :, 1].max().item() == pytest.approx(1.0)


def test_world_aligned_map_rotation_stays_finite() -> None:
    world_map = torch.zeros((4, 9, 9))
    world_map[0, 6, 4] = 1.0
    world_map[2:, 6, 4] = 1.0
    robot_map = world_aligned_map_to_robot_frame(world_map, torch.pi / 2, extent_m=9.0)
    assert robot_map.shape == world_map.shape
    assert torch.isfinite(robot_map).all()
    assert robot_map[3].sum().item() == pytest.approx(1.0)
