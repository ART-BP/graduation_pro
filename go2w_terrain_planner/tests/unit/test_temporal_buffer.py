import pytest

torch = pytest.importorskip("torch")

from go2w_terrain_planner.mapping.temporal_grid_buffer import TemporalGridBuffer


def test_temporal_buffer_reset_push_and_alignment() -> None:
    buffer = TemporalGridBuffer(
        num_envs=2,
        history_length=5,
        channels=4,
        size=8,
        device="cpu",
        normalized_ground_height_scale_m=1.5,
    )
    initial_map = torch.rand((2, 4, 8, 8))
    pose = torch.zeros((2, 3))
    ids = torch.arange(2)
    ground_reference = torch.tensor([0.0, 0.2])
    buffer.reset(ids, initial_map, pose, ground_reference)
    buffer.push(
        initial_map,
        pose,
        torch.tensor([[0.2, 0.1], [0.0, -0.1]]),
        ground_reference,
    )
    aligned = buffer.aligned_maps(10.0)
    assert aligned.shape == (2, 5, 4, 8, 8)
    assert buffer.commands.shape == (2, 4, 2)
    assert buffer.motion_history().shape == (2, 4, 3)
    assert torch.isfinite(aligned).all()


def test_temporal_buffer_compensates_ground_reference_change() -> None:
    buffer = TemporalGridBuffer(
        num_envs=1,
        history_length=3,
        channels=4,
        size=4,
        device="cpu",
        normalized_ground_height_scale_m=2.0,
    )
    local_map = torch.zeros((1, 4, 4, 4))
    local_map[:, 2:] = 1.0
    pose = torch.zeros((1, 3))
    buffer.reset(torch.tensor([0]), local_map, pose, torch.tensor([0.0]))
    buffer.push(local_map, pose, torch.zeros((1, 2)), torch.tensor([1.0]))

    aligned = buffer.aligned_maps(4.0)

    assert torch.allclose(aligned[:, :-1, 0], torch.full_like(aligned[:, :-1, 0], -0.5))
    assert torch.allclose(aligned[:, -1, 0], torch.zeros_like(aligned[:, -1, 0]))
