import pytest

torch = pytest.importorskip("torch")

from go2w_terrain_planner.mapping.simulated_local_map import (
    SimulatedLocalMap,
    SimulatedMapConfig,
    fuse_aligned_map_history,
)


def test_simulated_map_contract() -> None:
    generator = SimulatedLocalMap(4, "cpu", SimulatedMapConfig(size=24))
    result = generator.generate(torch.zeros((4, 3)))
    assert result.shape == (4, 4, 24, 24)
    assert torch.isfinite(result).all()
    assert result[:, 0].amin() >= -1.0
    assert result[:, 0].amax() <= 1.0
    assert result[:, 1:].amin() >= 0.0
    assert result[:, 1:].amax() <= 1.0


def test_flat_ground_is_zero_referenced() -> None:
    cfg = SimulatedMapConfig(
        size=24,
        enabled_terrain_names=("flat",),
        height_noise_std_m=0.0,
        range_noise_std_m=0.0,
        missing_probability=0.0,
        ray_only_probability=0.0,
        occlusion_sector_probability=0.0,
    )
    generator = SimulatedLocalMap(2, "cpu", cfg)

    result = generator.generate(torch.zeros((2, 3)))

    assert torch.allclose(result[:, 0], torch.zeros_like(result[:, 0]))
    assert torch.all(result[:, 3] == 1.0)


def test_task_goal_places_feature_between_start_and_goal() -> None:
    generator = SimulatedLocalMap(
        8,
        "cpu",
        SimulatedMapConfig(size=24, enabled_terrain_names=("step",)),
    )
    ids = torch.arange(8)
    generator.reset(ids, torch.full((8,), 2))
    pose = torch.zeros((8, 3))

    goal = generator.sample_task_goals(pose, 2.0, 4.0, ids)

    forward = torch.stack(
        (torch.cos(generator.route_yaw), torch.sin(generator.route_yaw)), dim=-1
    )
    feature = torch.stack((generator.feature_x, generator.feature_y), dim=-1)
    feature_along = (feature * forward).sum(dim=-1)
    goal_along = (goal * forward).sum(dim=-1)
    feature_cross = torch.abs(feature[:, 0] * -forward[:, 1] + feature[:, 1] * forward[:, 0])
    assert torch.all(feature_along > 0.0)
    assert torch.all(feature_along < goal_along)
    assert torch.all(feature_cross <= 0.081)


def test_reset_respects_terrain_sampling_range() -> None:
    generator = SimulatedLocalMap(256, "cpu", SimulatedMapConfig(size=12))
    env_ids = torch.arange(256)

    generator.reset(
        env_ids,
        maximum_terrain_index=6,
        minimum_terrain_index=2,
    )

    assert torch.all(generator.terrain_type >= 2)
    assert torch.all(generator.terrain_type <= 6)
    assert not torch.any(generator.terrain_type == 0)


def test_reset_focuses_on_curriculum_frontier_without_forgetting() -> None:
    torch.manual_seed(7)
    count = 4000
    generator = SimulatedLocalMap(count, "cpu", SimulatedMapConfig(size=12))
    env_ids = torch.arange(count)

    generator.reset(
        env_ids,
        maximum_terrain_index=5,
        minimum_terrain_index=0,
        preferred_terrain_index=5,
        preferred_probability=0.65,
    )

    frontier_ratio = (generator.terrain_type == 5).float().mean().item()
    assert frontier_ratio == pytest.approx(0.65, abs=0.03)
    assert torch.all(generator.terrain_type <= 5)
    assert torch.any(generator.terrain_type < 5)


def test_reset_mixes_replay_frontier_and_near_future_challenges() -> None:
    torch.manual_seed(11)
    count = 6000
    generator = SimulatedLocalMap(count, "cpu", SimulatedMapConfig(size=12))
    env_ids = torch.arange(count)

    generator.reset(
        env_ids,
        maximum_terrain_index=9,
        minimum_terrain_index=0,
        preferred_terrain_index=5,
        preferred_probability=0.55,
        challenge_probability=0.15,
        challenge_level_span=2,
    )

    terrain = generator.terrain_type
    assert (terrain == 5).float().mean().item() == pytest.approx(0.55, abs=0.03)
    assert (terrain < 5).float().mean().item() == pytest.approx(0.30, abs=0.03)
    assert ((terrain == 6) | (terrain == 7)).float().mean().item() == pytest.approx(
        0.15, abs=0.03
    )
    assert not torch.any(terrain > 7)


def test_future_challenge_uses_narrower_barrier_geometry() -> None:
    cfg = SimulatedMapConfig(
        size=12,
        enabled_terrain_names=("pit", "wall"),
        barrier_half_width_range_m=(1.0, 1.0),
        challenge_barrier_width_scale=0.65,
    )
    generator = SimulatedLocalMap(256, "cpu", cfg)
    env_ids = torch.arange(256)

    generator.reset(
        env_ids,
        maximum_terrain_index=6,
        preferred_terrain_index=5,
        preferred_probability=0.0,
        challenge_probability=1.0,
        challenge_level_span=1,
    )

    assert torch.all(generator.terrain_type == 6)
    assert torch.allclose(
        generator.barrier_half_width,
        torch.full_like(generator.barrier_half_width, 0.65),
    )


def test_pit_geometry_progresses_from_easy_to_full_range() -> None:
    torch.manual_seed(19)
    count = 1024
    cfg = SimulatedMapConfig(
        size=12,
        enabled_terrain_names=("pit",),
        pit_depth_range_m=(0.05, 0.40),
        pit_half_width_range_m=(0.25, 0.60),
        pit_curriculum_start_depth_max_m=0.12,
        pit_curriculum_start_half_width_max_m=0.35,
    )
    generator = SimulatedLocalMap(count, "cpu", cfg)
    env_ids = torch.arange(count)
    preferred = torch.full((count,), 5)

    generator.reset(
        env_ids,
        maximum_terrain_index=5,
        minimum_terrain_index=5,
        preferred_terrain_index=preferred,
        preferred_probability=1.0,
        preferred_difficulty=0.0,
    )
    assert generator.amplitude.max().item() <= 0.120001
    assert generator.feature_width.max().item() <= 0.350001

    generator.reset(
        env_ids,
        maximum_terrain_index=5,
        minimum_terrain_index=5,
        preferred_terrain_index=preferred,
        preferred_probability=1.0,
        preferred_difficulty=1.0,
    )
    assert generator.amplitude.max().item() > 0.35
    assert generator.feature_width.max().item() > 0.55


def test_pit_goal_is_beyond_far_edge_with_clearance() -> None:
    count = 128
    cfg = SimulatedMapConfig(
        size=12,
        enabled_terrain_names=("pit",),
        pit_goal_clearance_m=0.8,
    )
    generator = SimulatedLocalMap(count, "cpu", cfg)
    env_ids = torch.arange(count)
    generator.reset(
        env_ids,
        maximum_terrain_index=5,
        minimum_terrain_index=5,
    )
    generator.feature_width[:] = 0.6
    pose = torch.zeros((count, 3))

    goal = generator.sample_task_goals(pose, 1.5, 4.0, env_ids)
    forward = torch.stack(
        (torch.cos(generator.route_yaw), torch.sin(generator.route_yaw)), dim=-1
    )
    feature = torch.stack((generator.feature_x, generator.feature_y), dim=-1)
    feature_along = (feature * forward).sum(dim=-1)
    goal_along = (goal * forward).sum(dim=-1)

    assert torch.all(
        goal_along - feature_along - generator.feature_width
        >= cfg.pit_goal_clearance_m - 1.0e-5
    )


def test_wall_width_leaves_bounded_detour_corridor() -> None:
    cfg = SimulatedMapConfig(
        size=20,
        enabled_terrain_names=("wall",),
        barrier_half_width_range_m=(1.0, 1.0),
    )
    generator = SimulatedLocalMap(1, "cpu", cfg)
    generator.terrain_type[:] = 6
    generator.amplitude[:] = 0.8
    generator.feature_x[:] = 0.0
    generator.feature_y[:] = 0.0
    generator.feature_yaw[:] = 0.0

    _, obstacle = generator._terrain_surface(
        torch.tensor([[[0.0, 0.0]]]),
        torch.tensor([[[0.5, 1.1]]]),
    )

    assert obstacle[0, 0, 0].item() == pytest.approx(0.8)
    assert obstacle[0, 0, 1].item() == pytest.approx(0.0)


def test_wall_frontier_width_grows_with_intra_level_difficulty() -> None:
    count = 64
    cfg = SimulatedMapConfig(
        size=12,
        enabled_terrain_names=("wall",),
        barrier_half_width_range_m=(1.0, 1.0),
        challenge_barrier_width_scale=0.55,
    )
    generator = SimulatedLocalMap(count, "cpu", cfg)
    env_ids = torch.arange(count)
    preferred = torch.full((count,), 6)

    generator.reset(
        env_ids,
        maximum_terrain_index=6,
        minimum_terrain_index=6,
        preferred_terrain_index=preferred,
        preferred_probability=1.0,
        preferred_difficulty=0.0,
    )
    assert torch.allclose(
        generator.barrier_half_width,
        torch.full_like(generator.barrier_half_width, 0.55),
    )

    generator.reset(
        env_ids,
        maximum_terrain_index=6,
        minimum_terrain_index=6,
        preferred_terrain_index=preferred,
        preferred_probability=1.0,
        preferred_difficulty=1.0,
    )
    assert torch.allclose(
        generator.barrier_half_width,
        torch.ones_like(generator.barrier_half_width),
    )


def test_wall_navigation_potential_rewards_lateral_detour() -> None:
    cfg = SimulatedMapConfig(
        size=12,
        enabled_terrain_names=("wall",),
        barrier_navigation_clearance_m=0.35,
    )
    generator = SimulatedLocalMap(1, "cpu", cfg)
    generator.terrain_type[:] = 6
    generator.feature_x[:] = 0.0
    generator.feature_y[:] = 0.0
    generator.feature_yaw[:] = 0.0
    generator.barrier_half_width[:] = 1.0
    goal = torch.tensor([[1.0, 0.0]])

    direct_pose = torch.tensor([[-1.0, 0.0, 0.0]])
    lateral_pose = torch.tensor([[-1.0, 0.4, 0.0]])
    direct_euclidean = torch.linalg.vector_norm(
        goal - direct_pose[:, :2], dim=-1
    )
    lateral_euclidean = torch.linalg.vector_norm(
        goal - lateral_pose[:, :2], dim=-1
    )
    direct_potential = generator.navigation_potential(direct_pose, goal)
    lateral_potential = generator.navigation_potential(lateral_pose, goal)

    assert direct_potential > direct_euclidean
    assert lateral_euclidean > direct_euclidean
    assert lateral_potential < direct_potential


def test_navigation_potential_returns_to_euclidean_after_wall_is_cleared() -> None:
    cfg = SimulatedMapConfig(
        size=12,
        enabled_terrain_names=("wall",),
        barrier_navigation_clearance_m=0.35,
    )
    generator = SimulatedLocalMap(1, "cpu", cfg)
    generator.terrain_type[:] = 6
    generator.feature_x[:] = 0.0
    generator.feature_y[:] = 0.0
    generator.feature_yaw[:] = 0.0
    generator.barrier_half_width[:] = 1.0
    pose = torch.tensor([[-1.0, 3.0, 0.0]])
    goal = torch.tensor([[1.0, 0.0]])

    potential = generator.navigation_potential(pose, goal)
    euclidean = torch.linalg.vector_norm(goal - pose[:, :2], dim=-1)

    assert potential.item() == pytest.approx(euclidean.item())


def test_truth_query_is_independent_from_actor_observation() -> None:
    generator = SimulatedLocalMap(
        1,
        "cpu",
        SimulatedMapConfig(size=24, enabled_terrain_names=("wall",)),
    )
    generator.terrain_type[:] = 6
    generator.amplitude[:] = 0.8
    generator.feature_x[:] = 0.35
    generator.feature_y[:] = 0.0
    generator.feature_yaw[:] = 0.0
    pose = torch.zeros((1, 3))

    truth = generator.true_motion_metrics(pose, torch.tensor([0.2]))
    missing_actor_map = torch.zeros((1, 4, 24, 24))
    observed_risk, _ = generator.forward_risk(missing_actor_map, torch.tensor([0.2]))

    assert truth.obstacle_height_m.item() == pytest.approx(0.8)
    assert truth.collision_height_m.item() == pytest.approx(0.8)
    assert observed_risk.item() == pytest.approx(0.0)


def test_height_range_uses_within_cell_surface_span() -> None:
    cfg = SimulatedMapConfig(size=20, extent_m=10.0, enabled_terrain_names=("ramp",))
    generator = SimulatedLocalMap(1, "cpu", cfg)
    generator.terrain_type[:] = 1
    generator.amplitude[:] = 0.2
    generator.feature_x[:] = 0.0
    generator.feature_y[:] = 0.0
    generator.feature_yaw[:] = 0.0

    _, height_range = generator._height_and_obstacle_range(
        torch.tensor([[[1.0]]]), torch.tensor([[[0.0]]])
    )

    expected_span = 0.2 * cfg.extent_m / cfg.size
    assert height_range.item() == pytest.approx(expected_span, abs=1.0e-6)


def test_stairs_use_single_riser_for_collision_height() -> None:
    count = 7
    generator = SimulatedLocalMap(
        count,
        "cpu",
        SimulatedMapConfig(size=20, enabled_terrain_names=("stairs",)),
    )
    generator.terrain_type[:] = 3
    generator.amplitude[:] = 0.35
    generator.feature_width[:] = 0.25
    generator.feature_x[:] = 0.0
    generator.feature_y[:] = 0.0
    generator.feature_yaw[:] = 0.0
    pose = torch.zeros((count, 3))
    pose[:, 0] = torch.linspace(0.2, 0.8, count)

    truth = generator.true_motion_metrics(pose, torch.full((count,), 0.5))

    assert truth.support_span_m.max().item() > 0.45
    assert truth.maximum_discontinuity_m.max().item() == pytest.approx(0.1575)
    assert truth.collision_height_m.max().item() < 0.45


def test_subset_generation_returns_only_requested_environments() -> None:
    generator = SimulatedLocalMap(4, "cpu", SimulatedMapConfig(size=16))
    env_ids = torch.tensor([1, 3])
    local_map, reference = generator.generate(
        torch.zeros((2, 3)),
        torch.zeros((2, 2)),
        return_ground_reference=True,
        env_ids=env_ids,
    )

    assert local_map.shape == (2, 4, 16, 16)
    assert reference.shape == (2,)


def test_generate_returns_map_ground_reference() -> None:
    cfg = SimulatedMapConfig(
        size=24,
        enabled_terrain_names=("step",),
        height_noise_std_m=0.0,
        range_noise_std_m=0.0,
        missing_probability=0.0,
        ray_only_probability=0.0,
        occlusion_sector_probability=0.0,
        pose_xy_noise_std_m=0.0,
        pose_yaw_noise_std_rad=0.0,
    )
    generator = SimulatedLocalMap(1, "cpu", cfg)
    generator.terrain_type[:] = 2
    generator.amplitude[:] = 0.25
    generator.feature_x[:] = 0.0
    generator.feature_y[:] = 0.0
    generator.feature_yaw[:] = 0.0
    pose = torch.tensor([[0.5, 0.0, 0.0]])

    local_map, reference = generator.generate(pose, return_ground_reference=True)

    assert reference.item() == pytest.approx(0.25)
    center = cfg.size // 2
    assert local_map[0, 0, center, center].item() == pytest.approx(0.0, abs=1.0e-6)


def test_fusion_preserves_observed_without_inventing_height() -> None:
    history = torch.zeros((1, 3, 4, 2, 2))
    history[0, 0, 0, 0, 0] = 0.3
    history[0, 0, 1, 0, 0] = 0.2
    history[0, 0, 2, 0, 0] = 1.0
    history[0, 0, 3, 0, 0] = 1.0
    history[0, 2, 2, 1, 1] = 1.0
    fused = fuse_aligned_map_history(history)
    assert fused[0, 3, 0, 0] == 1.0
    assert fused[0, 0, 0, 0].item() == pytest.approx(0.3)
    assert fused[0, 2, 1, 1] == 1.0
    assert fused[0, 3, 1, 1] == 0.0


def test_fusion_rejects_single_frame_height_range_outlier() -> None:
    history = torch.zeros((1, 5, 4, 1, 1))
    history[:, :, 2:] = 1.0
    history[0, :, 0, 0, 0] = torch.tensor([0.1, 0.11, 0.09, 0.12, 0.1])
    history[0, :, 1, 0, 0] = torch.tensor([0.02, 0.03, 0.8, 0.04, 0.05])

    fused = fuse_aligned_map_history(history)

    assert fused[0, 0, 0, 0].item() == pytest.approx(0.1)
    assert fused[0, 1, 0, 0].item() == pytest.approx(0.05)


def test_fusion_retains_height_observed_more_than_ten_updates_ago() -> None:
    history = torch.zeros((1, 12, 4, 1, 1))
    history[0, 0, 0, 0, 0] = 0.25
    history[0, 0, 1, 0, 0] = 0.08
    history[0, 0, 2, 0, 0] = 1.0
    history[0, 0, 3, 0, 0] = 1.0

    fused = fuse_aligned_map_history(history)

    assert fused[0, 0, 0, 0].item() == pytest.approx(0.25)
    assert fused[0, 1, 0, 0].item() == pytest.approx(0.08)
    assert fused[0, 2, 0, 0].item() == pytest.approx(1.0)
    assert fused[0, 3, 0, 0].item() == pytest.approx(1.0)
