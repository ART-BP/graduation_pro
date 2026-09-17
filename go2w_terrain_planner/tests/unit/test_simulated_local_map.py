import pytest

torch = pytest.importorskip("torch")

from go2w_terrain_planner.mapping.simulated_local_map import (
    SimulatedLocalMap,
    SimulatedMapConfig,
    TERRAIN_NAMES,
    fuse_aligned_map_history,
)


def terrain_probabilities(count: int, **weights: float):
    result = torch.zeros((count, len(TERRAIN_NAMES)))
    for name, weight in weights.items():
        result[:, TERRAIN_NAMES.index(name)] = weight
    return result


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
    generator.reset(ids)
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


def test_reset_respects_explicit_terrain_probabilities() -> None:
    generator = SimulatedLocalMap(256, "cpu", SimulatedMapConfig(size=12))
    env_ids = torch.arange(256)

    generator.reset(
        env_ids,
        terrain_probabilities(count=256, step=0.4, wall=0.6),
    )

    assert torch.all(
        (generator.terrain_type == TERRAIN_NAMES.index("step"))
        | (generator.terrain_type == TERRAIN_NAMES.index("wall"))
    )


def test_reset_matches_requested_terrain_mixture() -> None:
    torch.manual_seed(7)
    count = 4000
    generator = SimulatedLocalMap(count, "cpu", SimulatedMapConfig(size=12))
    env_ids = torch.arange(count)

    generator.reset(
        env_ids,
        terrain_probabilities(count=count, pit=0.65, flat=0.35),
    )

    pit_ratio = (
        generator.terrain_type == TERRAIN_NAMES.index("pit")
    ).float().mean().item()
    assert pit_ratio == pytest.approx(0.65, abs=0.03)


def test_reset_rejects_probability_on_disabled_terrain() -> None:
    torch.manual_seed(11)
    count = 8
    generator = SimulatedLocalMap(
        count,
        "cpu",
        SimulatedMapConfig(size=12, enabled_terrain_names=("flat",)),
    )
    env_ids = torch.arange(count)

    with pytest.raises(ValueError, match="地形采样权重"):
        generator.reset(
            env_ids,
            terrain_probabilities(count=count, wall=1.0),
        )


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
        terrain_probabilities(count=256, wall=1.0),
        terrain_difficulty=0.0,
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
    generator.reset(
        env_ids,
        terrain_probabilities(count=count, pit=1.0),
        terrain_difficulty=0.0,
    )
    assert generator.amplitude.max().item() <= 0.120001
    assert generator.feature_width.max().item() <= 0.350001

    generator.reset(
        env_ids,
        terrain_probabilities(count=count, pit=1.0),
        terrain_difficulty=1.0,
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
        terrain_probabilities(count=count, pit=1.0),
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
    generator.reset(
        env_ids,
        terrain_probabilities(count=count, wall=1.0),
        terrain_difficulty=0.0,
    )
    assert torch.allclose(
        generator.barrier_half_width,
        torch.full_like(generator.barrier_half_width, 0.55),
    )

    generator.reset(
        env_ids,
        terrain_probabilities(count=count, wall=1.0),
        terrain_difficulty=1.0,
    )
    assert torch.allclose(
        generator.barrier_half_width,
        torch.ones_like(generator.barrier_half_width),
    )


def test_low_obstacle_is_local_and_below_collision_height() -> None:
    cfg = SimulatedMapConfig(
        size=12,
        enabled_terrain_names=("low_obstacle",),
        low_obstacle_height_range_m=(0.15, 0.15),
    )
    generator = SimulatedLocalMap(1, "cpu", cfg)
    generator.terrain_type[:] = TERRAIN_NAMES.index("low_obstacle")
    generator.amplitude[:] = 0.15
    generator.feature_x[:] = 0.0
    generator.feature_y[:] = 0.0
    generator.feature_yaw[:] = 0.0
    generator.feature_width[:] = 0.4

    _, obstacle = generator._terrain_surface(
        torch.tensor([[[0.0, 1.0]]]),
        torch.tensor([[[0.0, 0.0]]]),
    )

    assert obstacle[0, 0, 0].item() == pytest.approx(0.15)
    assert obstacle[0, 0, 1].item() == pytest.approx(0.0)


def test_mixed_terrain_contains_slope_step_and_obstacle() -> None:
    generator = SimulatedLocalMap(1, "cpu", SimulatedMapConfig(size=12))
    generator.terrain_type[:] = TERRAIN_NAMES.index("mixed")
    generator.amplitude[:] = 0.2
    generator.feature_x[:] = 0.0
    generator.feature_y[:] = 0.0
    generator.feature_yaw[:] = 0.0
    generator.barrier_half_width[:] = 1.0
    ground, obstacle = generator._terrain_surface(
        torch.tensor([[[-0.5, 0.5, 0.8]]]),
        torch.tensor([[[-0.5, -0.5, 0.5]]]),
    )

    assert ground[0, 0, 0] > 0.0
    assert ground[0, 0, 1] > ground[0, 0, 0]
    assert obstacle[0, 0, 2] > 0.0


def test_stairs_support_upward_and_downward_traversal() -> None:
    generator = SimulatedLocalMap(
        2,
        "cpu",
        SimulatedMapConfig(size=12, enabled_terrain_names=("stairs",)),
    )
    generator.terrain_type[:] = TERRAIN_NAMES.index("stairs")
    generator.amplitude[:] = 0.2
    generator.feature_width[:] = 0.5
    generator.feature_x[:] = 0.0
    generator.feature_y[:] = 0.0
    generator.feature_yaw[:] = 0.0
    generator.traversal_direction[:] = torch.tensor([1.0, -1.0])
    world_x = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])
    world_y = torch.zeros_like(world_x)

    ground, _ = generator._terrain_surface(world_x, world_y)

    assert ground[0, 1] > ground[0, 0]
    assert ground[1, 0] > ground[1, 1]


def test_domain_randomization_scale_gates_friction_and_occlusion() -> None:
    generator = SimulatedLocalMap(
        2,
        "cpu",
        SimulatedMapConfig(
            size=12,
            enabled_terrain_names=("flat",),
            friction_range=(0.4, 0.4),
            occlusion_sector_probability=1.0,
        ),
    )
    generator.reset(
        torch.arange(2),
        terrain_probabilities(2, flat=1.0),
        domain_randomization_scale=torch.tensor([0.0, 1.0]),
    )

    assert generator.friction.tolist() == pytest.approx([1.0, 0.4])
    assert generator.occlusion_enabled.tolist() == [False, True]


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
    pose = torch.tensor([[-1.0, 6.0, 0.0]])
    goal = torch.tensor([[1.0, 0.0]])

    potential = generator.navigation_potential(pose, goal)
    euclidean = torch.linalg.vector_norm(goal - pose[:, :2], dim=-1)

    assert potential.item() == pytest.approx(euclidean.item())


@pytest.mark.parametrize("terrain_index", [5, 6, 7, 8, 9])
def test_navigation_guidance_routes_around_non_traversable_geometry(
    terrain_index: int,
) -> None:
    generator = SimulatedLocalMap(
        1,
        "cpu",
        SimulatedMapConfig(size=12),
    )
    generator.terrain_type[:] = terrain_index
    generator.amplitude[:] = 0.3 if terrain_index == 5 else 0.8
    generator.feature_width[:] = 0.5
    generator.barrier_half_width[:] = 1.0
    generator.feature_x[:] = 0.0
    generator.feature_y[:] = 0.0
    generator.feature_yaw[:] = 0.0
    pose = torch.tensor([[-2.0, 0.0, 0.0]])
    goal = torch.tensor([[2.0, 0.0]])

    guidance = generator.navigation_guidance(pose, goal)

    assert guidance.blocked.item()
    assert guidance.potential_m.item() > 4.0
    assert abs(guidance.target_xy[0, 1].item()) >= 0.349


def test_shallow_pit_remains_directly_traversable() -> None:
    generator = SimulatedLocalMap(
        1,
        "cpu",
        SimulatedMapConfig(
            size=12,
            pit_navigation_avoidance_depth_m=0.18,
        ),
    )
    generator.terrain_type[:] = 5
    generator.amplitude[:] = 0.12
    generator.feature_width[:] = 0.5
    generator.feature_x[:] = 0.0
    generator.feature_y[:] = 0.0
    generator.feature_yaw[:] = 0.0
    pose = torch.tensor([[-2.0, 0.0, 0.0]])
    goal = torch.tensor([[2.0, 0.0]])

    guidance = generator.navigation_guidance(pose, goal)

    assert not guidance.blocked.item()
    assert guidance.potential_m.item() == pytest.approx(4.0)
    assert torch.equal(guidance.target_xy, goal)


def test_navigation_guidance_switches_from_entry_to_exit_corner() -> None:
    generator = SimulatedLocalMap(
        1,
        "cpu",
        SimulatedMapConfig(
            size=12,
            barrier_navigation_clearance_m=0.35,
        ),
    )
    generator.terrain_type[:] = 6
    generator.barrier_half_width[:] = 1.0
    generator.feature_x[:] = 0.0
    generator.feature_y[:] = 0.0
    generator.feature_yaw[:] = 0.0
    goal = torch.tensor([[2.0, 0.0]])

    entry_guidance = generator.navigation_guidance(
        torch.tensor([[-2.0, 0.0, 0.0]]), goal
    )
    side_guidance = generator.navigation_guidance(
        torch.tensor([[-0.40, 1.35, 0.0]]), goal
    )

    assert entry_guidance.target_xy[0, 0].item() < 0.0
    assert side_guidance.target_xy[0, 0].item() > 0.0
    assert side_guidance.potential_m < entry_guidance.potential_m


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


def test_fusion_selects_densest_consistent_ground_cluster() -> None:
    history = torch.zeros((1, 5, 4, 1, 1))
    history[:, :, 2:] = 1.0
    history[0, :, 0, 0, 0] = torch.tensor([0.00, 0.01, 0.02, 0.40, 0.41])
    history[0, :, 1, 0, 0] = torch.tensor([0.02, 0.03, 0.04, 0.30, 0.35])

    fused = fuse_aligned_map_history(
        history,
        maximum_ground_deviation=0.05,
    )

    assert fused[0, 0, 0, 0].item() == pytest.approx(0.01)
    assert fused[0, 1, 0, 0].item() == pytest.approx(0.04)


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
