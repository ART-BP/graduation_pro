import pytest

torch = pytest.importorskip("torch")

from go2w_terrain_planner.mapping.coordinate_transform import encode_local_goal
from go2w_terrain_planner.mapping.grid_preprocessor import downsample_map_tensor
from go2w_terrain_planner.mapping.simulated_local_map import (
    SimulatedLocalMap,
    SimulatedMapConfig,
    fuse_aligned_map_history,
)
from go2w_terrain_planner.mapping.temporal_grid_buffer import TemporalGridBuffer
from go2w_terrain_planner.models import Go2wActorCritic
from go2w_terrain_planner.tasks.direct.terrain_navigation.observations import (
    assemble_policy_observation,
)


def test_phase1_tensor_pipeline_contract() -> None:
    batch, source_size, output_size, history = 2, 32, 16, 5
    height_scale = 1.5
    generator = SimulatedLocalMap(
        batch,
        "cpu",
        SimulatedMapConfig(
            size=source_size,
            maximum_relative_height_m=height_scale,
            enabled_terrain_names=("step",),
            height_noise_std_m=0.0,
            range_noise_std_m=0.0,
            missing_probability=0.0,
            ray_only_probability=0.0,
            occlusion_sector_probability=0.0,
            pose_xy_noise_std_m=0.0,
            pose_yaw_noise_std_rad=0.0,
        ),
    )
    env_ids = torch.arange(batch)
    generator.reset(env_ids, torch.full((batch,), 2))
    pose = torch.zeros((batch, 3))
    goals = generator.sample_task_goals(pose, 2.0, 3.0, env_ids)
    pose[:, 2] = generator.route_yaw
    velocity = torch.tensor([[0.2, 0.0], [0.3, 0.1]])
    raw_map, reference = generator.generate(
        pose, velocity, return_ground_reference=True
    )
    local_map = downsample_map_tensor(raw_map, output_size)

    fusion = TemporalGridBuffer(
        batch,
        3,
        4,
        output_size,
        "cpu",
        normalized_ground_height_scale_m=height_scale,
    )
    fusion.reset(env_ids, local_map, pose, reference)
    fusion.push(local_map, pose, velocity, reference)
    fused_map = fuse_aligned_map_history(fusion.aligned_maps(10.0))

    temporal = TemporalGridBuffer(
        batch,
        history,
        4,
        output_size,
        "cpu",
        normalized_ground_height_scale_m=height_scale,
    )
    temporal.reset(env_ids, fused_map, pose, reference)
    temporal.push(fused_map, pose, velocity, reference)
    aligned_maps = temporal.aligned_maps(10.0)
    goal = encode_local_goal(pose, goals, 4.0)
    expected_dimension = history * 4 * output_size * output_size + 3 + 2 + 4 * 2 + 4 * 3
    policy_observation = assemble_policy_observation(
        aligned_maps,
        goal,
        velocity,
        temporal.commands,
        temporal.motion_history(),
        expected_dimension=expected_dimension,
    )

    observations = {
        "policy": policy_observation,
        "critic": torch.zeros((batch, 13)),
    }
    actor_critic = Go2wActorCritic(
        observations,
        {"policy": ["policy"], "critic": ["critic"]},
        2,
        map_history_length=history,
        map_channels=4,
        map_size=output_size,
        critic_hidden_dims=[16],
    )
    actions = actor_critic.act_inference(observations)

    assert policy_observation.shape == (batch, expected_dimension)
    assert actions.shape == (batch, 2)
    assert torch.isfinite(actions).all()
