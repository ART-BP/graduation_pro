"""Isaac Lab 2.3.2 configuration for the high-level navigation proxy."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from go2w_terrain_planner.simulation.terrain_bank import make_terrain_importer_cfg
from go2w_terrain_planner.simulation.xt16_pattern import Xt16PatternCfg
from go2w_terrain_planner.utils.observation_layout import (
    CRITIC_OBSERVATION_DIMENSION,
    DEFAULT_POLICY_OBSERVATION_DIMENSION,
)


@configclass
class TerrainNavigationSceneCfg(InteractiveSceneCfg):
    """Physical mesh terrain, visual planner proxy, and native lidar."""

    terrain = make_terrain_importer_cfg()
    # Phase one uses an analytic high-level execution model, so this cube is a
    # visualization/sensor frame rather than a second physical robot.  Keeping
    # it out of PhysX avoids a redundant GpuRigidBodyView and its direct-GPU
    # transform writes; a later full Go2W articulation can replace it cleanly.
    robot = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.CuboidCfg(
            size=(0.80, 0.50, 0.40),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.12, 0.42, 0.80)
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.20)),
    )
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        update_period=0.10,
        # Proxy centre is 0.20 m above support ground; 0.35 m offset gives the
        # 0.55 m lidar mounting height configured for the simulated XT16.
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.35)),
        ray_alignment="base",
        pattern_cfg=Xt16PatternCfg(
            vertical_angles_deg=tuple(
                -15.0 + 2.0 * index for index in range(16)
            ),
            horizontal_resolution_deg=0.18,
        ),
        max_distance=12.0,
        mesh_prim_paths=["/World/terrain"],
        debug_vis=False,
    )


@configclass
class TerrainNavigationEnvCfg(DirectRLEnvCfg):
    """The proxy is intentionally replaceable by a calibrated Go2W asset later."""

    decimation = 5
    episode_length_s = 40.0
    action_space = 2
    observation_space = DEFAULT_POLICY_OBSERVATION_DIMENSION
    state_space = CRITIC_OBSERVATION_DIMENSION
    sim: SimulationCfg = SimulationCfg(dt=0.02, render_interval=decimation, device="cuda:0")
    scene: TerrainNavigationSceneCfg = TerrainNavigationSceneCfg(
        num_envs=32,
        env_spacing=56.0,
        replicate_physics=True,
        clone_in_fabric=False,
        # The environment calls Warp raycast_mesh once with its authoritative
        # planner pose.  Keep the RayCaster as a pattern/mesh owner, but do not
        # let InteractiveScene perform a second, redundant sensor ray cast.
        lazy_sensor_update=True,
    )

    map_extent_m = 10.0
    map_size = 200
    map_channels = 4
    actor_map_channels = 7
    command_history_length = 8
    motion_history_length = 8
    minimum_observed_ratio = 0.05
    minimum_height_valid_ratio = 0.01
    local_goal_minimum_m = 0.0
    local_goal_maximum_m = 10.0
    goal_tolerance_m = 0.35
    maximum_distance_m = 15.0
    collision_height_range_m = 0.45
    unstable_risk_threshold = 0.15
    maximum_tilt_rad = 0.75
    fall_tilt_rad = 1.00
    stuck_timeout_s = 2.0
    maximum_bad_observation_steps = 3
    curriculum_maximum_stage = 9
    # -1表示正常前沿课程；非负值用于评估时固定能力阶段范围。
    curriculum_sampling_minimum_stage = -1
    curriculum_sampling_maximum_stage = -1
    project_config_directory = ""

    def __post_init__(self) -> None:
        self.viewer.eye = (8.0, 8.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.sim.render_interval = self.decimation
