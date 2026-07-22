"""Isaac Lab 2.3.2 configuration for the high-level navigation proxy."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class TerrainNavigationEnvCfg(DirectRLEnvCfg):
    """The proxy is intentionally replaceable by a calibrated Go2W asset later."""

    decimation = 5
    episode_length_s = 30.0
    action_space = 2
    observation_space = 5 * 4 * 100 * 100 + 3 + 2 + 4 * 2 + 4 * 3
    state_space = 13
    sim: SimulationCfg = SimulationCfg(dt=0.02, render_interval=decimation, device="cuda:0")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32,
        env_spacing=12.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )
    proxy_robot: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.CuboidCfg(
            size=(0.70, 0.40, 0.25),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.42, 0.80)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.20)),
    )

    map_extent_m = 10.0
    map_size = 100
    map_channels = 4
    map_history_length = 5
    command_history_length = 4
    motion_history_length = 4
    minimum_observed_ratio = 0.01
    minimum_height_valid_ratio = 0.005
    local_goal_minimum_m = 1.5
    local_goal_maximum_m = 4.0
    goal_tolerance_m = 0.50
    maximum_distance_m = 8.0
    collision_height_range_m = 0.45
    unstable_risk_threshold = 0.20
    maximum_tilt_rad = 0.65
    fall_tilt_rad = 1.00
    stuck_timeout_s = 2.0
    maximum_bad_observation_steps = 3
    curriculum_maximum_terrain_index = 9
    project_config_directory = ""

    def __post_init__(self) -> None:
        self.viewer.eye = (8.0, 8.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.sim.render_interval = self.decimation
