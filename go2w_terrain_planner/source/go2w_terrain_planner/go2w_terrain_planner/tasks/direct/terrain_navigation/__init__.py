"""Registration of the phase-one Go2W terrain-navigation task."""

import importlib.util
import os


if (
    os.environ.get("GO2W_SKIP_TASK_REGISTRATION") != "1"
    and importlib.util.find_spec("isaaclab_tasks") is not None
):
    import gymnasium as gym

    from . import agents

    gym.register(
        id="Go2W-Terrain-Navigation-Direct-v0",
        entry_point=f"{__name__}.terrain_navigation_env:TerrainNavigationEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.terrain_navigation_env_cfg:TerrainNavigationEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2wPPORunnerCfg",
        },
    )
