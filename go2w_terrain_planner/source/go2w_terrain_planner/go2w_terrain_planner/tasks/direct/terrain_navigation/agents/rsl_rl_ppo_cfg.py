"""RSL-RL 3.1.2 PPO configuration."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2wActorCriticCfg(RslRlPpoActorCriticCfg):
    """网络/策略模型配置"""
    class_name = "Go2wActorCritic"
    init_noise_std = 0.4
    noise_std_type = "scalar"
    actor_obs_normalization = False
    critic_obs_normalization = False
    runtime_finite_checks = False
    actor_hidden_dims = [128, 128]
    critic_hidden_dims = [256, 256]
    activation = "elu"
    map_channels = 7
    map_size = 200
    command_history_length = 8
    motion_history_length = 8
    architecture = "compact_map_motion_gru_v7"
    map_encoder_channels = [32, 64, 96, 128]
    map_pool_size = 8
    map_feature_dim = 384
    motion_gru_hidden_dim = 128
    auxiliary_hidden_dim = 128
    fusion_hidden_dim = 384
    minimum_action_std = 0.08
    maximum_action_std = 0.6
    maximum_pre_tanh_mean = 1.5


@configclass
class Go2wPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """整个训练运行器配置"""
    num_steps_per_env = 96
    max_iterations = 10000
    save_interval = 200
    experiment_name = "go2w_terrain_navigation"
    run_name = "phase13_raycast_reward_diversity"
    device = "cuda:0"
    seed = 42
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = Go2wActorCriticCfg()
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.001,
        num_learning_epochs=3,
        num_mini_batches=8,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.997,
        lam=0.98,
        desired_kl=0.01,
        max_grad_norm=0.5,
    )
