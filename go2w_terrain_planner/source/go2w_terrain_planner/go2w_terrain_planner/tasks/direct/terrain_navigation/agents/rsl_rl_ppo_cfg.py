"""RSL-RL 3.1.2 PPO configuration."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2wActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "Go2wActorCritic"
    init_noise_std = 0.4
    noise_std_type = "scalar"
    actor_obs_normalization = False
    critic_obs_normalization = False
    runtime_finite_checks = False
    actor_hidden_dims = [128, 128]
    critic_hidden_dims = [256, 256]
    activation = "elu"
    map_history_length = 5
    map_channels = 4
    map_size = 100
    map_encoder_channels = [24, 48, 96, 128]
    map_pool_size = 6
    map_feature_dim = 192
    temporal_hidden_dim = 192
    auxiliary_hidden_dim = 96
    fusion_hidden_dim = 256
    minimum_action_std = 0.08
    maximum_action_std = 0.6
    maximum_pre_tanh_mean = 1.5


@configclass
class Go2wPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 10000
    save_interval = 200
    experiment_name = "go2w_terrain_navigation"
    run_name = "phase6_xt16_raycast_observation"
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
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="fixed",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
    )
