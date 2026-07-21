"""RSL-RL 3.1.2 PPO configuration."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2wActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "Go2wActorCritic"
    init_noise_std = 0.5
    noise_std_type = "scalar"
    actor_obs_normalization = False
    critic_obs_normalization = False
    actor_hidden_dims = [128, 128]
    critic_hidden_dims = [128, 128]
    activation = "elu"
    map_history_length = 5
    map_channels = 4
    map_size = 100
    map_feature_dim = 128
    temporal_hidden_dim = 128
    auxiliary_hidden_dim = 64
    fusion_hidden_dim = 128


@configclass
class Go2wPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "go2w_terrain_navigation"
    run_name = "phase1_proxy"
    device = "cuda:0"
    seed = 42
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = Go2wActorCriticCfg()
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
