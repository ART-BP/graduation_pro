"""RSL-RL 3.1 compatible CNN-GRU actor-critic for flattened policy observations."""

from __future__ import annotations

import os
import warnings
from typing import Any, NoReturn

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from go2w_terrain_planner.utils.tensor_checks import runtime_checks_enabled

from .map_encoder import MapEncoder
from .temporal_encoder import TemporalEncoder


class Go2wActorCritic(nn.Module):
    """Custom policy used through a small RSL-RL 3.1 runner registration hook."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        *,
        map_history_length: int = 5,
        map_channels: int = 4,
        map_size: int = 100,
        map_encoder_channels: list[int] | tuple[int, ...] = (24, 48, 96, 128),
        map_pool_size: int = 6,
        map_feature_dim: int = 192,
        temporal_hidden_dim: int = 192,
        auxiliary_hidden_dim: int = 96,
        fusion_hidden_dim: int = 256,
        critic_hidden_dims: list[int] | tuple[int, ...] = (256, 256),
        init_noise_std: float = 0.4,
        noise_std_type: str = "scalar",
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        runtime_finite_checks: bool = False,
        minimum_action_std: float = 0.08,
        maximum_action_std: float = 0.6,
        maximum_pre_tanh_mean: float = 1.5,
        **kwargs: Any,
    ) -> None:
        del kwargs, actor_obs_normalization, critic_obs_normalization
        super().__init__()
        if noise_std_type != "scalar":
            raise ValueError("第一版Go2wActorCritic仅支持scalar动作标准差")
        if not (
            0.0 < minimum_action_std < init_noise_std < maximum_action_std
            and maximum_pre_tanh_mean > 0.0
        ):
            raise ValueError("动作分布参数范围无效")
        self.obs_groups = obs_groups
        self.minimum_action_std = float(minimum_action_std)
        self.maximum_action_std = float(maximum_action_std)
        self.initial_action_std = float(init_noise_std)
        self.maximum_pre_tanh_mean = float(maximum_pre_tanh_mean)
        self.runtime_finite_checks = bool(runtime_finite_checks) or (
            "GO2W_RUNTIME_CHECKS" in os.environ and runtime_checks_enabled()
        )
        self.map_history_length = map_history_length
        self.map_channels = map_channels
        self.map_size = map_size
        self.map_flat_dim = map_history_length * map_channels * map_size * map_size
        actor_dim = sum(obs[name].shape[-1] for name in obs_groups["policy"])
        if actor_dim <= self.map_flat_dim:
            raise ValueError(f"policy观测维度{actor_dim}不足以容纳地图{self.map_flat_dim}")
        self.auxiliary_dim = actor_dim - self.map_flat_dim

        self.map_encoder = MapEncoder(
            map_channels,
            map_feature_dim,
            encoder_channels=map_encoder_channels,
            spatial_pool_size=map_pool_size,
        )
        self.temporal_encoder = TemporalEncoder(map_feature_dim, temporal_hidden_dim)
        self.auxiliary_encoder = nn.Sequential(
            nn.Linear(self.auxiliary_dim, auxiliary_hidden_dim),
            nn.ELU(),
            nn.Linear(auxiliary_hidden_dim, auxiliary_hidden_dim),
            nn.ELU(),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(temporal_hidden_dim + auxiliary_hidden_dim, fusion_hidden_dim),
            nn.ELU(),
            nn.Linear(fusion_hidden_dim, num_actions),
        )

        critic_dim = sum(obs[name].shape[-1] for name in obs_groups["critic"])
        critic_layers: list[nn.Module] = []
        previous = critic_dim
        for hidden in critic_hidden_dims:
            critic_layers.extend((nn.Linear(previous, hidden), nn.ELU()))
            previous = hidden
        critic_layers.append(nn.Linear(previous, 1))
        self.critic = nn.Sequential(*critic_layers)
        # ``std``保存无约束logit，实际标准差由sigmoid平滑映射到配置范围。
        # 保留参数名是为了让旧checkpoint能够在load_state_dict中显式迁移。
        initial_std_logit = self._physical_std_to_logit(
            torch.tensor(init_noise_std, dtype=torch.float32)
        )
        self.std = nn.Parameter(initial_std_logit.repeat(num_actions))
        self.register_buffer(
            "_action_std_parameterization_version",
            torch.tensor(2, dtype=torch.int64),
        )
        self.register_buffer(
            "_policy_architecture_version",
            torch.tensor(2, dtype=torch.int64),
        )
        self.distribution: Normal | None = None
        self._pre_tanh_action: torch.Tensor | None = None
        Normal.set_default_validate_args(False)

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def _group(self, obs, group: str) -> torch.Tensor:
        values = [obs[name] for name in self.obs_groups[group]]
        return values[0] if len(values) == 1 else torch.cat(values, dim=-1)

    def _check_finite_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        module: nn.Module | None = None,
    ) -> None:
        """Raise a detailed error when a tensor first becomes non-finite."""

        # Do not inject diagnostic Python control flow into ONNX tracing.
        if torch.jit.is_tracing() or not self.runtime_finite_checks:
            return

        finite_mask = torch.isfinite(tensor)

        if bool(finite_mask.all().item()):
            return

        bad_mask = ~finite_mask
        bad_count = int(bad_mask.sum().item())

        first_bad_indices = (
            torch.nonzero(bad_mask, as_tuple=False)[:8]
            .detach()
            .cpu()
            .tolist()
        )

        finite_values = tensor[finite_mask]

        if finite_values.numel() > 0:
            finite_min = float(finite_values.min().item())
            finite_max = float(finite_values.max().item())
            finite_max_abs = float(finite_values.abs().max().item())
        else:
            finite_min = float("nan")
            finite_max = float("nan")
            finite_max_abs = float("nan")

        bad_parameters: list[str] = []

        if module is not None:
            for parameter_name, parameter in module.named_parameters():
                if not bool(torch.isfinite(parameter).all().item()):
                    bad_parameters.append(parameter_name)

        raise RuntimeError(
            f"Actor张量出现NaN或Inf: stage={name}, "
            f"shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, "
            f"device={tensor.device}, "
            f"bad_count={bad_count}, "
            f"first_bad_indices={first_bad_indices}, "
            f"finite_min={finite_min}, "
            f"finite_max={finite_max}, "
            f"finite_max_abs={finite_max_abs}, "
            f"bad_parameters={bad_parameters}"
        )

    def _actor_raw_mean(self, actor_obs: torch.Tensor) -> torch.Tensor:
        self._check_finite_tensor(
            "policy_observation",
            actor_obs,
        )

        batch = actor_obs.shape[0]

        map_observation = actor_obs[:, : self.map_flat_dim]
        auxiliary = actor_obs[:, self.map_flat_dim :]

        self._check_finite_tensor(
            "map_observation",
            map_observation,
        )
        self._check_finite_tensor(
            "auxiliary_observation",
            auxiliary,
        )

        maps = map_observation.reshape(
            batch,
            self.map_history_length,
            self.map_channels,
            self.map_size,
            self.map_size,
        )

        map_encoder_input = maps.reshape(
            -1,
            self.map_channels,
            self.map_size,
            self.map_size,
        )

        self._check_finite_tensor(
            "map_encoder_input",
            map_encoder_input,
        )

        per_frame = self.map_encoder(map_encoder_input)

        self._check_finite_tensor(
            "map_encoder_output",
            per_frame,
            self.map_encoder,
        )

        per_frame = per_frame.reshape(
            batch,
            self.map_history_length,
            -1,
        )

        self._check_finite_tensor(
            "temporal_encoder_input",
            per_frame,
        )

        temporal = self.temporal_encoder(per_frame)

        self._check_finite_tensor(
            "temporal_encoder_output",
            temporal,
            self.temporal_encoder,
        )

        auxiliary_feature = self.auxiliary_encoder(auxiliary)

        self._check_finite_tensor(
            "auxiliary_encoder_output",
            auxiliary_feature,
            self.auxiliary_encoder,
        )

        fused_feature = torch.cat(
            (temporal, auxiliary_feature),
            dim=-1,
        )

        self._check_finite_tensor(
            "actor_fused_feature",
            fused_feature,
        )

        unbounded_mean = self.actor_head(fused_feature)

        self._check_finite_tensor(
            "actor_head_unbounded_output",
            unbounded_mean,
            self.actor_head,
        )

        # 将高斯均值限制在tanh饱和区之前。否则极大的均值会使动作舍入为
        # 精确的±1，PPO回放时atanh无法恢复原始样本并导致概率比爆炸。
        mean = self.maximum_pre_tanh_mean * torch.tanh(
            unbounded_mean / self.maximum_pre_tanh_mean
        )
        self._check_finite_tensor("actor_head_bounded_output", mean)
        return mean

    def _actor_forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self._actor_raw_mean(actor_obs))

    def _physical_std_to_logit(self, physical_std: torch.Tensor) -> torch.Tensor:
        fraction = (
            (physical_std - self.minimum_action_std)
            / (self.maximum_action_std - self.minimum_action_std)
        ).clamp(1.0e-6, 1.0 - 1.0e-6)
        return torch.logit(fraction)

    def _bounded_action_std(self) -> torch.Tensor:
        return self.minimum_action_std + (
            self.maximum_action_std - self.minimum_action_std
        ) * torch.sigmoid(self.std)

    def _update_distribution(self, obs) -> None:
        policy_observation = self._group(obs, "policy")

        mean = self._actor_raw_mean(policy_observation)

        std = self._bounded_action_std().expand_as(mean)

        self._check_finite_tensor(
            "action_std",
            std,
        )

        self.distribution = Normal(mean, std)
        self._pre_tanh_action = None

    def act(self, obs, **kwargs) -> torch.Tensor:
        del kwargs
        self._update_distribution(obs)
        # PPO采样阶段由runner的inference_mode隔离计算图；更新阶段使用
        # rsample使squashed-Gaussian熵中的雅可比项能反传到均值和标准差。
        self._pre_tanh_action = self.distribution.rsample()
        return torch.tanh(self._pre_tanh_action)

    def act_inference(self, obs) -> torch.Tensor:
        return self._actor_forward(self._group(obs, "policy"))

    def evaluate(self, obs, **kwargs) -> torch.Tensor:
        del kwargs
        return self.critic(self._group(obs, "critic"))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        bounded = torch.clamp(actions, min=-1.0 + 1.0e-5, max=1.0 - 1.0e-5)
        pre_tanh = torch.atanh(bounded)
        correction = self._tanh_log_abs_det_jacobian(pre_tanh)
        return (self.distribution.log_prob(pre_tanh) - correction).sum(dim=-1)

    @staticmethod
    def _tanh_log_abs_det_jacobian(pre_tanh: torch.Tensor) -> torch.Tensor:
        """Numerically stable ``log(1 - tanh(x)^2)``."""
        return 2.0 * (
            0.6931471805599453
            - pre_tanh
            - F.softplus(-2.0 * pre_tanh)
        )

    @property
    def action_mean(self) -> torch.Tensor:
        # RSL-RL stores this value together with ``action_std`` and uses both
        # to estimate Gaussian KL divergence.  The KL must be evaluated in
        # the pre-tanh variable space;
        # the environment still receives the bounded sample returned by act().
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self._pre_tanh_action is None:
            return self.distribution.entropy().sum(dim=-1)
        correction = self._tanh_log_abs_det_jacobian(self._pre_tanh_action)
        return (self.distribution.entropy() + correction).sum(dim=-1)

    def update_normalization(self, obs) -> None:
        del obs

    def load_state_dict(self, state_dict, strict: bool = True):
        migrated = state_dict.copy()
        version_key = "_action_std_parameterization_version"
        if version_key not in migrated:
            # 第一、二版checkpoint中的``std``直接表示物理标准差。旧训练
            # 常把它锁死在上限，因此迁移时有意重置探索强度，而不继承坏状态。
            if "std" in migrated:
                migrated["std"] = self.std.detach().clone()
            migrated[version_key] = self._action_std_parameterization_version.clone()
            warnings.warn(
                "旧动作标准差checkpoint已迁移；探索标准差重置为当前配置初值",
                RuntimeWarning,
                stacklevel=2,
            )
        return super().load_state_dict(migrated, strict=strict)


class ActorExportWrapper(nn.Module):
    """ONNX wrapper that emits deployable physical ``[v_cmd,w_cmd]`` values."""

    def __init__(
        self,
        actor_critic: Go2wActorCritic,
        action_min: tuple[float, float],
        action_max: tuple[float, float],
    ) -> None:
        super().__init__()
        self.actor_critic = actor_critic
        self.actor_critic.map_encoder.prepare_for_onnx_export()
        self.register_buffer("action_min", torch.tensor(action_min, dtype=torch.float32))
        self.register_buffer("action_max", torch.tensor(action_max, dtype=torch.float32))

    def forward(self, policy_observation: torch.Tensor) -> torch.Tensor:
        normalized = self.actor_critic._actor_forward(policy_observation)
        return self.action_min + 0.5 * (normalized + 1.0) * (self.action_max - self.action_min)
