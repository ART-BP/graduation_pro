"""Global capability curriculum and its nine-stage task schedule."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CurriculumTaskBatch:
    """Per-environment tasks sampled from the shared capability frontier."""

    stages: object
    terrain_probabilities: object
    curriculum_difficulty: object
    geometry_difficulty: object
    goal_minimum_m: object
    goal_maximum_m: object
    heading_half_range_rad: object
    domain_randomization_scale: object


class CurriculumStageSchedule:
    """Turn capability stages into terrain, goal and randomization tasks.

    A curriculum stage is deliberately not a terrain index.  One stage may
    mix several terrain primitives, while the same primitive may reappear in
    later stages at a greater geometric difficulty.
    """

    def __init__(self, stage_configs: dict, terrain_names, device) -> None:
        import torch

        if not isinstance(stage_configs, dict) or not stage_configs:
            raise ValueError("课程阶段配置不能为空")
        self.device = torch.device(device)
        self.terrain_names = tuple(terrain_names)
        self.minimum_stage = min(int(stage) for stage in stage_configs)
        self.maximum_stage = max(int(stage) for stage in stage_configs)
        expected_stages = set(range(self.minimum_stage, self.maximum_stage + 1))
        if {int(stage) for stage in stage_configs} != expected_stages:
            raise ValueError("课程阶段必须是连续整数")

        count = self.maximum_stage + 1
        self.terrain_weights = torch.zeros(
            (count, len(self.terrain_names)), dtype=torch.float32, device=self.device
        )
        self.goal_minimum_m = torch.zeros(count, device=self.device)
        self.goal_maximum_m = torch.zeros(count, device=self.device)
        self.heading_half_range_rad = torch.zeros(count, device=self.device)
        self.domain_randomization_scale = torch.zeros(count, device=self.device)
        self.geometry_difficulty_range = torch.zeros(
            (count, 2), device=self.device
        )
        self.has_progressive_difficulty = torch.zeros(
            count, dtype=torch.bool, device=self.device
        )
        self.stage_names: dict[int, str] = {}

        terrain_to_index = {
            name: index for index, name in enumerate(self.terrain_names)
        }
        for stage_key, values in stage_configs.items():
            stage = int(stage_key)
            self.stage_names[stage] = str(values["name"])
            for terrain_name, weight in values["terrain_weights"].items():
                self.terrain_weights[stage, terrain_to_index[terrain_name]] = float(
                    weight
                )
            self.terrain_weights[stage] /= self.terrain_weights[stage].sum()
            self.goal_minimum_m[stage] = float(values["goal_distance_m"][0])
            self.goal_maximum_m[stage] = float(values["goal_distance_m"][1])
            self.heading_half_range_rad[stage] = float(
                values["heading_half_range_rad"]
            )
            self.domain_randomization_scale[stage] = float(
                values["domain_randomization_scale"]
            )
            self.geometry_difficulty_range[stage] = torch.tensor(
                values["geometry_difficulty_range"],
                dtype=torch.float32,
                device=self.device,
            )
            geometry_range = self.geometry_difficulty_range[stage]
            self.has_progressive_difficulty[stage] = (
                (geometry_range[1] - geometry_range[0]).abs() > 1.0e-6
            ) | (self.domain_randomization_scale[stage] > 1.0e-6)

    def sample(
        self,
        frontier_stages,
        frontier_difficulty,
        *,
        frontier_probability: float,
        challenge_probability: float,
        challenge_stage_span: int,
        fixed_minimum_stage: int | None = None,
        fixed_maximum_stage: int | None = None,
    ) -> CurriculumTaskBatch:
        """Sample replay, frontier and near-future tasks for each worker."""
        import torch

        frontier_stages = torch.as_tensor(
            frontier_stages, dtype=torch.long, device=self.device
        )
        frontier_difficulty = torch.as_tensor(
            frontier_difficulty, dtype=torch.float32, device=self.device
        )
        if (
            frontier_stages.ndim != 1
            or frontier_difficulty.shape != frontier_stages.shape
            or torch.any(
                (frontier_stages < self.minimum_stage)
                | (frontier_stages > self.maximum_stage)
            )
            or torch.any(
                (frontier_difficulty < 0.0) | (frontier_difficulty > 1.0)
            )
        ):
            raise ValueError("课程前沿批次无效")
        count = frontier_stages.numel()
        stage_indices = torch.arange(
            self.maximum_stage + 1, device=self.device
        )[None, :]

        fixed_sampling = (
            fixed_minimum_stage is not None or fixed_maximum_stage is not None
        )
        if fixed_sampling:
            minimum = (
                self.minimum_stage
                if fixed_minimum_stage is None
                else int(fixed_minimum_stage)
            )
            maximum = (
                self.maximum_stage
                if fixed_maximum_stage is None
                else int(fixed_maximum_stage)
            )
            if not self.minimum_stage <= minimum <= maximum <= self.maximum_stage:
                raise ValueError("固定课程阶段范围无效")
            weights = (
                (stage_indices >= minimum) & (stage_indices <= maximum)
            ).float().expand(count, -1)
        else:
            replay = (
                (stage_indices >= self.minimum_stage)
                & (stage_indices < frontier_stages[:, None])
            )
            frontier = stage_indices == frontier_stages[:, None]
            challenge = (
                (stage_indices > frontier_stages[:, None])
                & (
                    stage_indices
                    <= frontier_stages[:, None] + int(challenge_stage_span)
                )
                & (stage_indices <= self.maximum_stage)
            )
            replay_count = replay.sum(dim=1)
            challenge_count = challenge.sum(dim=1)
            replay_mass = torch.full(
                (count,),
                1.0 - frontier_probability - challenge_probability,
                device=self.device,
            )
            challenge_mass = torch.full(
                (count,), challenge_probability, device=self.device
            )
            frontier_mass = torch.full(
                (count,), frontier_probability, device=self.device
            )
            frontier_mass += torch.where(
                replay_count == 0, replay_mass, torch.zeros_like(replay_mass)
            )
            frontier_mass += torch.where(
                challenge_count == 0,
                challenge_mass,
                torch.zeros_like(challenge_mass),
            )
            replay_per_stage = torch.where(
                replay_count > 0,
                replay_mass / replay_count.clamp(min=1),
                torch.zeros_like(replay_mass),
            )
            challenge_per_stage = torch.where(
                challenge_count > 0,
                challenge_mass / challenge_count.clamp(min=1),
                torch.zeros_like(challenge_mass),
            )
            weights = (
                replay.float() * replay_per_stage[:, None]
                + frontier.float() * frontier_mass[:, None]
                + challenge.float() * challenge_per_stage[:, None]
            )

        stages = torch.multinomial(weights, 1).squeeze(1)
        if fixed_sampling:
            curriculum_difficulty = torch.ones(count, device=self.device)
        else:
            curriculum_difficulty = torch.where(
                stages < frontier_stages,
                torch.ones_like(frontier_difficulty),
                torch.where(
                    stages == frontier_stages,
                    frontier_difficulty,
                    torch.zeros_like(frontier_difficulty),
                ),
            )
        # Stages such as flat-forward and flat-random-goal have no geometry or
        # randomization axis to ramp. They are full-difficulty tasks from the
        # first episode and should not wait for a meaningless synthetic gate.
        curriculum_difficulty = torch.where(
            self.has_progressive_difficulty[stages],
            curriculum_difficulty,
            torch.ones_like(curriculum_difficulty),
        )
        geometry_bounds = self.geometry_difficulty_range[stages]
        geometry_difficulty = geometry_bounds[:, 0] + curriculum_difficulty * (
            geometry_bounds[:, 1] - geometry_bounds[:, 0]
        )
        randomization = (
            self.domain_randomization_scale[stages] * curriculum_difficulty
        )
        return CurriculumTaskBatch(
            stages=stages,
            terrain_probabilities=self.terrain_weights[stages],
            curriculum_difficulty=curriculum_difficulty,
            geometry_difficulty=geometry_difficulty,
            goal_minimum_m=self.goal_minimum_m[stages],
            goal_maximum_m=self.goal_maximum_m[stages],
            heading_half_range_rad=self.heading_half_range_rad[stages],
            domain_randomization_scale=randomization,
        )


class CapabilityCurriculum:
    """One shared curriculum frontier mirrored across all parallel environments.

    Parallel environments are rollout workers, not independent learners. A
    single shared state makes sampling consistent and lets every completed
    frontier episode contribute to the same statistically meaningful gate.
    Public tensors retain one value per environment so the environment can use
    them without scalar synchronization or host-device transfers.
    """

    def __init__(
        self,
        num_envs: int,
        device,
        *,
        initial_level: int,
        minimum_level: int = 0,
        maximum_level: int,
        success_rate_up: float,
        success_rate_down: float,
        minimum_episodes_per_level: int = 8,
        minimum_full_difficulty_episodes: int = 1,
        full_difficulty_threshold: float = 0.9,
        smoothing: float = 0.15,
        allow_level_demotion: bool = False,
    ) -> None:
        import torch

        if num_envs <= 0:
            raise ValueError("并行环境数量必须为正整数")
        if not 0.0 <= success_rate_down < success_rate_up <= 1.0:
            raise ValueError("课程成功率阈值无效")
        if (
            not 0.0 < smoothing <= 1.0
            or not 0 <= minimum_level <= initial_level <= maximum_level
            or minimum_episodes_per_level <= 0
            or minimum_full_difficulty_episodes <= 0
            or not 0.0 < full_difficulty_threshold <= 1.0
        ):
            raise ValueError("课程参数无效")
        self.minimum_level = minimum_level
        self.maximum_level = maximum_level
        self.success_rate_up = success_rate_up
        self.success_rate_down = success_rate_down
        self.smoothing = smoothing
        self.minimum_episodes_per_level = minimum_episodes_per_level
        self.minimum_full_difficulty_episodes = (
            minimum_full_difficulty_episodes
        )
        self.full_difficulty_threshold = full_difficulty_threshold
        self.allow_level_demotion = bool(allow_level_demotion)
        self.levels = torch.full(
            (num_envs,), initial_level, dtype=torch.long, device=device
        )
        self.success_rate = torch.full(
            (num_envs,), 0.5, dtype=torch.float32, device=device
        )
        self.episode_count = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self.full_difficulty_success_rate = torch.full(
            (num_envs,), 0.5, dtype=torch.float32, device=device
        )
        self.full_difficulty_episode_count = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )

    def state_dict(self) -> dict:
        """Return the scalar global state for exact training continuation."""
        return {
            "version": 5,
            "minimum_level": self.minimum_level,
            "maximum_level": self.maximum_level,
            "level": int(self.levels[0].item()),
            "success_rate": float(self.success_rate[0].item()),
            "episode_count": int(self.episode_count[0].item()),
            "full_difficulty_success_rate": float(
                self.full_difficulty_success_rate[0].item()
            ),
            "full_difficulty_episode_count": int(
                self.full_difficulty_episode_count[0].item()
            ),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore a global curriculum state for any environment count."""
        import math

        if not isinstance(state, dict) or state.get("version") != 5:
            raise ValueError("不支持的课程checkpoint格式")
        if int(state.get("maximum_level", -1)) != self.maximum_level:
            raise ValueError("checkpoint课程最高等级与当前配置不一致")
        if int(state.get("minimum_level", -1)) != self.minimum_level:
            raise ValueError("checkpoint课程最低等级与当前配置不一致")
        try:
            level = int(state["level"])
            rate = float(state["success_rate"])
            count = int(state["episode_count"])
            full_rate = float(state["full_difficulty_success_rate"])
            full_count = int(state["full_difficulty_episode_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("课程checkpoint包含无效状态") from error
        if (
            not self.minimum_level <= level <= self.maximum_level
            or not math.isfinite(rate)
            or not 0.0 <= rate <= 1.0
            or count < 0
            or not math.isfinite(full_rate)
            or not 0.0 <= full_rate <= 1.0
            or full_count < 0
        ):
            raise ValueError("课程checkpoint包含无效状态")
        self.levels.fill_(level)
        self.success_rate.fill_(rate)
        self.episode_count.fill_(count)
        self.full_difficulty_success_rate.fill_(full_rate)
        self.full_difficulty_episode_count.fill_(full_count)

    @staticmethod
    def _batch_ema(previous, samples, smoothing: float):
        """Order-invariant EMA update with the same horizon for batch arrivals."""
        effective_smoothing = 1.0 - (1.0 - smoothing) ** samples.numel()
        return previous + effective_smoothing * (samples.mean() - previous)

    def update(self, env_ids, succeeded, difficulty=None) -> None:
        import torch

        env_ids = torch.as_tensor(
            env_ids, dtype=torch.long, device=self.levels.device
        )
        if env_ids.ndim != 1:
            raise ValueError("课程更新的环境索引必须是一维张量")
        if env_ids.numel() == 0:
            return
        if torch.any((env_ids < 0) | (env_ids >= self.levels.numel())):
            raise ValueError("课程更新包含越界环境索引")
        success = torch.as_tensor(
            succeeded, dtype=torch.float32, device=self.levels.device
        )
        if success.shape != env_ids.shape or not torch.isfinite(success).all():
            raise ValueError("课程更新的环境索引与成功标记尺寸不一致")
        if torch.any((success < 0.0) | (success > 1.0)):
            raise ValueError("课程成功标记必须位于[0,1]")
        if difficulty is None:
            difficulty = torch.ones_like(success)
        else:
            difficulty = torch.as_tensor(
                difficulty, dtype=torch.float32, device=self.levels.device
            )
        if (
            difficulty.shape != env_ids.shape
            or not torch.isfinite(difficulty).all()
            or torch.any((difficulty < 0.0) | (difficulty > 1.0))
        ):
            raise ValueError("课程更新难度必须为与环境索引等长的[0,1]张量")

        rate = self._batch_ema(
            self.success_rate[0], success, self.smoothing
        )
        episode_count = self.episode_count[0] + env_ids.numel()
        full_mask = difficulty >= self.full_difficulty_threshold
        full_rate = self.full_difficulty_success_rate[0]
        full_episode_count = self.full_difficulty_episode_count[0]
        if bool(full_mask.any()):
            full_success = success[full_mask]
            full_rate = self._batch_ema(
                full_rate, full_success, self.smoothing
            )
            full_episode_count = full_episode_count + full_success.numel()

        level = self.levels[0]
        enough_history = episode_count >= self.minimum_episodes_per_level
        enough_full_difficulty = (
            full_episode_count >= self.minimum_full_difficulty_episodes
        )
        mastered = bool(
            enough_history
            & enough_full_difficulty
            & (rate >= self.success_rate_up)
            & (full_rate >= self.success_rate_up)
        )
        promote = mastered and int(level.item()) < self.maximum_level
        demote = bool(
            enough_history
            & (rate <= self.success_rate_down)
            & (level > self.minimum_level)
        ) and self.allow_level_demotion

        if promote:
            level = level + 1
        elif demote:
            level = level - 1
        changed = promote or demote
        if changed:
            rate = torch.full_like(rate, 0.5)
            episode_count = torch.zeros_like(episode_count)
            full_rate = torch.full_like(full_rate, 0.5)
            full_episode_count = torch.zeros_like(full_episode_count)

        self.levels.fill_(int(level.item()))
        self.success_rate.fill_(float(rate.item()))
        self.episode_count.fill_(int(episode_count.item()))
        self.full_difficulty_success_rate.fill_(float(full_rate.item()))
        self.full_difficulty_episode_count.fill_(
            int(full_episode_count.item())
        )

    def frontier_difficulty(self, env_ids=None):
        """Return the shared smooth [0, 1] geometry scale."""
        import torch

        if env_ids is None:
            count = self.levels.numel()
        else:
            env_ids = torch.as_tensor(
                env_ids, dtype=torch.long, device=self.levels.device
            )
            if env_ids.ndim != 1 or torch.any(
                (env_ids < 0) | (env_ids >= self.levels.numel())
            ):
                raise ValueError("课程查询包含无效环境索引")
            count = env_ids.numel()
        history_fraction = torch.clamp(
            self.episode_count[0].float()
            / float(self.minimum_episodes_per_level),
            0.0,
            1.0,
        )
        success_fraction = torch.clamp(
            (self.success_rate[0] - self.success_rate_down)
            / (self.success_rate_up - self.success_rate_down),
            0.0,
            1.0,
        )
        return (history_fraction * success_fraction).expand(count)
