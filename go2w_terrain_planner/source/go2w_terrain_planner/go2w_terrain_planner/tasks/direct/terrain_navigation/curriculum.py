"""Per-environment terrain curriculum driven only by execution outcomes."""

from __future__ import annotations


def goal_distance_maximum_for_levels(
    levels,
    *,
    initial_level: int,
    maximum_level: int,
    start_maximum_m: float,
    final_maximum_m: float,
):
    """Map terrain curriculum levels linearly onto the goal-distance ceiling."""
    import torch

    if (
        maximum_level <= initial_level
        or start_maximum_m <= 0.0
        or final_maximum_m < start_maximum_m
    ):
        raise ValueError("目标距离课程参数无效")
    level_tensor = torch.as_tensor(levels)
    fraction = (
        (level_tensor.to(dtype=torch.float32) - float(initial_level))
        / float(maximum_level - initial_level)
    ).clamp(0.0, 1.0)
    return start_maximum_m + fraction * (final_maximum_m - start_maximum_m)


class TerrainCurriculum:
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
        self.levels = torch.full((num_envs,), initial_level, dtype=torch.long, device=device)
        # 目标距离依据已经掌握的等级推进，而不是依据当前正在挑战的等级。
        # 初始等级对应基础目标距离；之后每次掌握当前前沿时才单调增加。
        self.goal_levels = torch.full(
            (num_envs,), initial_level, dtype=torch.long, device=device
        )
        self.success_rate = torch.full((num_envs,), 0.5, dtype=torch.float32, device=device)
        self.episode_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.full_difficulty_success_rate = torch.full(
            (num_envs,), 0.5, dtype=torch.float32, device=device
        )
        self.full_difficulty_episode_count = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )

    def state_dict(self) -> dict:
        """Return a portable state for exact training continuation."""
        return {
            "version": 3,
            "minimum_level": self.minimum_level,
            "maximum_level": self.maximum_level,
            "levels": self.levels.detach().cpu(),
            "goal_levels": self.goal_levels.detach().cpu(),
            "success_rate": self.success_rate.detach().cpu(),
            "episode_count": self.episode_count.detach().cpu(),
            "full_difficulty_success_rate": (
                self.full_difficulty_success_rate.detach().cpu()
            ),
            "full_difficulty_episode_count": (
                self.full_difficulty_episode_count.detach().cpu()
            ),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore curriculum state, repeating it if the environment count changed."""
        import torch

        if not isinstance(state, dict) or state.get("version") not in (1, 2, 3):
            raise ValueError("不支持的课程checkpoint格式")
        if int(state.get("maximum_level", -1)) != self.maximum_level:
            raise ValueError("checkpoint课程最高等级与当前配置不一致")
        if int(state.get("minimum_level", 0)) != self.minimum_level:
            raise ValueError("checkpoint课程最低等级与当前配置不一致")
        saved_levels = torch.as_tensor(state.get("levels"), dtype=torch.long)
        if state.get("version") in (2, 3):
            saved_goal_levels = torch.as_tensor(
                state.get("goal_levels"), dtype=torch.long
            )
        else:
            # 旧checkpoint没有独立目标课程。恢复时让目标难度落后当前
            # 挑战前沿一级，避免续训立即同时承受远目标和高难地形。
            saved_goal_levels = (saved_levels - 1).clamp(
                min=self.minimum_level,
                max=self.maximum_level,
            )
        saved_rate = torch.as_tensor(state.get("success_rate"), dtype=torch.float32)
        saved_count = torch.as_tensor(state.get("episode_count"), dtype=torch.long)
        if state.get("version") == 3:
            saved_full_rate = torch.as_tensor(
                state.get("full_difficulty_success_rate"),
                dtype=torch.float32,
            )
            saved_full_count = torch.as_tensor(
                state.get("full_difficulty_episode_count"),
                dtype=torch.long,
            )
        else:
            # 旧课程没有完整难度验证统计。保留其当前前沿，但必须在该
            # 前沿重新通过完整难度验证，避免旧的容易回合直接触发升级。
            saved_full_rate = torch.full_like(saved_rate, 0.5)
            saved_full_count = torch.zeros_like(saved_count)
        if (
            saved_levels.ndim != 1
            or saved_levels.numel() == 0
            or saved_goal_levels.shape != saved_levels.shape
            or saved_rate.shape != saved_levels.shape
            or saved_count.shape != saved_levels.shape
            or saved_full_rate.shape != saved_levels.shape
            or saved_full_count.shape != saved_levels.shape
        ):
            raise ValueError("课程checkpoint张量尺寸无效")
        if (
            torch.any(
                (saved_levels < self.minimum_level)
                | (saved_levels > self.maximum_level)
            )
            or torch.any(
                (saved_goal_levels < self.minimum_level)
                | (saved_goal_levels > self.maximum_level)
            )
            or not torch.isfinite(saved_rate).all()
            or torch.any((saved_rate < 0.0) | (saved_rate > 1.0))
            or torch.any(saved_count < 0)
            or not torch.isfinite(saved_full_rate).all()
            or torch.any((saved_full_rate < 0.0) | (saved_full_rate > 1.0))
            or torch.any(saved_full_count < 0)
        ):
            raise ValueError("课程checkpoint包含无效状态")
        restore_index = torch.arange(
            self.levels.numel(), device=self.levels.device
        ) % saved_levels.numel()
        self.levels.copy_(
            saved_levels.to(self.levels.device)[restore_index]
        )
        self.goal_levels.copy_(
            saved_goal_levels.to(self.goal_levels.device)[restore_index]
        )
        self.success_rate.copy_(
            saved_rate.to(self.success_rate.device)[restore_index]
        )
        self.episode_count.copy_(
            saved_count.to(self.episode_count.device)[restore_index]
        )
        self.full_difficulty_success_rate.copy_(
            saved_full_rate.to(
                self.full_difficulty_success_rate.device
            )[restore_index]
        )
        self.full_difficulty_episode_count.copy_(
            saved_full_count.to(
                self.full_difficulty_episode_count.device
            )[restore_index]
        )

    def update(self, env_ids, succeeded, difficulty=None) -> None:
        import torch

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.levels.device)
        if env_ids.numel() == 0:
            return
        success = torch.as_tensor(succeeded, dtype=torch.float32, device=self.levels.device)
        if success.shape != env_ids.shape:
            raise ValueError("课程更新的环境索引与成功标记尺寸不一致")
        if difficulty is None:
            difficulty = torch.ones_like(success)
        else:
            difficulty = torch.as_tensor(
                difficulty, dtype=torch.float32, device=self.levels.device
            )
        if (
            difficulty.shape != env_ids.shape
            or torch.any((difficulty < 0.0) | (difficulty > 1.0))
        ):
            raise ValueError("课程更新难度必须为与环境索引等长的[0,1]张量")
        rate = (1.0 - self.smoothing) * self.success_rate[env_ids] + self.smoothing * success
        episode_count = self.episode_count[env_ids] + 1
        full_difficulty = difficulty >= self.full_difficulty_threshold
        previous_full_rate = self.full_difficulty_success_rate[env_ids]
        updated_full_rate = (
            (1.0 - self.smoothing) * previous_full_rate
            + self.smoothing * success
        )
        full_rate = torch.where(
            full_difficulty, updated_full_rate, previous_full_rate
        )
        full_episode_count = (
            self.full_difficulty_episode_count[env_ids]
            + full_difficulty.to(dtype=torch.long)
        )
        level = self.levels[env_ids]
        enough_history = episode_count >= self.minimum_episodes_per_level
        enough_full_difficulty = (
            full_episode_count >= self.minimum_full_difficulty_episodes
        )
        mastered = (
            enough_history
            & enough_full_difficulty
            & (rate >= self.success_rate_up)
            & (full_rate >= self.success_rate_up)
        )
        promote = mastered & (level < self.maximum_level)
        demote = (
            enough_history
            & (rate <= self.success_rate_down)
            & (level > self.minimum_level)
            & self.allow_level_demotion
        )
        # 先记录已掌握的当前等级，再把地形前沿推进一级。这样新前沿不会
        # 与更远目标在同一回合同时出现，目标课程也不会随失败而回退。
        self.goal_levels[env_ids] = torch.where(
            mastered,
            torch.maximum(self.goal_levels[env_ids], level),
            self.goal_levels[env_ids],
        )
        level = torch.where(promote, level + 1, level)
        level = torch.where(demote, level - 1, level)
        changed = promote | demote
        self.levels[env_ids] = level.clamp(
            self.minimum_level, self.maximum_level
        )
        self.success_rate[env_ids] = torch.where(changed, torch.full_like(rate, 0.5), rate)
        self.episode_count[env_ids] = torch.where(
            changed, torch.zeros_like(episode_count), episode_count
        )
        self.full_difficulty_success_rate[env_ids] = torch.where(
            changed, torch.full_like(full_rate, 0.5), full_rate
        )
        self.full_difficulty_episode_count[env_ids] = torch.where(
            changed,
            torch.zeros_like(full_episode_count),
            full_episode_count,
        )

    def frontier_difficulty(self, env_ids=None):
        """Return a smooth [0, 1] geometry scale for the current frontier.

        A newly unlocked level starts from its easiest geometry. Difficulty
        grows only when both the number of frontier episodes and their EMA
        success rate increase. This creates an intra-level curriculum without
        moving the discrete terrain frontier backwards.
        """
        import torch

        if env_ids is None:
            env_ids = torch.arange(
                self.levels.numel(), dtype=torch.long, device=self.levels.device
            )
        else:
            env_ids = torch.as_tensor(
                env_ids, dtype=torch.long, device=self.levels.device
            )
        history_fraction = (
            self.episode_count[env_ids].to(dtype=torch.float32)
            / float(self.minimum_episodes_per_level)
        ).clamp(0.0, 1.0)
        success_fraction = (
            (self.success_rate[env_ids] - self.success_rate_down)
            / (self.success_rate_up - self.success_rate_down)
        ).clamp(0.0, 1.0)
        return history_fraction * success_fraction
