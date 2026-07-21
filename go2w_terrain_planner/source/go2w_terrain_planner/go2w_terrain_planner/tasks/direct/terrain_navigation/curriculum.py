"""Per-environment terrain curriculum driven only by execution outcomes."""

from __future__ import annotations


class TerrainCurriculum:
    def __init__(
        self,
        num_envs: int,
        device,
        *,
        initial_level: int,
        maximum_level: int,
        success_rate_up: float,
        success_rate_down: float,
        minimum_episodes_per_level: int = 8,
        smoothing: float = 0.15,
    ) -> None:
        import torch

        if not 0.0 <= success_rate_down < success_rate_up <= 1.0:
            raise ValueError("课程成功率阈值无效")
        if (
            not 0.0 < smoothing <= 1.0
            or not 0 <= initial_level <= maximum_level
            or minimum_episodes_per_level <= 0
        ):
            raise ValueError("课程参数无效")
        self.maximum_level = maximum_level
        self.success_rate_up = success_rate_up
        self.success_rate_down = success_rate_down
        self.smoothing = smoothing
        self.minimum_episodes_per_level = minimum_episodes_per_level
        self.levels = torch.full((num_envs,), initial_level, dtype=torch.long, device=device)
        self.success_rate = torch.full((num_envs,), 0.5, dtype=torch.float32, device=device)
        self.episode_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    def update(self, env_ids, succeeded) -> None:
        import torch

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.levels.device)
        if env_ids.numel() == 0:
            return
        success = torch.as_tensor(succeeded, dtype=torch.float32, device=self.levels.device)
        if success.shape != env_ids.shape:
            raise ValueError("课程更新的环境索引与成功标记尺寸不一致")
        rate = (1.0 - self.smoothing) * self.success_rate[env_ids] + self.smoothing * success
        episode_count = self.episode_count[env_ids] + 1
        level = self.levels[env_ids]
        enough_history = episode_count >= self.minimum_episodes_per_level
        promote = enough_history & (rate >= self.success_rate_up)
        demote = enough_history & (rate <= self.success_rate_down)
        level = torch.where(promote, level + 1, level)
        level = torch.where(demote, level - 1, level)
        changed = promote | demote
        self.levels[env_ids] = level.clamp(0, self.maximum_level)
        self.success_rate[env_ids] = torch.where(changed, torch.full_like(rate, 0.5), rate)
        self.episode_count[env_ids] = torch.where(
            changed, torch.zeros_like(episode_count), episode_count
        )
