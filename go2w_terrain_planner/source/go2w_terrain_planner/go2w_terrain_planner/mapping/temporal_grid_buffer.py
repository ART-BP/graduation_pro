"""Fixed-length temporal buffer for maps, poses, and velocity commands."""

from __future__ import annotations


class TemporalGridBuffer:
    """GPU-resident history with current-frame SE(2) and height-reference alignment."""

    def __init__(
        self,
        num_envs: int,
        history_length: int,
        channels: int,
        size: int,
        device,
        normalized_ground_height_scale_m: float,
    ) -> None:
        import torch

        if (
            num_envs <= 0
            or history_length < 2
            or channels <= 0
            or size <= 0
            or normalized_ground_height_scale_m <= 0.0
        ):
            raise ValueError("时序缓存尺寸参数无效")
        self.num_envs = num_envs
        self.history_length = history_length
        self.device = torch.device(device)
        self.normalized_ground_height_scale_m = float(normalized_ground_height_scale_m)
        self.maps = torch.zeros((num_envs, history_length, channels, size, size), device=self.device)
        self.poses = torch.zeros((num_envs, history_length, 3), device=self.device)
        self.ground_references_z = torch.zeros((num_envs, history_length), device=self.device)
        self.commands = torch.zeros((num_envs, history_length - 1, 2), device=self.device)

    def reset(self, env_ids, current_map, current_pose, ground_reference_z) -> None:
        import torch

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if current_map.shape != self.maps[env_ids, 0].shape or current_pose.shape != self.poses[env_ids, 0].shape:
            raise ValueError("reset输入尺寸与缓存不一致")
        ground_reference_z = torch.as_tensor(
            ground_reference_z, dtype=self.maps.dtype, device=self.device
        )
        if ground_reference_z.shape != (env_ids.numel(),):
            raise ValueError("ground_reference_z必须与env_ids等长")
        self.maps[env_ids] = current_map[:, None].expand(-1, self.history_length, -1, -1, -1)
        self.poses[env_ids] = current_pose[:, None].expand(-1, self.history_length, -1)
        self.ground_references_z[env_ids] = ground_reference_z[:, None]
        self.commands[env_ids] = 0.0

    def push(self, current_map, current_pose, command, ground_reference_z) -> None:
        if current_map.shape != self.maps[:, 0].shape or current_pose.shape != self.poses[:, 0].shape:
            raise ValueError("push地图或位姿尺寸错误")
        if command.shape != (self.num_envs, 2):
            raise ValueError("command形状必须为[B,2]")
        if ground_reference_z.shape != (self.num_envs,):
            raise ValueError("ground_reference_z形状必须为[B]")
        self.maps = self.maps.roll(-1, dims=1)
        self.poses = self.poses.roll(-1, dims=1)
        self.ground_references_z = self.ground_references_z.roll(-1, dims=1)
        self.commands = self.commands.roll(-1, dims=1)
        self.maps[:, -1] = current_map
        self.poses[:, -1] = current_pose
        self.ground_references_z[:, -1] = ground_reference_z
        self.commands[:, -1] = command

    def aligned_maps(self, extent_m: float):
        from .coordinate_transform import warp_map_sequence

        return warp_map_sequence(
            self.maps,
            self.poses,
            self.poses[:, -1],
            extent_m,
            source_ground_reference_z=self.ground_references_z,
            target_ground_reference_z=self.ground_references_z[:, -1],
            normalized_ground_height_scale_m=self.normalized_ground_height_scale_m,
        )

    def motion_history(self):
        from .coordinate_transform import relative_pose_deltas

        return relative_pose_deltas(self.poses)
