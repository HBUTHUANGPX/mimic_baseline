"""motion reconstruction 训练使用的 GPU 常驻 window 采样。

本模块假设 frame feature 已经构建完成，并将它们常驻同一个 device。
每个 epoch 只在 `valid_center_indices` 上做无放回随机打乱，然后用向量化索引
抽取 history/current/future window。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MotionWindowBatch:
    """一批对齐的 robot/human window。

    `window_indices` 主要用于调试和未来评估，训练只需要两个 window 张量。
    """

    robot_window: torch.Tensor
    human_window: torch.Tensor
    center_indices: torch.Tensor
    window_indices: torch.Tensor


class MotionWindowBuffer:
    """让 feature tensor 常驻一个 device，并在该 device 上采样 window。

    这里不实现 PyTorch Dataset 的逐样本 `__getitem__`，避免大规模随机训练
    时在 CPU/Python 层逐帧取数。
    """

    def __init__(
        self,
        *,
        robot_features: torch.Tensor,
        human_features: torch.Tensor,
        motion_lengths: torch.Tensor,
        history: int,
        future: int,
        device: str | torch.device | None = None,
    ):
        if robot_features.shape[0] != human_features.shape[0]:
            raise ValueError("robot_features and human_features must have the same frame count.")
        if history < 0 or future < 0:
            raise ValueError("history and future must be non-negative.")
        self.device = torch.device(device) if device is not None else robot_features.device
        self.robot_features = robot_features.to(self.device)
        self.human_features = human_features.to(self.device)
        self.motion_lengths = motion_lengths.to(self.device, dtype=torch.long)
        self.history = int(history)
        self.future = int(future)
        self.window_offsets = torch.arange(-history, future + 1, device=self.device, dtype=torch.long)
        self.valid_center_indices = self._build_valid_center_indices()
        if self.valid_center_indices.numel() == 0:
            raise ValueError("No valid center frames found for the requested history/future window.")

    @property
    def window_size(self) -> int:
        return int(self.window_offsets.numel())

    def iter_epoch_batches(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        order = torch.randperm(self.valid_center_indices.numel(), device=self.device, generator=generator)
        epoch_centers = self.valid_center_indices[order]
        for start in range(0, epoch_centers.numel(), batch_size):
            centers = epoch_centers[start : start + batch_size]
            # 中文：最后不足 batch_size 的 batch 会保留；索引构造完全发生在 device 上。
            window_indices = centers[:, None] + self.window_offsets[None, :]
            yield MotionWindowBatch(
                robot_window=self.robot_features[window_indices],
                human_window=self.human_features[window_indices],
                center_indices=centers,
                window_indices=window_indices,
            )

    def _build_valid_center_indices(self) -> torch.Tensor:
        """构建不会跨 clip 边界的中心帧池。

        每个 motion clip 独立计算可用中心帧，因此 history/future 不会越界到
        相邻 clip。
        """
        starts = torch.cat(
            [
                torch.zeros(1, device=self.device, dtype=torch.long),
                torch.cumsum(self.motion_lengths[:-1], dim=0),
            ]
        )
        centers: list[torch.Tensor] = []
        for start, length in zip(starts.tolist(), self.motion_lengths.tolist()):
            first = start + self.history
            last_exclusive = start + length - self.future
            if first < last_exclusive:
                centers.append(torch.arange(first, last_exclusive, device=self.device, dtype=torch.long))
        if not centers:
            return torch.empty(0, device=self.device, dtype=torch.long)
        return torch.cat(centers, dim=0)
