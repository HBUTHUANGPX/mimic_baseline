"""从单帧特征统计并 repeat 到窗口维度的固定归一化器。

离线训练启动时一次性从全量单帧特征统计 mean/std，再 repeat 到窗口
维度；训练过程中不再更新统计量。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class WindowFeatureNormalizer:
    """面向展平窗口的 EmpiricalNormalization 风格固定归一化器。

    接口保留 `__call__` 和 `inverse`，便于之后导出重构结果或可视化时反归一化。
    """

    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-2

    @classmethod
    def from_frame_features(
        cls,
        frame_features: torch.Tensor,
        *,
        window_size: int,
        eps: float = 1e-2,
    ) -> "WindowFeatureNormalizer":
        if frame_features.ndim != 2:
            raise ValueError("frame_features must have shape [frames, feature_dim].")
        # 中文：按单帧特征统计，再 repeat 到 `[history + current + future]`
        # 展平后的窗口特征。
        mean = frame_features.mean(dim=0)
        std = frame_features.var(dim=0, unbiased=False).sqrt()
        return cls(mean=mean.repeat(window_size), std=std.repeat(window_size), eps=eps)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device, x.dtype)) / (self.std.to(x.device, x.dtype) + self.eps)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y * (self.std.to(y.device, y.dtype) + self.eps) + self.mean.to(y.device, y.dtype)

    def to(self, device: str | torch.device) -> "WindowFeatureNormalizer":
        return WindowFeatureNormalizer(mean=self.mean.to(device), std=self.std.to(device), eps=self.eps)

    def state_dict(self) -> dict:
        return {"mean": self.mean.detach().cpu(), "std": self.std.detach().cpu(), "eps": self.eps}

    @classmethod
    def from_state_dict(cls, state: dict, device: str | torch.device = "cpu") -> "WindowFeatureNormalizer":
        return cls(
            mean=state["mean"].to(device),
            std=state["std"].to(device),
            eps=float(state.get("eps", 1e-2)),
        )
