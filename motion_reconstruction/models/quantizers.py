"""与 FSQ/iFSQ 参考行为对齐的有限标量量化器。

本模块实现面向 motion 潜变量 `[B, latent_dim]` 的标量量化器。第一版保留
per-dimension `level_indices`，不引入 token/multi-codebook/global index。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass
class QuantizerOutput:
    """量化器输出容器。

    `z_q` 参与解码器和潜变量损失；`level_indices` 用于日志/调试。
    """

    z_q: torch.Tensor
    level_indices: torch.Tensor
    stats: dict[str, torch.Tensor | float]


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """round 操作的直通估计器。

    前向使用 round，反向把梯度按恒等映射传过 round；bound 函数自己的
    tanh/sigmoid 梯度仍然保留。
    """
    return x + (x.round() - x).detach()


class _BaseScalarQuantizer(nn.Module):
    def __init__(self, levels: int | Sequence[int], eps: float = 1e-3):
        super().__init__()
        self._levels_config = int(levels) if isinstance(levels, int) else [int(level) for level in levels]
        self.eps = float(eps)

    def _levels_for(self, dim: int, device: torch.device) -> torch.Tensor:
        if isinstance(self._levels_config, int):
            levels = torch.full((dim,), self._levels_config, device=device, dtype=torch.float32)
        else:
            if len(self._levels_config) != dim:
                raise ValueError(f"levels length must match latent dim {dim}, got {len(self._levels_config)}.")
            levels = torch.tensor(self._levels_config, device=device, dtype=torch.float32)
        if torch.any(levels < 2):
            raise ValueError("All FSQ levels must be >= 2.")
        return levels

    def config_dict(self, latent_dim: int | None = None) -> dict:
        levels = self._levels_config
        if latent_dim is not None and isinstance(levels, int):
            levels = [levels] * latent_dim
        return {"levels": levels, "eps": self.eps}


class FSQQuantizer(_BaseScalarQuantizer):
    """使用 bounded tanh 和 STE rounding 的有限标量量化。

    公式与 lucidrains FSQ 的核心标量量化对齐。
    """

    def forward(self, z: torch.Tensor) -> QuantizerOutput:
        levels = self._levels_for(z.shape[-1], z.device)
        quantized = round_ste(self._bound(z, levels))
        half_width = torch.floor(levels / 2)
        z_q = quantized / half_width
        level_indices = _level_indices(quantized, levels)
        return QuantizerOutput(
            z_q=z_q,
            level_indices=level_indices,
            stats={
                "mean_abs_z": z.detach().abs().mean(),
                "mean_abs_z_q": z_q.detach().abs().mean(),
                "unique_codes": level_indices.detach().unique().numel() / level_indices.numel(),
            },
        )

    def _bound(self, z: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        # 中文：偶数 level 需要 0.5 offset，让可量化点对称落在半整数网格。
        half_l = (levels - 1) * (1.0 - self.eps) / 2.0
        offset = torch.where((levels.long() % 2) == 0, torch.full_like(levels, 0.5), torch.zeros_like(levels))
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset


class IFSQQuantizer(_BaseScalarQuantizer):
    """默认使用 Tencent-Hunyuan 风格 simple bound 的 iFSQ。

    默认使用 Tencent-Hunyuan/iFSQ 常用的 simple_bound +
    `scale_sigmoid_16`，即 `2 * sigmoid(1.6x) - 1`。
    """

    def __init__(
        self,
        levels: int | Sequence[int],
        *,
        do_simple_bound: bool = True,
        act_fun: str = "scale_sigmoid_16",
        eps: float = 1e-3,
    ):
        super().__init__(levels, eps=eps)
        self.do_simple_bound = bool(do_simple_bound)
        self.act_fun = act_fun

    def forward(self, z: torch.Tensor) -> QuantizerOutput:
        levels = self._levels_for(z.shape[-1], z.device)
        bounded = self._simple_bound(z, levels) if self.do_simple_bound else self._fsq_bound(z, levels)
        quantized = round_ste(bounded)
        half_width = torch.floor(levels / 2)
        z_q = quantized / half_width
        level_indices = _level_indices(quantized, levels)
        return QuantizerOutput(
            z_q=z_q,
            level_indices=level_indices,
            stats={
                "mean_abs_z": z.detach().abs().mean(),
                "mean_abs_z_q": z_q.detach().abs().mean(),
                "unique_codes": level_indices.detach().unique().numel() / level_indices.numel(),
            },
        )

    def config_dict(self, latent_dim: int | None = None) -> dict:
        config = super().config_dict(latent_dim=latent_dim)
        config.update({"do_simple_bound": self.do_simple_bound, "act_fun": self.act_fun})
        return config

    def _simple_bound(self, z: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        half_l = (levels - 1) / 2.0
        offset = torch.where((levels.long() % 2) == 0, torch.full_like(levels, 0.5), torch.zeros_like(levels))
        return _activation(self.act_fun, z) * half_l - offset

    def _fsq_bound(self, z: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        half_l = (levels - 1) * (1.0 - self.eps) / 2.0
        offset = torch.where((levels.long() % 2) == 0, torch.full_like(levels, 0.5), torch.zeros_like(levels))
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset


def build_quantizer(config: dict, latent_dim: int | None = None) -> nn.Module:
    qtype = str(config.get("type", "ifsq")).lower()
    levels = config.get("levels", 17)
    if latent_dim is not None and isinstance(levels, int):
        levels = [levels] * latent_dim
    if qtype == "fsq":
        return FSQQuantizer(levels=levels, eps=float(config.get("eps", 1e-3)))
    if qtype == "ifsq":
        return IFSQQuantizer(
            levels=levels,
            do_simple_bound=bool(config.get("do_simple_bound", True)),
            act_fun=str(config.get("act_fun", "scale_sigmoid_16")),
            eps=float(config.get("eps", 1e-3)),
        )
    raise ValueError(f"Unknown quantizer type: {qtype}")


def normalized_quantizer_config(config: dict, latent_dim: int) -> dict:
    """将便于书写的量化器配置规范化成 checkpoint 稳定配置。

    用户可以写 `levels: 17`，内部和 checkpoint 统一展开成长度为
    `latent_dim` 的 list。
    """
    qtype = str(config.get("type", "ifsq")).lower()
    levels = config.get("levels", 17)
    if isinstance(levels, int):
        levels = [int(levels)] * latent_dim
    else:
        levels = [int(level) for level in levels]
    if len(levels) != latent_dim:
        raise ValueError(f"levels length must match latent_dim={latent_dim}, got {len(levels)}.")
    normalized = {"type": qtype, "levels": levels, "eps": float(config.get("eps", 1e-3))}
    if qtype == "ifsq":
        normalized.update(
            {
                "do_simple_bound": bool(config.get("do_simple_bound", True)),
                "act_fun": str(config.get("act_fun", "scale_sigmoid_16")),
            }
        )
    return normalized


def _activation(name: str, z: torch.Tensor) -> torch.Tensor:
    if name == "scale_sigmoid_16":
        return 2.0 * torch.sigmoid(1.6 * z) - 1.0
    if name == "scale_sigmoid_20":
        return 2.0 * torch.sigmoid(2.0 * z) - 1.0
    if name == "tanh":
        return torch.tanh(z)
    raise ValueError(f"Unknown iFSQ activation: {name}")


def _level_indices(quantized: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """将有符号标量网格值转成非负的逐维 level id。"""
    half_width = torch.floor(levels / 2)
    offset = torch.where((levels.long() % 2) == 0, torch.full_like(levels, 0.5), torch.zeros_like(levels))
    return torch.round(quantized - offset + half_width).long().clamp_min(0).clamp_max((levels - 1).long())
