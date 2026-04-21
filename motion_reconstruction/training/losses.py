"""双 encoder FSQ reconstruction 的损失函数。

第一版只使用 MSE 组合，不引入额外正则项。latent loss 使用量化后的 latent，
且 cycle path 不 detach。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from motion_reconstruction.models.components import DualAutoEncoderOutput


@dataclass
class LossOutput:
    """加权总损失和未加权的命名损失项。"""

    total: torch.Tensor
    terms: dict[str, torch.Tensor]


class DualReconstructionLoss:
    """robot/human 重构与 latent 对齐使用的 MSE 组合损失。

    中文：
    - robot_recon: robot encoder -> decoder 的 robot window 重构
    - human_recon: human encoder -> decoder 的 robot window 重构
    - latent_align: q_human 与 q_robot 对齐
    - cycle_latent: recon_from_human -> robot_encoder 后与 q_human 对齐
    """

    def __init__(
        self,
        *,
        robot_recon: float = 1.0,
        human_recon: float = 1.0,
        latent_align: float = 0.25,
        cycle_latent: float = 0.25,
    ):
        self.weights = {
            "robot_recon": float(robot_recon),
            "human_recon": float(human_recon),
            "latent_align": float(latent_align),
            "cycle_latent": float(cycle_latent),
        }

    def __call__(self, output: DualAutoEncoderOutput, robot_target: torch.Tensor) -> LossOutput:
        terms = {
            "robot_recon": F.mse_loss(output.recon_from_robot, robot_target),
            "human_recon": F.mse_loss(output.recon_from_human, robot_target),
            "latent_align": F.mse_loss(output.q_human, output.q_robot),
            "cycle_latent": F.mse_loss(output.q_cycle, output.q_human),
        }
        total = sum(self.weights[name] * value for name, value in terms.items())
        return LossOutput(total=total, terms=terms)
