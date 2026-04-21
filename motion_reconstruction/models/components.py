"""双编码器、单解码器的 FSQ 重构模块。

模型层不依赖 npz、motion group 或 body 名字。它只处理已经归一化并展平的
robot/human 窗口特征。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from motion_reconstruction.models.quantizers import QuantizerOutput


@dataclass
class DualAutoEncoderOutput:
    """前向输出，供重构损失和潜变量对齐损失使用。"""

    z_robot: torch.Tensor
    z_human: torch.Tensor
    z_cycle: torch.Tensor
    q_robot: torch.Tensor
    q_human: torch.Tensor
    q_cycle: torch.Tensor
    recon_from_robot: torch.Tensor
    recon_from_human: torch.Tensor
    robot_quantizer: QuantizerOutput
    human_quantizer: QuantizerOutput
    cycle_quantizer: QuantizerOutput


class FSQMLPEncoder(nn.Module):
    """将展平窗口映射到 FSQ latent_dim 的 MLP 编码器。"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int],
        *,
        activation: str = "elu",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = [int(dim) for dim in hidden_dims]
        self.net = _make_mlp(self.input_dim, self.hidden_dims, self.latent_dim, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FSQMLPDecoder(nn.Module):
    """重构展平 robot 窗口的 MLP 解码器。"""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        *,
        activation: str = "elu",
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        self.hidden_dims = [int(dim) for dim in hidden_dims]
        self.net = _make_mlp(self.latent_dim, self.hidden_dims, self.output_dim, activation)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DualFSQAutoEncoder(nn.Module):
    """包含共享量化器和 robot 解码器的 robot/human 编码器。

    robot 编码器和 human 编码器各自输入不同特征，但共享同一个
    量化器和解码器。解码器的重构目标始终是完整 robot 窗口。
    """

    def __init__(
        self,
        *,
        robot_input_dim: int,
        human_input_dim: int,
        latent_dim: int,
        robot_encoder_hidden_dims: Sequence[int],
        human_encoder_hidden_dims: Sequence[int],
        decoder_hidden_dims: Sequence[int],
        quantizer: nn.Module,
        activation: str = "elu",
    ):
        super().__init__()
        self.robot_input_dim = int(robot_input_dim)
        self.human_input_dim = int(human_input_dim)
        self.latent_dim = int(latent_dim)
        self.robot_encoder = FSQMLPEncoder(
            self.robot_input_dim,
            self.latent_dim,
            robot_encoder_hidden_dims,
            activation=activation,
        )
        self.human_encoder = FSQMLPEncoder(
            self.human_input_dim,
            self.latent_dim,
            human_encoder_hidden_dims,
            activation=activation,
        )
        self.shared_quantizer = quantizer
        self.decoder = FSQMLPDecoder(
            self.latent_dim,
            self.robot_input_dim,
            decoder_hidden_dims,
            activation=activation,
        )

    def forward(self, robot_window: torch.Tensor, human_window: torch.Tensor) -> DualAutoEncoderOutput:
        z_robot = self.robot_encoder(robot_window)
        z_human = self.human_encoder(human_window)
        robot_quantizer = self.shared_quantizer(z_robot)
        human_quantizer = self.shared_quantizer(z_human)

        recon_from_robot = self.decoder(robot_quantizer.z_q)
        recon_from_human = self.decoder(human_quantizer.z_q)

        # 中文：cycle 潜变量不 detach，让 human->decoder->robot_encoder 的路径也受
        # 潜变量对齐目标约束。
        z_cycle = self.robot_encoder(recon_from_human)
        cycle_quantizer = self.shared_quantizer(z_cycle)

        return DualAutoEncoderOutput(
            z_robot=z_robot,
            z_human=z_human,
            z_cycle=z_cycle,
            q_robot=robot_quantizer.z_q,
            q_human=human_quantizer.z_q,
            q_cycle=cycle_quantizer.z_q,
            recon_from_robot=recon_from_robot,
            recon_from_human=recon_from_human,
            robot_quantizer=robot_quantizer,
            human_quantizer=human_quantizer,
            cycle_quantizer=cycle_quantizer,
        )


def _make_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(_activation_module(activation))
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)


def _activation_module(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")
