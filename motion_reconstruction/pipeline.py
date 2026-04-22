"""训练、评估和可视化共享的构建流程。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from motion_reconstruction.config.schema import MotionReconstructionConfig
from motion_reconstruction.data import MotionSourceResolver, MotionWindowBuffer, RawMotionDataset, RawMotionLoader
from motion_reconstruction.features import FeatureBuilder, FeatureBuilderConfig, FeatureBundle
from motion_reconstruction.models import DualFSQAutoEncoder
from motion_reconstruction.models.quantizers import build_quantizer, normalized_quantizer_config

EmitFn = Callable[[str], None]


@dataclass
class ResolvedMotionFiles:
    """已经展开并带有 group 信息的 motion 文件。"""

    paths: list[Path]
    groups: list[str]


@dataclass
class MotionRuntimeBundle:
    """网络运行前需要共享的数据对象。"""

    raw: RawMotionDataset
    features: FeatureBundle
    buffer: MotionWindowBuffer

    @property
    def window_size(self) -> int:
        return self.buffer.window_size

    @property
    def robot_input_dim(self) -> int:
        return self.features.schema.robot_feature_dim * self.window_size

    @property
    def human_input_dim(self) -> int:
        return self.features.schema.human_feature_dim * self.window_size


def resolve_motion_files(config: MotionReconstructionConfig) -> ResolvedMotionFiles:
    """根据配置解析参与本次运行的 npz 文件。"""
    if config.data.motion_yaml:
        resolver = MotionSourceResolver.from_legacy_yaml(config.data.motion_yaml)
    else:
        resolver = MotionSourceResolver.from_direct_inputs(
            files=config.data.files,
            dirs=config.data.dirs,
            exclude_files=config.data.exclude_files,
            exclude_dirs=config.data.exclude_dirs,
        )
    resolved = resolver.resolve(groups=config.data.groups or None)
    pairs = resolved.file_group_pairs
    return ResolvedMotionFiles(paths=[path for path, _ in pairs], groups=[group for _, group in pairs])


def build_motion_runtime(
    config: MotionReconstructionConfig,
    *,
    device: str | torch.device,
    emit: EmitFn | None = None,
) -> MotionRuntimeBundle:
    """加载 raw motion、构建 feature，并创建 window buffer。"""
    device = torch.device(device)
    resolved = resolve_motion_files(config)
    _emit(emit, f"解析到 motion 文件: {len(resolved.paths)}")

    raw = RawMotionLoader(resolved.paths, groups=resolved.groups).load(device=device)
    _emit(emit, f"加载完成: frames={raw.num_frames}, clips={len(resolved.paths)}, fps={raw.fps}")

    features = FeatureBuilder(
        FeatureBuilderConfig(
            robot_anchor_body=config.features.robot_anchor_body,
            human_anchor_body=config.features.human_anchor_body,
            human_body_names=config.features.human_body_names,
        )
    ).build(raw)
    _emit(emit, f"特征维度: robot={features.schema.robot_feature_dim}, human={features.schema.human_feature_dim}")

    buffer = MotionWindowBuffer(
        robot_features=features.robot,
        human_features=features.human,
        motion_lengths=raw.motion_lengths,
        history=config.train.history,
        future=config.train.future,
        device=device,
    )
    _emit(
        emit,
        "窗口采样: "
        f"history={config.train.history}, future={config.train.future}, "
        f"window={buffer.window_size}, 合法中心帧={buffer.valid_center_indices.numel()}",
    )
    return MotionRuntimeBundle(raw=raw, features=features, buffer=buffer)


def build_autoencoder(
    config: MotionReconstructionConfig,
    *,
    robot_input_dim: int,
    human_input_dim: int,
    quantizer_config: dict | None = None,
) -> tuple[DualFSQAutoEncoder, dict]:
    """根据配置构建双编码器自编码器。"""
    if quantizer_config is None:
        quantizer_config = normalized_quantizer_config(config.model.quantizer.__dict__, config.model.latent_dim)
    quantizer = build_quantizer(quantizer_config, latent_dim=config.model.latent_dim)
    model = DualFSQAutoEncoder(
        robot_input_dim=robot_input_dim,
        human_input_dim=human_input_dim,
        latent_dim=config.model.latent_dim,
        robot_encoder_hidden_dims=config.model.robot_encoder_hidden_dims,
        human_encoder_hidden_dims=config.model.human_encoder_hidden_dims,
        decoder_hidden_dims=config.model.decoder_hidden_dims,
        quantizer=quantizer,
        activation=config.model.activation,
    )
    return model, quantizer_config


def _emit(emit: EmitFn | None, message: str) -> None:
    if emit is not None:
        emit(message)
