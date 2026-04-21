"""基于 dataclass 的配置 schema。

配置层负责给 YAML 提供稳定结构和默认值，不包含训练逻辑。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from motion_reconstruction.features.builder import DEFAULT_HUMAN_BODY_NAMES


@dataclass
class DataConfig:
    """motion 数据来源配置。

    `motion_yaml` 兼容旧工程；`files/dirs/exclude_*` 支持直接配置。
    """

    motion_yaml: str | None = None
    files: list[str] = field(default_factory=list)
    dirs: list[str] = field(default_factory=list)
    exclude_files: list[str] = field(default_factory=list)
    exclude_dirs: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)


@dataclass
class FeatureConfig:
    """feature 构建配置。"""

    robot_anchor_body: str = "torso_link"
    human_anchor_body: str = "Hips"
    human_body_names: list[str] = field(default_factory=lambda: list(DEFAULT_HUMAN_BODY_NAMES))


@dataclass
class QuantizerConfig:
    """FSQ/iFSQ quantizer 配置。"""

    type: str = "ifsq"
    levels: int | list[int] = 17
    do_simple_bound: bool = True
    act_fun: str = "scale_sigmoid_16"
    eps: float = 1e-3


@dataclass
class ModelConfig:
    """双 encoder autoencoder 网络结构配置。"""

    latent_dim: int = 16
    robot_encoder_hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    human_encoder_hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    decoder_hidden_dims: list[int] = field(default_factory=lambda: [256, 512])
    activation: str = "elu"
    quantizer: QuantizerConfig = field(default_factory=QuantizerConfig)


@dataclass
class LossConfig:
    """MSE 组合损失的权重配置。"""

    robot_recon: float = 1.0
    human_recon: float = 1.0
    latent_align: float = 0.25
    cycle_latent: float = 0.25


@dataclass
class TrainConfig:
    """训练循环、采样、日志和 checkpoint 配置。"""

    device: str = "cuda"
    epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    history: int = 5
    future: int = 5
    seed: int = 1
    log_every_steps: int = 20
    log_histograms: bool = True
    progress: bool = True
    checkpoint_interval_epochs: int = 10
    normalizer_eps: float = 1e-2


@dataclass
class OutputConfig:
    """训练输出目录配置。"""

    root_dir: str = "outputs/motion_reconstruction"
    run_name: str | None = None


@dataclass
class MotionReconstructionConfig:
    """包级顶层配置。"""

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
