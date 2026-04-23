"""checkpoint 辅助函数。

checkpoint 保存的不只是模型权重，还包含 normalizer、feature schema 和
quantizer config，保证未来复用网络或反归一化时有足够上下文。
"""

from __future__ import annotations

from pathlib import Path

import torch

from motion_reconstruction.training.normalization import WindowFeatureNormalizer


def save_checkpoint(
    *,
    output_dir: str | Path,
    name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    global_step: int,
    config: dict,
    normalizers: dict[str, WindowFeatureNormalizer],
    feature_schema: dict,
    quantizer_config: dict,
) -> Path:
    """保存一个训练 checkpoint。

    调用方决定文件名，例如 `latest.pt` 或 `epoch_0010.pt`。
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / name
    stateful_model = model.module if hasattr(model, "module") else model
    payload = {
        "model": stateful_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": config,
        "normalizers": {key: value.state_dict() for key, value in normalizers.items()},
        "feature_schema": feature_schema,
        "quantizer_config": quantizer_config,
    }
    torch.save(payload, path)
    return path
