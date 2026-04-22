"""从 checkpoint 生成重构结果和基础误差。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from motion_reconstruction.config.schema import MotionReconstructionConfig
from motion_reconstruction.pipeline import build_autoencoder, build_motion_runtime
from motion_reconstruction.training.normalization import WindowFeatureNormalizer


@dataclass
class ReconstructionResult:
    """当前帧重构结果。

    decoder 输出完整窗口，这里默认导出窗口中心帧，方便评估和 MuJoCo 播放。
    """

    fps: int
    center_indices: np.ndarray
    original_robot_feature: np.ndarray
    recon_from_robot_feature: np.ndarray
    recon_from_human_feature: np.ndarray
    robot_anchor_pos_w: np.ndarray
    human_body_pos_w: np.ndarray
    robot_joint_names: list[str]
    robot_body_names: list[str]
    human_body_names: list[str]
    robot_anchor_body: str
    human_anchor_body: str
    display_human_body_names: list[str] | None = None

    def metrics(self) -> dict[str, float]:
        """返回当前帧 robot feature 的基础 MSE。"""
        original = self.original_robot_feature
        recon_robot = self.recon_from_robot_feature
        recon_human = self.recon_from_human_feature
        return {
            "robot_from_robot_mse": _mse(original, recon_robot),
            "robot_from_human_mse": _mse(original, recon_human),
            "joint_from_robot_mse": _mse(original[:, 6:], recon_robot[:, 6:]),
            "joint_from_human_mse": _mse(original[:, 6:], recon_human[:, 6:]),
        }

    def save_npz(self, path: str | Path) -> Path:
        """保存重构结果，便于后处理脚本复用。"""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            fps=np.array(self.fps),
            center_indices=self.center_indices,
            original_robot_feature=self.original_robot_feature,
            recon_from_robot_feature=self.recon_from_robot_feature,
            recon_from_human_feature=self.recon_from_human_feature,
            robot_anchor_pos_w=self.robot_anchor_pos_w,
            human_body_pos_w=self.human_body_pos_w,
            robot_joint_names=np.asarray(self.robot_joint_names, dtype=object),
            robot_body_names=np.asarray(self.robot_body_names, dtype=object),
            human_body_names=np.asarray(self.human_body_names, dtype=object),
            display_human_body_names=np.asarray(
                self.display_human_body_names or self.human_body_names,
                dtype=object,
            ),
            robot_anchor_body=np.array(self.robot_anchor_body),
            human_anchor_body=np.array(self.human_anchor_body),
        )
        return output_path

    def save_metrics_json(self, path: str | Path) -> Path:
        """保存基础评估指标。"""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(self.metrics(), file, ensure_ascii=False, indent=2)
            file.write("\n")
        return output_path


def reconstruct_motion(
    *,
    config: MotionReconstructionConfig,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    batch_size: int = 4096,
    max_frames: int | None = None,
) -> ReconstructionResult:
    """加载数据和 checkpoint，并生成当前帧重构结果。"""
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0。")
    device = torch.device(device)
    payload = torch.load(checkpoint_path, map_location=device)
    runtime = build_motion_runtime(config, device=device)
    raw = runtime.raw
    features = runtime.features
    buffer = runtime.buffer
    model, _ = build_autoencoder(
        config,
        robot_input_dim=runtime.robot_input_dim,
        human_input_dim=runtime.human_input_dim,
        quantizer_config=payload.get("quantizer_config"),
    )
    model = model.to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    normalizers = {
        name: WindowFeatureNormalizer.from_state_dict(state, device=device)
        for name, state in payload["normalizers"].items()
    }

    centers = buffer.valid_center_indices
    if max_frames is not None:
        centers = centers[: int(max_frames)]
    if centers.numel() == 0:
        raise ValueError("没有可评估的中心帧。")

    original_chunks: list[torch.Tensor] = []
    recon_robot_chunks: list[torch.Tensor] = []
    recon_human_chunks: list[torch.Tensor] = []
    history_index = int(config.train.history)
    robot_dim = int(features.schema.robot_feature_dim)

    with torch.no_grad():
        for start in range(0, centers.numel(), int(batch_size)):
            batch_centers = centers[start : start + int(batch_size)]
            window_indices = batch_centers[:, None] + buffer.window_offsets[None, :]
            robot_window = buffer.robot_features[window_indices]
            human_window = buffer.human_features[window_indices]
            robot_flat = robot_window.reshape(robot_window.shape[0], -1)
            human_flat = human_window.reshape(human_window.shape[0], -1)
            output = model(normalizers["robot"](robot_flat), normalizers["human"](human_flat))
            recon_from_robot = normalizers["robot"].inverse(output.recon_from_robot).view(
                -1, buffer.window_size, robot_dim
            )
            recon_from_human = normalizers["robot"].inverse(output.recon_from_human).view(
                -1, buffer.window_size, robot_dim
            )
            original_chunks.append(robot_window[:, history_index].detach().cpu())
            recon_robot_chunks.append(recon_from_robot[:, history_index].detach().cpu())
            recon_human_chunks.append(recon_from_human[:, history_index].detach().cpu())

    center_cpu = centers.detach().cpu()
    anchor_index = raw.robot_body_names.index(config.features.robot_anchor_body)
    display_human_body_names = _configured_display_human_body_names(config, raw.human_body_names)
    return ReconstructionResult(
        fps=int(raw.fps),
        center_indices=center_cpu.numpy(),
        original_robot_feature=torch.cat(original_chunks, dim=0).numpy(),
        recon_from_robot_feature=torch.cat(recon_robot_chunks, dim=0).numpy(),
        recon_from_human_feature=torch.cat(recon_human_chunks, dim=0).numpy(),
        robot_anchor_pos_w=raw.body_pos_w[centers, anchor_index].detach().cpu().numpy(),
        human_body_pos_w=raw.human_body_pos_w[centers].detach().cpu().numpy(),
        robot_joint_names=list(raw.robot_joint_names),
        robot_body_names=list(raw.robot_body_names),
        human_body_names=list(raw.human_body_names),
        robot_anchor_body=config.features.robot_anchor_body,
        human_anchor_body=config.features.human_anchor_body,
        display_human_body_names=display_human_body_names,
    )


def _configured_display_human_body_names(
    config: MotionReconstructionConfig,
    source_names: list[str],
) -> list[str]:
    source_set = set(source_names)
    names: list[str] = []
    for name in [config.features.human_anchor_body, *config.features.human_body_names]:
        if name not in source_set:
            raise ValueError(f"human body '{name}' 不存在，无法用于可视化。可用名字: {source_names}")
        if name not in names:
            names.append(name)
    return names


def _mse(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0:
        return 0.0
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.mean(diff * diff))
