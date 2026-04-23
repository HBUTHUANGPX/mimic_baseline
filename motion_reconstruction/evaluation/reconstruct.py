"""从 checkpoint 生成重构结果和基础误差。"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from motion_reconstruction.config.io import _from_dict
from motion_reconstruction.config.schema import MotionReconstructionConfig
from motion_reconstruction.inference.sources import InferenceSourceBundle, build_inference_source
from motion_reconstruction.pipeline import build_autoencoder
from motion_reconstruction.training.normalization import WindowFeatureNormalizer


@dataclass
class ReconstructionResult:
    """当前帧重构结果。

    decoder 输出完整窗口，这里默认导出窗口中心帧，方便评估和 MuJoCo 播放。
    human-only source 下，robot 原始分支可以为空。
    """

    fps: int
    center_indices: np.ndarray
    original_robot_feature: np.ndarray | None
    recon_from_robot_feature: np.ndarray | None
    recon_from_human_feature: np.ndarray | None
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
        metrics: dict[str, float] = {}
        original = self.original_robot_feature
        if original is None:
            return metrics
        if self.recon_from_robot_feature is not None:
            recon_robot = self.recon_from_robot_feature
            metrics.update(
                {
                    "robot_from_robot_mse": _mse(original, recon_robot),
                    "joint_from_robot_mse": _mse(original[:, 6:], recon_robot[:, 6:]),
                }
            )
        if self.recon_from_human_feature is not None:
            recon_human = self.recon_from_human_feature
            metrics.update(
                {
                    "robot_from_human_mse": _mse(original, recon_human),
                    "joint_from_human_mse": _mse(original[:, 6:], recon_human[:, 6:]),
                }
            )
        return metrics

    def save_npz(self, path: str | Path) -> Path:
        """保存重构结果，便于后处理脚本复用。"""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "fps": np.array(self.fps),
            "center_indices": self.center_indices,
            "robot_anchor_pos_w": self.robot_anchor_pos_w,
            "human_body_pos_w": self.human_body_pos_w,
            "robot_joint_names": np.asarray(self.robot_joint_names, dtype=object),
            "robot_body_names": np.asarray(self.robot_body_names, dtype=object),
            "human_body_names": np.asarray(self.human_body_names, dtype=object),
            "display_human_body_names": np.asarray(
                self.display_human_body_names or self.human_body_names,
                dtype=object,
            ),
            "robot_anchor_body": np.array(self.robot_anchor_body),
            "human_anchor_body": np.array(self.human_anchor_body),
        }
        if self.original_robot_feature is not None:
            payload["original_robot_feature"] = self.original_robot_feature
        if self.recon_from_robot_feature is not None:
            payload["recon_from_robot_feature"] = self.recon_from_robot_feature
        if self.recon_from_human_feature is not None:
            payload["recon_from_human_feature"] = self.recon_from_human_feature
        np.savez(output_path, **payload)
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
    source: str = "raw",
    motion_npz: str | Path | None = None,
    inference_path: str = "both",
) -> ReconstructionResult:
    """加载数据和 checkpoint，并生成当前帧重构结果。"""
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0。")
    if inference_path not in {"robot", "human", "both"}:
        raise ValueError("inference_path 必须是 robot、human 或 both。")
    device = torch.device(device)
    payload = torch.load(checkpoint_path, map_location=device)
    inference_config = build_inference_config(config, payload)
    source_bundle = build_inference_source(
        source=source,
        config=inference_config,
        device=device,
        feature_schema=payload.get("feature_schema", {}),
        motion_npz=motion_npz,
        progress=False,
    )
    model, normalizers, history_index, robot_dim = load_reconstruction_modules(
        config=inference_config,
        checkpoint_payload=payload,
        device=device,
    )
    return reconstruct_from_source_bundle(
        source=source_bundle,
        model=model,
        normalizers=normalizers,
        history_index=history_index,
        robot_dim=robot_dim,
        inference_path=inference_path,
        batch_size=batch_size,
        max_frames=max_frames,
    )


def load_reconstruction_modules(
    *,
    config: MotionReconstructionConfig,
    checkpoint_payload: dict[str, Any],
    device: str | torch.device,
) -> tuple[object, dict[str, WindowFeatureNormalizer], int, int]:
    """根据 checkpoint 构建模型和归一化器。"""
    device = torch.device(device)
    feature_schema = checkpoint_payload.get("feature_schema", {})
    robot_feature_dim = int(feature_schema.get("robot_feature_dim", 0))
    human_feature_dim = int(feature_schema.get("human_feature_dim", 0))
    window_size = int(config.train.history) + int(config.train.future) + 1
    model, _ = build_autoencoder(
        config,
        robot_input_dim=robot_feature_dim * window_size,
        human_input_dim=human_feature_dim * window_size,
        quantizer_config=checkpoint_payload.get("quantizer_config"),
    )
    model = model.to(device)
    model.load_state_dict(checkpoint_payload["model"])
    model.eval()
    normalizers = {
        name: WindowFeatureNormalizer.from_state_dict(state, device=device)
        for name, state in checkpoint_payload["normalizers"].items()
    }
    return model, normalizers, int(config.train.history), robot_feature_dim


def build_inference_config(
    base_config: MotionReconstructionConfig,
    checkpoint_payload: dict[str, Any],
) -> MotionReconstructionConfig:
    """构建推理期配置。

    数据来源保留用户当前配置，模型结构、feature 选择和窗口长度优先使用 checkpoint
    里固化的训练元数据，避免 YAML 漂移导致维度不匹配。
    """
    merged = deepcopy(base_config)
    saved_config_raw = checkpoint_payload.get("config")
    if not isinstance(saved_config_raw, dict):
        return merged
    saved_config = _from_dict(MotionReconstructionConfig, saved_config_raw)
    merged.model = deepcopy(saved_config.model)
    merged.features = deepcopy(saved_config.features)
    merged.train.history = int(saved_config.train.history)
    merged.train.future = int(saved_config.train.future)
    return merged


def reconstruct_from_source_bundle(
    *,
    source: InferenceSourceBundle,
    model,
    normalizers: dict[str, Any],
    history_index: int,
    robot_dim: int,
    inference_path: str,
    batch_size: int = 4096,
    max_frames: int | None = None,
) -> ReconstructionResult:
    """对统一 source bundle 执行重构。"""
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0。")
    if inference_path not in {"robot", "human", "both"}:
        raise ValueError("inference_path 必须是 robot、human 或 both。")
    centers = source.center_indices
    if max_frames is not None:
        centers = centers[: int(max_frames)]
    if centers.numel() == 0:
        raise ValueError("没有可评估的中心帧。")
    if inference_path in {"robot", "both"} and source.robot_features is None:
        raise ValueError("当前 source 不包含 robot_features，无法执行 robot 推理路径。")

    original_chunks: list[torch.Tensor] = []
    recon_robot_chunks: list[torch.Tensor] = []
    recon_human_chunks: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, centers.numel(), int(batch_size)):
            batch_centers = centers[start : start + int(batch_size)]
            window_indices = batch_centers[:, None] + source.window_offsets[None, :]
            human_window = source.human_features[window_indices]
            human_flat = human_window.reshape(human_window.shape[0], -1)
            human_norm = normalizers["human"](human_flat)

            robot_flat: torch.Tensor | None = None
            robot_norm: torch.Tensor | None = None
            if source.robot_features is not None:
                robot_window = source.robot_features[window_indices]
                robot_flat = robot_window.reshape(robot_window.shape[0], -1)
                robot_norm = normalizers["robot"](robot_flat)
            if inference_path in {"robot", "both"} and robot_norm is None:
                raise ValueError("robot_features 为空，无法执行 robot 推理路径。")

            recon_robot_flat, recon_human_flat = _run_inference_path(
                model=model,
                robot_norm=robot_norm,
                human_norm=human_norm,
                inference_path=inference_path,
            )

            if robot_flat is not None and inference_path in {"robot", "both"}:
                original_center = robot_flat.view(-1, source.window_offsets.numel(), robot_dim)[:, history_index]
                original_chunks.append(original_center.detach().cpu())
            if recon_robot_flat is not None:
                recon_center = normalizers["robot"].inverse(recon_robot_flat).view(
                    -1, source.window_offsets.numel(), robot_dim
                )[:, history_index]
                recon_robot_chunks.append(recon_center.detach().cpu())
            if recon_human_flat is not None:
                recon_center = normalizers["robot"].inverse(recon_human_flat).view(
                    -1, source.window_offsets.numel(), robot_dim
                )[:, history_index]
                recon_human_chunks.append(recon_center.detach().cpu())

    center_cpu = centers.detach().cpu()
    return ReconstructionResult(
        fps=int(source.fps),
        center_indices=center_cpu.numpy(),
        original_robot_feature=_concat_or_none(original_chunks),
        recon_from_robot_feature=_concat_or_none(recon_robot_chunks),
        recon_from_human_feature=_concat_or_none(recon_human_chunks),
        robot_anchor_pos_w=source.robot_anchor_pos_w[centers].detach().cpu().numpy(),
        human_body_pos_w=source.human_body_pos_w[centers].detach().cpu().numpy(),
        robot_joint_names=list(source.robot_joint_names),
        robot_body_names=list(source.robot_body_names),
        human_body_names=list(source.human_body_names),
        robot_anchor_body=source.robot_anchor_body,
        human_anchor_body=source.human_anchor_body,
        display_human_body_names=list(source.display_human_body_names),
    )


def _run_inference_path(
    *,
    model,
    robot_norm: torch.Tensor | None,
    human_norm: torch.Tensor,
    inference_path: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if inference_path == "both":
        if robot_norm is None:
            raise ValueError("执行 both 推理路径时需要 robot_norm。")
        output = model(robot_norm, human_norm)
        return output.recon_from_robot, output.recon_from_human
    if inference_path == "robot":
        if robot_norm is None:
            raise ValueError("执行 robot 推理路径时需要 robot_norm。")
        if all(hasattr(model, name) for name in ("robot_encoder", "shared_quantizer", "decoder")):
            latent = model.robot_encoder(robot_norm)
            quantized = model.shared_quantizer(latent)
            return model.decoder(quantized.z_q), None
        output = model(robot_norm, human_norm)
        return output.recon_from_robot, None
    if all(hasattr(model, name) for name in ("human_encoder", "shared_quantizer", "decoder")):
        latent = model.human_encoder(human_norm)
        quantized = model.shared_quantizer(latent)
        return None, model.decoder(quantized.z_q)
    empty_robot = _empty_like_robot_input(robot_norm=robot_norm, human_norm=human_norm, model=model)
    output = model(empty_robot, human_norm)
    return None, output.recon_from_human


def _empty_like_robot_input(*, robot_norm: torch.Tensor | None, human_norm: torch.Tensor, model) -> torch.Tensor:
    if robot_norm is not None:
        return torch.zeros_like(robot_norm)
    input_dim = int(getattr(model, "robot_input_dim", 0))
    return torch.zeros((human_norm.shape[0], input_dim), dtype=human_norm.dtype, device=human_norm.device)


def _concat_or_none(chunks: list[torch.Tensor]) -> np.ndarray | None:
    if not chunks:
        return None
    return torch.cat(chunks, dim=0).numpy()


def _mse(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0:
        return 0.0
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.mean(diff * diff))
