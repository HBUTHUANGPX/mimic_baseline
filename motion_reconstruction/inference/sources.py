"""将不同来源的 motion 数据整理成统一推理输入。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from motion_reconstruction.config.schema import MotionReconstructionConfig
from motion_reconstruction.features.rotation import (
    quat_inverse_rotate_wxyz,
    quat_to_rot6d_wxyz,
)
from motion_reconstruction.human_pose import compute_visualized_global_joint_transforms_xyzw
from motion_reconstruction.pipeline import ResolvedMotionFiles, build_motion_runtime


@dataclass
class InferenceSourceBundle:
    """统一的推理输入。

    `robot_features` 在 human-only 来源中可以为 `None`。
    """

    fps: int
    center_indices: torch.Tensor
    window_offsets: torch.Tensor
    robot_features: torch.Tensor | None
    human_features: torch.Tensor
    robot_anchor_pos_w: torch.Tensor
    human_body_pos_w: torch.Tensor
    robot_joint_names: list[str]
    robot_body_names: list[str]
    human_body_names: list[str]
    robot_anchor_body: str
    human_anchor_body: str
    display_human_body_names: list[str]


def build_inference_source(
    *,
    source: str,
    config: MotionReconstructionConfig,
    device: str | torch.device,
    feature_schema: dict[str, Any],
    motion_npz: str | Path | None = None,
    resolved: ResolvedMotionFiles | None = None,
    emit=None,
    progress: bool | None = None,
) -> InferenceSourceBundle:
    if source == "raw":
        return build_raw_source(
            config=config,
            device=device,
            resolved=resolved,
            emit=emit,
            progress=progress,
        )
    if source == "hdf5-human":
        if motion_npz is None:
            raise ValueError("source='hdf5-human' 时必须提供 motion_npz。")
        return build_hdf5_human_source(
            motion_npz=motion_npz,
            config=config,
            feature_schema=feature_schema,
            device=device,
        )
    raise ValueError(f"不支持的 source: {source}")


def build_raw_source(
    *,
    config: MotionReconstructionConfig,
    device: str | torch.device,
    resolved: ResolvedMotionFiles | None = None,
    emit=None,
    progress: bool | None = None,
) -> InferenceSourceBundle:
    runtime = build_motion_runtime(
        config,
        device=device,
        emit=emit,
        resolved=resolved,
        progress=progress,
    )
    raw = runtime.raw
    anchor_index = raw.robot_body_names.index(config.features.robot_anchor_body)
    return InferenceSourceBundle(
        fps=int(raw.fps),
        center_indices=runtime.buffer.valid_center_indices,
        window_offsets=runtime.buffer.window_offsets,
        robot_features=runtime.buffer.robot_features,
        human_features=runtime.buffer.human_features,
        robot_anchor_pos_w=raw.body_pos_w[:, anchor_index],
        human_body_pos_w=raw.human_body_pos_w,
        robot_joint_names=list(raw.robot_joint_names),
        robot_body_names=list(raw.robot_body_names),
        human_body_names=list(raw.human_body_names),
        robot_anchor_body=config.features.robot_anchor_body,
        human_anchor_body=config.features.human_anchor_body,
        display_human_body_names=configured_display_human_body_names(config, raw.human_body_names),
    )


def build_hdf5_human_source(
    *,
    motion_npz: str | Path,
    config: MotionReconstructionConfig,
    feature_schema: dict[str, Any],
    device: str | torch.device,
) -> InferenceSourceBundle:
    motion_path = Path(motion_npz)
    if not motion_path.is_file():
        raise FileNotFoundError(f"找不到 human motion npz: {motion_path}")
    with np.load(motion_path, allow_pickle=True) as data:
        fps = int(np.asarray(data["fps"]).item())
        scalar_first = _read_scalar_first(data)
        human_body_names = _read_names(data, "human_joint_names", "human_body_names")
        if "human_local_transforms" in data and "human_parent_indices" in data:
            human_global_pos_np, human_global_quat_np = _global_pose_from_local_transforms(
                local_transforms=np.asarray(data["human_local_transforms"], dtype=np.float32),
                parent_indices=np.asarray(data["human_parent_indices"], dtype=np.int32),
                scalar_first=scalar_first,
            )
            human_global_pos = torch.as_tensor(human_global_pos_np, device=device)
            human_global_quat = torch.as_tensor(_to_wxyz(human_global_quat_np, False), device=device)
        elif "human_global_pos" in data and "human_global_quat" in data:
            human_global_pos = torch.as_tensor(np.asarray(data["human_global_pos"], dtype=np.float32), device=device)
            human_global_quat = torch.as_tensor(
                _to_wxyz(np.asarray(data["human_global_quat"], dtype=np.float32), scalar_first),
                device=device,
            )
        else:
            raise KeyError(
                "human motion npz 必须提供 human_local_transforms/human_parent_indices，"
                "或提供 human_global_pos/human_global_quat 作为回退。"
            )

    human_features = build_human_features(
        human_body_pos_w=human_global_pos,
        human_body_quat_w=human_global_quat,
        human_body_names=human_body_names,
        human_anchor_body=config.features.human_anchor_body,
        human_body_names_for_feature=config.features.human_body_names,
    )
    num_frames = int(human_global_pos.shape[0])
    history = int(config.train.history)
    future = int(config.train.future)
    center_indices = _build_center_indices(num_frames=num_frames, history=history, future=future, device=device)
    window_offsets = torch.arange(-history, future + 1, dtype=torch.long, device=device)
    anchor_index = human_body_names.index(config.features.human_anchor_body)
    robot_joint_names = [str(name) for name in feature_schema.get("robot_joint_names", [])]
    robot_body_names = [str(name) for name in feature_schema.get("robot_body_names", [])]
    return InferenceSourceBundle(
        fps=fps,
        center_indices=center_indices,
        window_offsets=window_offsets,
        robot_features=None,
        human_features=human_features,
        robot_anchor_pos_w=human_global_pos[:, anchor_index],
        human_body_pos_w=human_global_pos,
        robot_joint_names=robot_joint_names,
        robot_body_names=robot_body_names,
        human_body_names=human_body_names,
        robot_anchor_body=config.features.robot_anchor_body,
        human_anchor_body=config.features.human_anchor_body,
        display_human_body_names=configured_display_human_body_names(config, human_body_names),
    )


def build_human_features(
    *,
    human_body_pos_w: torch.Tensor,
    human_body_quat_w: torch.Tensor,
    human_body_names: list[str],
    human_anchor_body: str,
    human_body_names_for_feature: list[str],
) -> torch.Tensor:
    anchor_index = _index(human_body_names, human_anchor_body, "human body")
    selected_indices = [
        _index(human_body_names, name, "human body") for name in human_body_names_for_feature
    ]

    human_anchor_quat = human_body_quat_w[:, anchor_index]
    human_anchor_rot6d = quat_to_rot6d_wxyz(human_anchor_quat)
    anchor_pos = human_body_pos_w[:, anchor_index]
    selected_pos = human_body_pos_w[:, selected_indices]
    rel_world = selected_pos - anchor_pos[:, None, :]
    expanded_anchor_quat = human_anchor_quat[:, None, :].expand(-1, len(selected_indices), -1)
    rel_anchor = quat_inverse_rotate_wxyz(expanded_anchor_quat, rel_world).reshape(human_body_pos_w.shape[0], -1)
    return torch.cat((human_anchor_rot6d, rel_anchor), dim=-1)


def configured_display_human_body_names(
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


def _build_center_indices(*, num_frames: int, history: int, future: int, device: str | torch.device) -> torch.Tensor:
    start = int(history)
    end = max(num_frames - int(future), start)
    if end <= start:
        return torch.empty((0,), dtype=torch.long, device=device)
    return torch.arange(start, end, dtype=torch.long, device=device)


def _read_names(data: np.lib.npyio.NpzFile, *names: str) -> list[str]:
    for name in names:
        if name in data:
            return [str(value) for value in np.asarray(data[name]).tolist()]
    raise KeyError(f"缺少名字字段，期望其中之一: {names}")


def _read_scalar_first(data: np.lib.npyio.NpzFile) -> bool:
    if "scalar_first" not in data:
        return True
    value = np.asarray(data["scalar_first"])
    return bool(value.item() if value.shape == () else value.reshape(-1)[0])


def _to_wxyz(quat: np.ndarray, scalar_first: bool) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    if quat.shape[-1] != 4:
        raise ValueError(f"quaternion 最后一维必须为 4，得到 {quat.shape}")
    if scalar_first:
        return quat
    return quat[..., [3, 0, 1, 2]]


def _index(names: list[str], name: str, label: str) -> int:
    try:
        return names.index(name)
    except ValueError as exc:
        raise ValueError(f"Unknown {label} '{name}'. Available names: {names}") from exc


def _global_pose_from_local_transforms(
    *,
    local_transforms: np.ndarray,
    parent_indices: np.ndarray,
    scalar_first: bool,
) -> tuple[np.ndarray, np.ndarray]:
    local_transforms = np.asarray(local_transforms, dtype=np.float32).copy()
    if scalar_first:
        local_transforms[..., 3:7] = local_transforms[..., [4, 5, 6, 3]]
    return compute_visualized_global_joint_transforms_xyzw(local_transforms, np.asarray(parent_indices, dtype=np.int32))
