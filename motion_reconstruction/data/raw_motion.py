"""带严格 schema 校验的 raw npz motion 加载。

本模块把多个 npz 文件加载成一个连续 raw motion buffer。这里保留原始语义
字段，做严格 schema 校验，并把 quaternion 统一成内部 wxyz 格式。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tqdm.auto import tqdm


# 中文：兼容参考工程和新导出文件的字段命名差异，输出字段名固定为参考
# `_motion_data_np_list_to_tensor` 的语义。
FIELD_ALIASES = {
    "joint_pos": ("joint_pos", "robot_joint_pos"),
    "joint_vel": ("joint_vel", "robot_joint_vel"),
    "body_pos_w": ("body_pos_w", "robot_body_pos"),
    "body_quat_w": ("body_quat_w", "robot_body_quat"),
    "body_lin_vel_w": ("body_lin_vel_w", "robot_body_lin_vel"),
    "body_ang_vel_w": ("body_ang_vel_w", "robot_body_ang_vel"),
    "human_body_pos_w": ("human_body_pos_w", "human_global_pos"),
    "human_body_quat_w": ("human_body_quat_w", "human_global_quat"),
}


@dataclass
class RawMotionDataset:
    """拼接后的 raw motion tensor 和 schema metadata。

    所有张量按时间维拼接，但 `motion_lengths`/`motion_start_indices`
    保留 clip 边界，window 采样时不会跨 motion。
    """

    fps: int
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    human_body_pos_w: torch.Tensor
    human_body_quat_w: torch.Tensor
    robot_joint_names: list[str]
    robot_body_names: list[str]
    human_body_names: list[str]
    motion_lengths: torch.Tensor
    motion_start_indices: torch.Tensor
    motion_groups: list[str]
    motion_paths: list[str]

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])


@dataclass
class RawMotionMetadata:
    """单个 raw motion 文件的轻量元数据。"""

    path: str
    group: str
    fps: int
    num_frames: int
    valid_center_count: int
    robot_joint_names: list[str]
    robot_body_names: list[str]
    human_body_names: list[str]


class RawMotionLoader:
    """将多个 npz 文件加载为一个严格且名字稳定的 raw dataset。

    第一版要求同一次训练内所有 npz 的关节/body 名字和顺序完全一致；
    不做隐式重排，避免静默训练错 schema。
    """

    def __init__(self, files: Sequence[str | Path], groups: Sequence[str] | None = None):
        if not files:
            raise ValueError("At least one motion file is required.")
        self.files = [Path(path) for path in files]
        if groups is None:
            self.groups = ["default"] * len(self.files)
        else:
            if len(groups) != len(self.files):
                raise ValueError("groups length must match files length.")
            self.groups = list(groups)

    def load(
        self,
        device: str | torch.device = "cpu",
        *,
        progress: bool = True,
    ) -> RawMotionDataset:
        """加载全部文件，并将 tensor 放到指定 device。

        可以直接加载到 GPU，后续 window 采样和训练都在同一 device 上完成。
        """
        tensors: dict[str, list[np.ndarray]] = {name: [] for name in FIELD_ALIASES}
        fps: int | None = None
        robot_joint_names: list[str] | None = None
        robot_body_names: list[str] | None = None
        human_body_names: list[str] | None = None
        lengths: list[int] = []

        file_iter = _make_progress(
            self.files,
            progress=progress,
            total=len(self.files),
            desc="加载 raw motion",
            unit="file",
        )
        for path in file_iter:
            if not path.is_file():
                raise FileNotFoundError(f"Invalid motion file: {path}")
            with np.load(path, allow_pickle=True) as data:
                file_fps = int(np.asarray(data["fps"]).item())
                fps = file_fps if fps is None else fps
                if file_fps != fps:
                    raise ValueError(f"All motion files must have the same fps. Got {file_fps} for {path}.")

                current_robot_joint_names = _read_names(data, ("robot_joint_names", "joint_names"))
                current_robot_body_names = _read_names(data, ("robot_body_names", "body_names"))
                current_human_body_names = _read_names(data, ("human_body_names", "human_joint_names"))
                robot_joint_names = _check_names(
                    "robot_joint_names", robot_joint_names, current_robot_joint_names, path
                )
                robot_body_names = _check_names("robot_body_names", robot_body_names, current_robot_body_names, path)
                human_body_names = _check_names("human_body_names", human_body_names, current_human_body_names, path)

                scalar_first = _read_scalar_first(data)
                for canonical, aliases in FIELD_ALIASES.items():
                    array = _read_array(data, aliases, path)
                    if canonical in ("body_quat_w", "human_body_quat_w"):
                        # 中文：npz 允许逐文件声明 quaternion 是否 scalar-first；
                        # 内部统一为 wxyz，FeatureBuilder 不再关心原始顺序。
                        array = _to_wxyz(array, scalar_first)
                    tensors[canonical].append(np.asarray(array, dtype=np.float32))
                lengths.append(int(tensors["joint_pos"][-1].shape[0]))

        starts = np.cumsum([0] + lengths[:-1], dtype=np.int64)
        tensor_kwargs = {"device": torch.device(device), "dtype": torch.float32}

        return RawMotionDataset(
            fps=int(fps),
            joint_pos=torch.as_tensor(np.concatenate(tensors["joint_pos"], axis=0), **tensor_kwargs),
            joint_vel=torch.as_tensor(np.concatenate(tensors["joint_vel"], axis=0), **tensor_kwargs),
            body_pos_w=torch.as_tensor(np.concatenate(tensors["body_pos_w"], axis=0), **tensor_kwargs),
            body_quat_w=torch.as_tensor(np.concatenate(tensors["body_quat_w"], axis=0), **tensor_kwargs),
            body_lin_vel_w=torch.as_tensor(np.concatenate(tensors["body_lin_vel_w"], axis=0), **tensor_kwargs),
            body_ang_vel_w=torch.as_tensor(np.concatenate(tensors["body_ang_vel_w"], axis=0), **tensor_kwargs),
            human_body_pos_w=torch.as_tensor(np.concatenate(tensors["human_body_pos_w"], axis=0), **tensor_kwargs),
            human_body_quat_w=torch.as_tensor(np.concatenate(tensors["human_body_quat_w"], axis=0), **tensor_kwargs),
            robot_joint_names=robot_joint_names or [],
            robot_body_names=robot_body_names or [],
            human_body_names=human_body_names or [],
            motion_lengths=torch.as_tensor(lengths, dtype=torch.long, device=device),
            motion_start_indices=torch.as_tensor(starts, dtype=torch.long, device=device),
            motion_groups=self.groups,
            motion_paths=[str(path) for path in self.files],
        )

    def scan(
        self,
        *,
        history: int = 0,
        future: int = 0,
        progress: bool = True,
    ) -> list[RawMotionMetadata]:
        """扫描文件 schema 和 frame 数，不构建大张量。"""
        if history < 0 or future < 0:
            raise ValueError("history and future must be non-negative.")
        fps: int | None = None
        robot_joint_names: list[str] | None = None
        robot_body_names: list[str] | None = None
        human_body_names: list[str] | None = None
        metadata: list[RawMotionMetadata] = []

        file_iter = _make_progress(
            zip(self.files, self.groups, strict=True),
            progress=progress,
            total=len(self.files),
            desc="扫描 raw motion",
            unit="file",
        )
        for path, group in file_iter:
            if not path.is_file():
                raise FileNotFoundError(f"Invalid motion file: {path}")
            with np.load(path, allow_pickle=True) as data:
                file_fps = int(np.asarray(data["fps"]).item())
                fps = file_fps if fps is None else fps
                if file_fps != fps:
                    raise ValueError(f"All motion files must have the same fps. Got {file_fps} for {path}.")

                current_robot_joint_names = _read_names(data, ("robot_joint_names", "joint_names"))
                current_robot_body_names = _read_names(data, ("robot_body_names", "body_names"))
                current_human_body_names = _read_names(data, ("human_body_names", "human_joint_names"))
                robot_joint_names = _check_names(
                    "robot_joint_names", robot_joint_names, current_robot_joint_names, path
                )
                robot_body_names = _check_names("robot_body_names", robot_body_names, current_robot_body_names, path)
                human_body_names = _check_names("human_body_names", human_body_names, current_human_body_names, path)

                num_frames = int(_read_array(data, FIELD_ALIASES["joint_pos"], path).shape[0])
                metadata.append(
                    RawMotionMetadata(
                        path=str(path),
                        group=str(group),
                        fps=file_fps,
                        num_frames=num_frames,
                        valid_center_count=max(num_frames - history - future, 0),
                        robot_joint_names=list(robot_joint_names or []),
                        robot_body_names=list(robot_body_names or []),
                        human_body_names=list(human_body_names or []),
                    )
                )
        return metadata


def _read_array(data: np.lib.npyio.NpzFile, names: tuple[str, ...], path: Path) -> np.ndarray:
    for name in names:
        if name in data:
            return np.asarray(data[name])
    raise KeyError(f"{path} does not contain any of {names}.")


def _read_names(data: np.lib.npyio.NpzFile, names: tuple[str, ...]) -> list[str]:
    for name in names:
        if name in data:
            values = np.asarray(data[name]).tolist()
            return [str(value) for value in values]
    raise KeyError(f"Missing names field. Expected one of {names}.")


def _check_names(name: str, expected: list[str] | None, current: list[str], path: Path) -> list[str]:
    if expected is None:
        return current
    if expected != current:
        raise ValueError(f"{name} mismatch in {path}. Expected {expected}, got {current}.")
    return expected


def _read_scalar_first(data: np.lib.npyio.NpzFile) -> bool:
    """返回 quaternion 数据是否按 wxyz 存储。

    缺省按 scalar-first 处理，兼容已有多数 motion 文件。
    """
    if "scalar_first" not in data:
        return True
    value = np.asarray(data["scalar_first"])
    return bool(value.item() if value.shape == () else value.reshape(-1)[0])


def _to_wxyz(quat: np.ndarray, scalar_first: bool) -> np.ndarray:
    if quat.shape[-1] != 4:
        raise ValueError(f"Quaternion arrays must have final dim 4, got {quat.shape}.")
    if scalar_first:
        return quat
    return quat[..., [3, 0, 1, 2]]


def _make_progress(iterable, *, progress: bool, **kwargs):
    if not progress:
        return iterable
    return tqdm(iterable, disable=False, dynamic_ncols=True, **kwargs)
