from __future__ import annotations

"""MotionBank 与 MotionView 的基础数据结构与视图工具。

本模块提供“全局时间轴 + 批量索引视图”的核心抽象，调用链如下：

1) ClipData: 表示单个动作片段（帧级字段 + 静态字段 + 元信息）。
2) MotionBank:
   - 将多个 ClipData 的帧级字段拼接为全局时间轴
   - 维护 clip_indices / clip_ids / new_clip_flag 等索引结构
3) MotionView:
   - 接收 time_steps（可为 (T,) 或 (B, T)）
   - 使用 tensor[time_steps] 完成批量索引
   - concat/concat_static 会自动展平多维特征（如 joints: (T, J, 3)）
4) FeatureSpec / MotionSample:
   - 规定模型输入/输出字段的组织方式
   - 为上层 Dataset 或训练循环提供统一样本结构

设计目标：
- 让训练侧仅关心“哪些字段作为输入/输出”，而不需要关心底层拼接逻辑；
- 保持对 Tensor 索引广播/自动扩张的自然支持。
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch


TensorLike = torch.Tensor


@dataclass
class ClipData:
    """单个动作片段数据（帧级字段 + 静态字段）。

    Attributes:
        name: 片段名称或标识。
        fps: 帧率（Hz）。
        frames: 帧级字段，形状通常为 (T, D)。
        static: 静态字段，与时间无关。
        meta: 其他元信息。
    """

    name: str
    fps: float
    frames: Dict[str, TensorLike]
    static: Dict[str, TensorLike]
    meta: Dict[str, object]


@dataclass
class FeatureSpec:
    """定义如何从 MotionBank 构建模型输入/输出。

    Attributes:
        inputs: 输入字段名列表（帧级）。
        targets: 输出字段名列表（帧级）。
        static: 静态字段名列表。
        concat_inputs: 是否将输入字段拼接为一个张量。
        concat_targets: 是否将输出字段拼接为一个张量。
    """

    inputs: Tuple[str, ...]
    targets: Tuple[str, ...] = ()
    static: Tuple[str, ...] = ()
    concat_inputs: bool = True
    concat_targets: bool = True


@dataclass
class MotionSample:
    """单个训练样本（inputs/targets + 元数据）。

    Attributes:
        inputs: 模型输入（张量或字典）。
        targets: 模型输出（张量或字典，可为空）。
        static: 静态字段（字典）。
        time_indices: 全局时间轴索引。
        clip_index: 所属片段索引。
    """

    inputs: Union[TensorLike, Dict[str, TensorLike]]
    targets: Optional[Union[TensorLike, Dict[str, TensorLike]]]
    static: Dict[str, TensorLike]
    time_indices: TensorLike
    clip_index: int


class MotionBank:
    """将多个动作片段拼接到全局时间轴的 Bank。

    该类负责将多个 clip 的帧级字段拼接成单一时间轴，并提供
    clip 索引、起止区间等辅助信息。
    """

    def __init__(
        self,
        frames: Dict[str, TensorLike],
        clip_lengths: Sequence[int],
        clip_names: Sequence[str],
        clip_fps: Sequence[float],
        static: Optional[Dict[str, TensorLike]] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """构建 MotionBank。

        Args:
            frames: 帧级字段字典，必须包含相同的总帧数。
            clip_lengths: 每个片段的长度（帧数）。
            clip_names: 每个片段的名称。
            clip_fps: 每个片段的帧率。
            static: 静态字段字典（按片段堆叠）。
            device: 张量放置设备。

        Raises:
            ValueError: 参数为空或长度不一致，或帧数不匹配。
        """
        # 基本合法性检查
        if len(clip_lengths) == 0:
            raise ValueError("clip_lengths must be non-empty.")
        if len(clip_lengths) != len(clip_names):
            raise ValueError("clip_lengths and clip_names must have the same length.")
        if len(clip_lengths) != len(clip_fps):
            raise ValueError("clip_lengths and clip_fps must have the same length.")
        if len(frames) == 0:
            raise ValueError("frames must be non-empty.")

        self.device = torch.device(device)
        self.frames: Dict[str, TensorLike] = {
            k: torch.as_tensor(v, device=self.device) for k, v in frames.items()
        }
        self.static: Dict[str, TensorLike] = {}
        if static is not None:
            self.static = {k: torch.as_tensor(v, device=self.device) for k, v in static.items()}

        self.clip_lengths = list(int(x) for x in clip_lengths)
        self.clip_names = list(clip_names)
        self.clip_fps = list(float(x) for x in clip_fps)

        # 总帧数 = 各片段长度之和
        total_frames = sum(self.clip_lengths)
        for name, tensor in self.frames.items():
            if tensor.shape[0] != total_frames:
                raise ValueError(
                    f"Frame field '{name}' has {tensor.shape[0]} frames, "
                    f"expected {total_frames}."
                )

        self.num_clips = len(self.clip_lengths)
        self.num_frames = total_frames

        # clip_indices: [start, end) 用于从全局时间轴定位某个 clip
        self.clip_indices = torch.zeros(self.num_clips, 2, dtype=torch.long, device=self.device)
        start = 0
        for i, length in enumerate(self.clip_lengths):
            end = start + length
            self.clip_indices[i] = torch.tensor([start, end], dtype=torch.long, device=self.device)
            start = end

        # new_clip_flag: 标记每个 clip 的起始帧（除第一个外为 True）
        self.new_clip_flag = torch.zeros(self.num_frames, dtype=torch.bool, device=self.device)
        offset = 0
        for i, length in enumerate(self.clip_lengths):
            if i > 0:
                self.new_clip_flag[offset] = True
            offset += length

        # clip_ids: 每一帧对应的 clip id
        self.clip_ids = torch.empty(self.num_frames, dtype=torch.long, device=self.device)
        offset = 0
        for i, length in enumerate(self.clip_lengths):
            self.clip_ids[offset : offset + length] = i
            offset += length

    @classmethod
    def from_clips(
        cls,
        clips: Sequence[ClipData],
        device: Union[str, torch.device] = "cpu",
    ) -> "MotionBank":
        """从 ClipData 列表构建 MotionBank。

        Args:
            clips: ClipData 列表。
            device: 张量放置设备。

        Returns:
            MotionBank 实例。

        Raises:
            ValueError: clips 为空或字段不一致。
        """
        # 统一帧级/静态字段名
        if len(clips) == 0:
            raise ValueError("clips must be non-empty.")

        frame_fields = set(clips[0].frames.keys())
        static_fields = set(clips[0].static.keys())
        for clip in clips[1:]:
            if set(clip.frames.keys()) != frame_fields:
                raise ValueError("All clips must share the same frame field keys.")
            if set(clip.static.keys()) != static_fields:
                raise ValueError("All clips must share the same static field keys.")

        frame_lists: Dict[str, List[TensorLike]] = {k: [] for k in frame_fields}
        static_lists: Dict[str, List[TensorLike]] = {k: [] for k in static_fields}
        clip_lengths: List[int] = []
        clip_names: List[str] = []
        clip_fps: List[float] = []

        for clip in clips:
            if len(clip.frames) == 0:
                raise ValueError("Clip frames must be non-empty.")
            lengths = {v.shape[0] for v in clip.frames.values()}
            if len(lengths) != 1:
                raise ValueError(f"Clip '{clip.name}' has inconsistent frame lengths: {lengths}")
            length = lengths.pop()
            clip_lengths.append(length)
            clip_names.append(clip.name)
            clip_fps.append(clip.fps)
            for name, value in clip.frames.items():
                frame_lists[name].append(value)
            for name, value in clip.static.items():
                static_lists[name].append(value)

        # 拼接成全局时间轴
        frames = {name: torch.cat(values, dim=0) for name, values in frame_lists.items()}
        static = {}
        if len(static_lists) > 0:
            # 静态字段按 clip 维度堆叠 (num_clips, D)
            static = {name: torch.stack(values, dim=0) for name, values in static_lists.items()}

        return cls(
            frames=frames,
            clip_lengths=clip_lengths,
            clip_names=clip_names,
            clip_fps=clip_fps,
            static=static,
            device=device,
        )


class MotionView:
    """基于 time_steps 的批量视图（核心：Tensor 索引自动扩张）。

    通过 time_steps 对 MotionBank 的帧级字段进行索引，
    自动支持 (T,) 或 (B, T) 等索引形状。
    """

    def __init__(self, bank: MotionBank, time_steps: TensorLike) -> None:
        """创建视图。

        Args:
            bank: MotionBank 实例。
            time_steps: 全局时间轴索引。
        """
        self.bank = bank
        self.time_steps = time_steps

    def _index(self, tensor: TensorLike) -> TensorLike:
        return tensor[self.time_steps]

    def _flatten_feature(self, tensor: TensorLike) -> TensorLike:
        """按 time_steps 维度保留前缀维度，其余维度展平为特征维。

        Args:
            tensor: 输入张量。

        Returns:
            展平后的张量。
        """
        lead_dims = int(self.time_steps.ndim)
        if tensor.ndim <= lead_dims + 1:
            return tensor
        return tensor.reshape(*tensor.shape[:lead_dims], -1)

    def field(self, name: str) -> TensorLike:
        """读取帧级字段（会按 time_steps 取 batch）。

        Args:
            name: 字段名。

        Returns:
            索引后的张量。

        Raises:
            KeyError: 字段名不存在。
        """
        if name not in self.bank.frames:
            raise KeyError(f"Unknown frame field '{name}'.")
        return self._index(self.bank.frames[name])

    def static(self, name: str) -> TensorLike:
        """读取静态字段（先映射到 clip_id，再按 clip 取值）。

        Args:
            name: 静态字段名。

        Returns:
            索引后的静态张量。

        Raises:
            KeyError: 字段名不存在。
        """
        if name not in self.bank.static:
            raise KeyError(f"Unknown static field '{name}'.")
        clip_ids = self._index(self.bank.clip_ids)
        return self.bank.static[name][clip_ids]

    def concat(self, *names: str) -> TensorLike:
        """将多个帧级字段按最后一维拼接。

        Args:
            *names: 帧级字段名列表。

        Returns:
            拼接后的张量。

        Raises:
            ValueError: names 为空。
        """
        if len(names) == 0:
            raise ValueError("concat expects at least one field name.")
        tensors = [self._flatten_feature(self.field(name)) for name in names]
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=-1)

    def concat_static(self, *names: str) -> TensorLike:
        """将多个静态字段拼接。

        Args:
            *names: 静态字段名列表。

        Returns:
            拼接后的张量。

        Raises:
            ValueError: names 为空。
        """
        if len(names) == 0:
            raise ValueError("concat_static expects at least one field name.")
        tensors = [self._flatten_feature(self.static(name)) for name in names]
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=-1)

    @property
    def clip_id(self) -> TensorLike:
        """当前 time_steps 对应的 clip id。"""
        return self._index(self.bank.clip_ids)

    @property
    def new_clip_flag(self) -> TensorLike:
        """当前 time_steps 对应的新片段起始标记。"""
        return self._index(self.bank.new_clip_flag)
