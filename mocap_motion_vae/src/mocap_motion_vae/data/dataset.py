"""窗口化数据集与索引器。

本模块将 MotionBank 转为可直接训练的窗口化数据集。调用链如下：

1) WindowIndex:
   - 根据 clip_lengths、window、stride 计算可用窗口数量
   - 将 dataset 索引映射到 (clip_index, start_index_in_clip)
2) MotionWindowDataset:
   - 根据 WindowIndex 找到窗口起点
   - 使用 MotionView 在全局时间轴上批量索引
   - 按 FeatureSpec 构建 inputs / targets / static

该设计保证训练侧只需面向 Dataset 接口，而不关心底层拼接细节。
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset

from .bank import FeatureSpec, MotionBank, MotionSample, MotionView


@dataclass
class WindowIndex:
    """将数据集索引映射到 (clip_index, start_index_in_clip)。

    Attributes:
        clip_lengths: 每个 clip 的长度（帧数）。
        window: 窗口长度（帧数）。
        stride: 滑动步长（帧数）。
        drop_last: 是否丢弃不足窗口长度的片段。
    """

    clip_lengths: Tuple[int, ...]
    window: int
    stride: int
    drop_last: bool = True

    def __post_init__(self) -> None:
        """预计算索引映射所需的累积计数。

        该过程会为每个 clip 计算可采样的窗口数量，并累积成
        数据集索引到 clip 的映射表。
        """
        # 预计算每个 clip 能取到的 window 数量
        if self.window <= 0:
            raise ValueError("window must be > 0.")
        if self.stride <= 0:
            raise ValueError("stride must be > 0.")
        counts = []
        for length in self.clip_lengths:
            if length < self.window:
                counts.append(0)
                continue
            count = 1 + (length - self.window) // self.stride
            counts.append(count)
        self._counts = counts
        self._cum_counts = []
        total = 0
        for count in counts:
            total += count
            self._cum_counts.append(total)
        self._total = total

    def __len__(self) -> int:
        """返回可采样的窗口总数。"""
        return self._total

    def locate(self, index: int) -> Tuple[int, int]:
        """定位数据集索引到对应的 clip 和起始帧。

        Args:
            index: 数据集索引。

        Returns:
            (clip_index, start_index_in_clip)。

        Raises:
            IndexError: index 超出范围。
        """
        # 根据累计计数定位所属 clip，并得到 window 起点
        if index < 0 or index >= self._total:
            raise IndexError("index out of range.")
        clip_idx = bisect.bisect_right(self._cum_counts, index)
        prev = 0 if clip_idx == 0 else self._cum_counts[clip_idx - 1]
        local = index - prev
        start = local * self.stride
        return clip_idx, start


class MotionWindowDataset(Dataset[MotionSample]):
    """从 MotionBank 中采样固定窗口长度的数据。"""

    def __init__(
        self,
        bank: MotionBank,
        window: int,
        stride: int,
        feature_spec: FeatureSpec,
        drop_last: bool = True,
    ) -> None:
        """构建窗口数据集。

        Args:
            bank: MotionBank 实例。
            window: 窗口长度（帧数）。
            stride: 滑动步长（帧数）。
            feature_spec: 输入/输出字段配置。
            drop_last: 是否丢弃不足窗口长度的片段。

        Raises:
            ValueError: 输入字段为空。
        """
        self.bank = bank
        self.window = int(window)
        self.stride = int(stride)
        self.feature_spec = feature_spec
        self.indexer = WindowIndex(tuple(bank.clip_lengths), self.window, self.stride, drop_last)

        if len(self.feature_spec.inputs) == 0:
            raise ValueError("FeatureSpec.inputs must be non-empty.")

    def __len__(self) -> int:
        """返回数据集长度（窗口数量）。"""
        return len(self.indexer)

    def __getitem__(self, index: int) -> MotionSample:
        """获取一个窗口样本。

        Args:
            index: 数据集索引。

        Returns:
            MotionSample 样本，包含 inputs/targets/static。
        """
        # 1) 定位到具体 clip 和窗口起点
        clip_idx, start = self.indexer.locate(index)
        clip_start = int(self.bank.clip_indices[clip_idx, 0].item())
        time_idx = clip_start + torch.arange(
            start, start + self.window, device=self.bank.device, dtype=torch.long
        )
        # 2) 构造 MotionView，通过 time_idx 自动批量索引
        view = MotionView(self.bank, time_idx)

        # 3) 构造模型输入（可拼接或字典）
        if self.feature_spec.concat_inputs:
            inputs = view.concat(*self.feature_spec.inputs)
        else:
            inputs = {name: view.field(name) for name in self.feature_spec.inputs}

        # 4) 构造模型输出（可选）
        targets = None
        if len(self.feature_spec.targets) > 0:
            if self.feature_spec.concat_targets:
                targets = view.concat(*self.feature_spec.targets)
            else:
                targets = {name: view.field(name) for name in self.feature_spec.targets}

        # 5) 静态字段（按 clip_id 取）
        static = {name: view.static(name) for name in self.feature_spec.static}

        return MotionSample(
            inputs=inputs,
            targets=targets,
            static=static,
            time_indices=time_idx,
            clip_index=clip_idx,
        )
