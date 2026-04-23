"""motion 文件分片。

第一版按每个文件可采样的中心帧数量做贪心均衡分片，目标是让每个 rank
持有相近的数据量，同时保持实现足够直观。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from motion_reconstruction.data.raw_motion import RawMotionLoader, RawMotionMetadata


@dataclass
class MotionFileShard:
    """单个 rank 负责的文件子集。"""

    paths: list[Path]
    groups: list[str]
    metadata: list[RawMotionMetadata]
    frame_count: int
    valid_center_count: int


@dataclass
class MotionShardPlan:
    """完整分片规划。"""

    shards: list[MotionFileShard]
    metadata: list[RawMotionMetadata]

    @property
    def total_frames(self) -> int:
        return sum(item.num_frames for item in self.metadata)

    @property
    def total_valid_centers(self) -> int:
        return sum(item.valid_center_count for item in self.metadata)


def build_motion_shard_plan(
    *,
    files: Sequence[str | Path],
    groups: Sequence[str] | None,
    history: int,
    future: int,
    world_size: int,
    progress: bool,
) -> MotionShardPlan:
    """扫描元数据并按可用中心帧数量均衡分片。"""
    metadata = RawMotionLoader(files, groups=groups).scan(
        history=history,
        future=future,
        progress=progress,
    )
    shards = _empty_shards(world_size)
    ordered = sorted(
        metadata,
        key=lambda item: (item.valid_center_count, item.num_frames, str(item.path)),
        reverse=True,
    )
    for item in ordered:
        target = min(
            range(world_size),
            key=lambda index: (shards[index].valid_center_count, shards[index].frame_count, index),
        )
        shard = shards[target]
        shard.paths.append(Path(item.path))
        shard.groups.append(item.group)
        shard.metadata.append(item)
        shard.frame_count += item.num_frames
        shard.valid_center_count += item.valid_center_count

    empty_ranks = [index for index, shard in enumerate(shards) if shard.valid_center_count <= 0]
    if empty_ranks:
        raise ValueError(
            "存在未分到可训练样本的 rank。"
            f"当前 world_size={world_size}, empty_ranks={empty_ranks}。"
            "请减少进程数，或调整 history/future 以保证每个分片都有合法中心帧。"
        )
    return MotionShardPlan(shards=shards, metadata=metadata)


def _empty_shards(world_size: int) -> list[MotionFileShard]:
    return [
        MotionFileShard(paths=[], groups=[], metadata=[], frame_count=0, valid_center_count=0)
        for _ in range(world_size)
    ]
