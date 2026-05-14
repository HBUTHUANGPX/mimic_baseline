from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

_MOTION_SHARDING_PATH = (
    Path(__file__).resolve().parents[1]
    / "general_motion_tracker_whole_body_teleoperation"
    / "utils"
    / "motion_sharding.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "motion_sharding_under_test",
    _MOTION_SHARDING_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
motion_sharding = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("motion_sharding_under_test", motion_sharding)
_SPEC.loader.exec_module(motion_sharding)

DistributedRuntimeInfo = motion_sharding.DistributedRuntimeInfo
FrameBalancedMotionFileShardStrategy = (
    motion_sharding.FrameBalancedMotionFileShardStrategy
)
MotionFileMetadata = motion_sharding.MotionFileMetadata
MotionMetadataReader = motion_sharding.MotionMetadataReader
NpzMotionMetadataReader = motion_sharding.NpzMotionMetadataReader


class FakeMotionMetadataReader(MotionMetadataReader):
    """Return deterministic frame counts for tests.

    Preconditions:
        The queried path must exist in ``frame_counts``.
    Postconditions:
        The returned metadata uses the input group name and configured frame count.
    """

    def __init__(self, frame_counts: dict[str, int]) -> None:
        """Store path-to-frame-count fixtures.

        Preconditions:
            ``frame_counts`` contains string paths and positive frame counts.
        Postconditions:
            Future ``read`` calls use this mapping without touching the filesystem.
        """
        self._frame_counts = frame_counts

    def read(self, path: str, group_name: str) -> MotionFileMetadata:
        """Build metadata for one motion file.

        Preconditions:
            ``path`` is a key in the fixture mapping.
        Postconditions:
            A metadata object is returned with the configured frame count.
        """
        return MotionFileMetadata(
            path=path,
            group_name=group_name,
            frame_count=self._frame_counts[path],
        )


def test_runtime_info_reads_multi_node_global_rank(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("RANK", "10")
    monkeypatch.setenv("LOCAL_RANK", "2")

    runtime = DistributedRuntimeInfo.from_environment()

    assert runtime.world_size == 16
    assert runtime.global_rank == 10
    assert runtime.local_rank == 2
    assert runtime.enabled


def test_frame_balanced_strategy_partitions_by_valid_frame_count():
    motion_file_group = {
        "walk": ["walk_long.npz", "walk_short.npz"],
        "jump": ["jump_mid.npz", "jump_tiny.npz"],
    }
    reader = FakeMotionMetadataReader(
        {
            "walk_long.npz": 100,
            "walk_short.npz": 20,
            "jump_mid.npz": 70,
            "jump_tiny.npz": 15,
        }
    )
    strategy = FrameBalancedMotionFileShardStrategy(reader)
    runtimes = [
        DistributedRuntimeInfo(global_rank=rank, local_rank=rank, world_size=2)
        for rank in range(2)
    ]

    shards = [
        strategy.shard(motion_file_group, runtime, history_frames=5, future_frames=5)
        for runtime in runtimes
    ]

    shard_paths = [
        {path for paths in shard.values() for path in paths} for shard in shards
    ]
    assert shard_paths[0].isdisjoint(shard_paths[1])
    assert shard_paths[0] | shard_paths[1] == {
        "walk_long.npz",
        "walk_short.npz",
        "jump_mid.npz",
        "jump_tiny.npz",
    }

    weights = []
    for shard in shards:
        total = 0
        for paths in shard.values():
            for path in paths:
                total += max(reader.read(path, "unused").frame_count - 10, 0)
        weights.append(total)
    assert max(weights) - min(weights) <= 20


def test_frame_balanced_strategy_preserves_group_structure():
    motion_file_group = {
        "walk": ["a.npz", "b.npz"],
        "jump": ["c.npz"],
    }
    reader = FakeMotionMetadataReader({"a.npz": 30, "b.npz": 40, "c.npz": 50})
    strategy = FrameBalancedMotionFileShardStrategy(reader)
    runtime = DistributedRuntimeInfo(global_rank=0, local_rank=0, world_size=2)

    shard = strategy.shard(
        motion_file_group,
        runtime,
        history_frames=0,
        future_frames=0,
    )

    assert set(shard).issubset({"walk", "jump"})
    assert all(isinstance(paths, list) for paths in shard.values())
    assert all(
        Path(path).suffix == ".npz" for paths in shard.values() for path in paths
    )


def test_npz_metadata_reader_reads_frame_count_without_loading_full_loader(
    tmp_path: Path,
):
    motion_path = tmp_path / "motion.npz"
    import numpy as np

    np.savez(
        motion_path,
        robot_joint_pos=np.zeros((7, 3), dtype=np.float32),
        unrelated=np.zeros((99,), dtype=np.float32),
    )
    reader = NpzMotionMetadataReader()

    metadata = reader.read(str(motion_path), "default")

    assert metadata.path == str(motion_path)
    assert metadata.group_name == "default"
    assert metadata.frame_count == 7
