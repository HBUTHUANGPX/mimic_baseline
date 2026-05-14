from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from typing import TypeAlias
import zipfile

from numpy.lib import format as np_format

MotionFileGroupInput: TypeAlias = Mapping[str, Sequence[str] | str] | str
MotionFileGroupDict: TypeAlias = OrderedDict[str, list[str]]


@dataclass(frozen=True)
class DistributedRuntimeInfo:
    """Describe the distributed process that owns one motion-data shard.

    Preconditions:
        ``world_size`` is positive, ``global_rank`` is in ``[0, world_size)``, and
        ``local_rank`` is non-negative.
    Postconditions:
        The object can be used by sharding strategies without reading global state.
    """

    global_rank: int
    local_rank: int
    world_size: int

    def __post_init__(self) -> None:
        """Validate rank fields after dataclass construction.

        Preconditions:
            The dataclass fields have been assigned.
        Postconditions:
            Invalid distributed settings raise ``ValueError`` immediately.
        """
        if self.world_size < 1:
            raise ValueError(f"world_size must be positive, got {self.world_size}.")
        if self.global_rank < 0 or self.global_rank >= self.world_size:
            raise ValueError(
                "global_rank must be in [0, world_size), got "
                f"{self.global_rank} for world_size {self.world_size}."
            )
        if self.local_rank < 0:
            raise ValueError(f"local_rank must be non-negative, got {self.local_rank}.")

    @property
    def enabled(self) -> bool:
        """Return whether distributed sharding should alter the file list.

        Preconditions:
            The runtime object has passed validation.
        Postconditions:
            ``True`` is returned only when more than one process participates.
        """
        return self.world_size > 1

    @classmethod
    def from_environment(cls) -> DistributedRuntimeInfo:
        """Build runtime information from torchrun-compatible environment variables.

        Preconditions:
            ``RANK``, ``LOCAL_RANK``, and ``WORLD_SIZE`` are either unset or integer
            strings. Multi-node launchers must expose global ``RANK``.
        Postconditions:
            A validated runtime object is returned; missing values default to single
            process execution.
        """
        return cls(
            global_rank=int(os.getenv("RANK", "0")),
            local_rank=int(os.getenv("LOCAL_RANK", "0")),
            world_size=int(os.getenv("WORLD_SIZE", "1")),
        )


@dataclass(frozen=True)
class MotionFileMetadata:
    """Store the lightweight metadata needed to shard one motion file.

    Preconditions:
        ``frame_count`` is the number of temporal frames in ``path``.
    Postconditions:
        ``valid_weight`` can be computed without loading full motion arrays.
    """

    path: str
    group_name: str
    frame_count: int

    def valid_weight(self, history_frames: int, future_frames: int) -> int:
        """Compute how many valid center frames this file contributes.

        Preconditions:
            ``history_frames`` and ``future_frames`` are non-negative.
        Postconditions:
            The returned weight is never negative.
        """
        if history_frames < 0 or future_frames < 0:
            raise ValueError("history_frames and future_frames must be non-negative.")
        return max(self.frame_count - history_frames - future_frames, 0)


class MotionFileGroup:
    """Normalize and rebuild grouped motion file collections.

    Preconditions:
        Input groups use string paths or sequences of string paths.
    Postconditions:
        Consumers receive ordered ``dict[group_name, list[path]]`` collections.
    """

    def __init__(self, motion_file_group: MotionFileGroupInput) -> None:
        """Create a normalized motion-file group wrapper.

        Preconditions:
            ``motion_file_group`` is a string path or a mapping of group names to
            paths.
        Postconditions:
            The normalized group preserves input order and stores each group as a list.
        """
        self._groups = self._normalize(motion_file_group)

    @property
    def groups(self) -> MotionFileGroupDict:
        """Return the normalized group mapping.

        Preconditions:
            The wrapper has been constructed successfully.
        Postconditions:
            The returned mapping is the canonical representation used by sharding.
        """
        return self._groups

    def iter_paths(self) -> list[tuple[str, str]]:
        """Return all ``(group_name, path)`` pairs in stable order.

        Preconditions:
            The wrapper has been constructed successfully.
        Postconditions:
            The list can be scanned repeatedly by deterministic strategies.
        """
        return [
            (group_name, path)
            for group_name, paths in self._groups.items()
            for path in paths
        ]

    @staticmethod
    def rebuild(metadata_items: Sequence[MotionFileMetadata]) -> MotionFileGroupDict:
        """Rebuild a grouped path mapping from selected metadata.

        Preconditions:
            ``metadata_items`` contains file metadata from a normalized group.
        Postconditions:
            Paths are grouped by ``group_name`` while preserving selected item order.
        """
        grouped: MotionFileGroupDict = OrderedDict()
        for metadata in metadata_items:
            grouped.setdefault(metadata.group_name, []).append(metadata.path)
        return grouped

    @staticmethod
    def _normalize(motion_file_group: MotionFileGroupInput) -> MotionFileGroupDict:
        """Convert accepted motion-file inputs into an ordered mapping.

        Preconditions:
            ``motion_file_group`` is a valid path or mapping.
        Postconditions:
            Empty groups are removed and unsupported values raise ``TypeError``.
        """
        if isinstance(motion_file_group, str):
            return OrderedDict({"default": [motion_file_group]})
        if not isinstance(motion_file_group, Mapping):
            raise TypeError(
                "motion_file_group must be a path string or a mapping of groups."
            )

        normalized: MotionFileGroupDict = OrderedDict()
        for group_name, paths in motion_file_group.items():
            if isinstance(paths, str):
                path_list = [paths]
            else:
                path_list = list(paths)
            if path_list:
                normalized[group_name] = path_list
        return normalized


class MotionMetadataReader(ABC):
    """Read lightweight metadata for motion files.

    Preconditions:
        Implementations must not keep full motion arrays alive.
    Postconditions:
        Callers receive enough metadata to decide distributed sharding.
    """

    @abstractmethod
    def read(self, path: str, group_name: str) -> MotionFileMetadata:
        """Read metadata for one motion file.

        Preconditions:
            ``path`` points to a readable motion file.
        Postconditions:
            A ``MotionFileMetadata`` object is returned or an exception explains why
            the file cannot be inspected.
        """


class NpzMotionMetadataReader(MotionMetadataReader):
    """Read frame counts from ``.npz`` files using ``.npy`` headers when possible.

    Preconditions:
        Motion files are NumPy ``.npz`` archives containing at least one configured
        frame-count key.
    Postconditions:
        Full arrays are not retained after metadata inspection.
    """

    def __init__(self, frame_keys: Sequence[str] | None = None) -> None:
        """Configure candidate arrays used to infer temporal length.

        Preconditions:
            ``frame_keys`` is ``None`` or a non-empty sequence of archive keys.
        Postconditions:
            The reader tries keys in order until one is present.
        """
        self._frame_keys = tuple(frame_keys or ("robot_joint_pos", "joint_pos"))
        if not self._frame_keys:
            raise ValueError("At least one frame key is required.")

    @property
    def frame_keys(self) -> tuple[str, ...]:
        """Return frame keys used by this reader.

        Preconditions:
            The reader has been constructed.
        Postconditions:
            A tuple of candidate keys is returned without exposing mutable state.
        """
        return self._frame_keys

    def read(self, path: str, group_name: str) -> MotionFileMetadata:
        """Read frame-count metadata for one ``.npz`` motion file.

        Preconditions:
            ``path`` is a readable ``.npz`` file containing one configured frame key.
        Postconditions:
            Returned metadata contains the first dimension of the selected array.
        """
        frame_count = self._read_frame_count_from_header(path)
        return MotionFileMetadata(
            path=path,
            group_name=group_name,
            frame_count=frame_count,
        )

    def _read_frame_count_from_header(self, path: str) -> int:
        """Read an array shape from a zipped ``.npy`` header.

        Preconditions:
            ``path`` points to a NumPy archive and at least one frame key exists.
        Postconditions:
            The first dimension of the selected array is returned.
        """
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            for key in self._frame_keys:
                npy_name = f"{key}.npy"
                if npy_name not in names:
                    continue
                with archive.open(npy_name) as member:
                    version = np_format.read_magic(member)
                    if version == (1, 0):
                        shape, _, _ = np_format.read_array_header_1_0(member)
                    elif version == (2, 0):
                        shape, _, _ = np_format.read_array_header_2_0(member)
                    else:
                        raise ValueError(
                            f"Unsupported .npy header version {version} in {path}."
                        )
                if len(shape) == 0:
                    raise ValueError(f"Motion array '{key}' in {path} is scalar.")
                return int(shape[0])

        raise KeyError(
            f"None of frame keys {self._frame_keys} were found in motion file {path}."
        )


@dataclass
class MotionFileShard:
    """Collect files assigned to one distributed rank.

    Preconditions:
        ``rank`` identifies the shard in ``[0, world_size)``.
    Postconditions:
        Metadata can be added while tracking cumulative valid-frame weight.
    """

    rank: int
    items: list[MotionFileMetadata]
    total_valid_weight: int = 0

    def add(self, metadata: MotionFileMetadata, valid_weight: int) -> None:
        """Add one motion file to this shard.

        Preconditions:
            ``valid_weight`` is non-negative and belongs to ``metadata``.
        Postconditions:
            The file is appended and cumulative weight is increased.
        """
        if valid_weight < 0:
            raise ValueError(f"valid_weight must be non-negative, got {valid_weight}.")
        self.items.append(metadata)
        self.total_valid_weight += valid_weight


class MotionFileShardStrategy(ABC):
    """Select the subset of motion files one distributed rank should load.

    Preconditions:
        Implementations are deterministic for identical metadata and runtime values.
    Postconditions:
        The returned group contains only files owned by the current rank.
    """

    @abstractmethod
    def shard(
        self,
        motion_file_group: MotionFileGroupInput,
        runtime: DistributedRuntimeInfo,
        history_frames: int,
        future_frames: int,
    ) -> MotionFileGroupDict:
        """Return the grouped file subset for ``runtime.global_rank``.

        Preconditions:
            ``motion_file_group`` contains readable motion files and valid runtime
            information.
        Postconditions:
            Only the current rank's files are returned.
        """


class FrameBalancedMotionFileShardStrategy(MotionFileShardStrategy):
    """Balance motion files across ranks by valid center-frame count.

    Preconditions:
        The metadata reader can inspect every input motion file.
    Postconditions:
        Each rank independently computes the same global assignment table, making the
        strategy safe for multi-node launches without process synchronization.
    """

    def __init__(self, metadata_reader: MotionMetadataReader | None = None) -> None:
        """Create a frame-balanced sharding strategy.

        Preconditions:
            ``metadata_reader`` is ``None`` or implements ``MotionMetadataReader``.
        Postconditions:
            The strategy is ready to shard normalized or raw motion-file groups.
        """
        self._metadata_reader = metadata_reader or NpzMotionMetadataReader()
        self._last_shards: list[MotionFileShard] = []

    @property
    def metadata_reader(self) -> MotionMetadataReader:
        """Return the metadata reader used by this strategy.

        Preconditions:
            The strategy has been constructed.
        Postconditions:
            Callers receive the reader without taking ownership.
        """
        return self._metadata_reader

    @property
    def last_shards(self) -> tuple[MotionFileShard, ...]:
        """Return the most recent global shard assignment.

        Preconditions:
            ``shard`` or ``build_shards`` may have been called already.
        Postconditions:
            A tuple is returned so callers can inspect assignment without mutating it.
        """
        return tuple(self._last_shards)

    def shard(
        self,
        motion_file_group: MotionFileGroupInput,
        runtime: DistributedRuntimeInfo,
        history_frames: int,
        future_frames: int,
    ) -> MotionFileGroupDict:
        """Return this rank's frame-balanced motion-file subset.

        Preconditions:
            ``runtime`` describes a valid global rank across all nodes. Every input
            file contains a positive number of frames.
        Postconditions:
            The current rank receives a non-empty grouped subset, or ``ValueError`` is
            raised with an actionable message.
        """
        normalized = MotionFileGroup(motion_file_group)
        if not runtime.enabled:
            return normalized.groups

        shards = self.build_shards(
            normalized,
            runtime.world_size,
            history_frames,
            future_frames,
        )
        local_shard = shards[runtime.global_rank]
        if not local_shard.items:
            raise ValueError(
                "Distributed motion sharding assigned no valid motion files to "
                f"rank {runtime.global_rank}/{runtime.world_size}. Reduce world_size "
                "or provide more trajectories with valid frames."
            )
        return MotionFileGroup.rebuild(local_shard.items)

    def build_shards(
        self,
        motion_file_group: MotionFileGroup,
        world_size: int,
        history_frames: int,
        future_frames: int,
    ) -> list[MotionFileShard]:
        """Build the deterministic global shard assignment table.

        Preconditions:
            ``world_size`` is positive and all paths in ``motion_file_group`` are
            readable by the metadata reader.
        Postconditions:
            A list with one shard per rank is returned; only files with positive valid
            weight are assigned.
        """
        if world_size < 1:
            raise ValueError(f"world_size must be positive, got {world_size}.")

        weighted_metadata = self._read_weighted_metadata(
            motion_file_group,
            history_frames,
            future_frames,
        )
        if not weighted_metadata:
            raise ValueError(
                "No motion files contain valid center frames for the configured "
                f"history_frames={history_frames}, future_frames={future_frames}."
            )

        weighted_metadata.sort(
            key=lambda item: (
                -item[1],
                item[0].group_name,
                item[0].path,
            )
        )
        shards = [MotionFileShard(rank=rank, items=[]) for rank in range(world_size)]
        for metadata, valid_weight in weighted_metadata:
            target_shard = min(
                shards,
                key=lambda shard: (shard.total_valid_weight, shard.rank),
            )
            target_shard.add(metadata, valid_weight)
        self._last_shards = shards
        return shards

    def _read_weighted_metadata(
        self,
        motion_file_group: MotionFileGroup,
        history_frames: int,
        future_frames: int,
    ) -> list[tuple[MotionFileMetadata, int]]:
        """Read metadata and attach valid-frame weights.

        Preconditions:
            ``motion_file_group`` contains candidate files.
        Postconditions:
            Files with no valid center frames are omitted from the returned list.
        """
        weighted_metadata: list[tuple[MotionFileMetadata, int]] = []
        for group_name, path in motion_file_group.iter_paths():
            metadata = self._metadata_reader.read(path, group_name)
            valid_weight = metadata.valid_weight(history_frames, future_frames)
            if valid_weight > 0:
                weighted_metadata.append((metadata, valid_weight))
        return weighted_metadata
