"""解析旧 YAML 或直接配置中的 motion 文件来源。

本模块只负责把“用户给出的 motion 来源描述”解析成 npz 文件列表，并保留
motion_group 元数据。它不读取 npz 内容，也不理解网络特征。
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


@dataclass(frozen=True)
class ResolvedMotionSources:
    """按 motion group 分组后的 npz 路径。

    `by_group` 是稳定的分组视图，`files` 和 `file_group_pairs` 是训练流程
    常用的扁平视图。
    """

    by_group: dict[str, list[Path]]

    @property
    def files(self) -> list[Path]:
        files: list[Path] = []
        for group_files in self.by_group.values():
            files.extend(group_files)
        return sorted(files)

    @property
    def file_group_pairs(self) -> list[tuple[Path, str]]:
        pairs: list[tuple[Path, str]] = []
        for group, group_files in self.by_group.items():
            pairs.extend((path, group) for path in group_files)
        return sorted(pairs, key=lambda item: str(item[0]))


class MotionSourceResolver:
    """收集 npz 路径，并保留 motion group 元数据。

    兼容旧工程的 `motion_group/file_name/folder_name/wo_*` YAML，也支持
    新训练配置直接传入文件和目录。
    """

    def __init__(self, by_group_spec: dict[str, dict], *, base_dir: Path | None = None):
        self._by_group_spec = by_group_spec
        self._base_dir = base_dir or Path.cwd()

    @classmethod
    def from_legacy_yaml(cls, yaml_path: str | Path) -> "MotionSourceResolver":
        """从旧版 `motion_file.yaml` 格式创建 resolver。

        默认不筛选 group，后续 `resolve(groups=...)` 再决定训练哪些组。
        """
        path = Path(yaml_path)
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        motion_groups = data.get("motion_group", {})
        if not isinstance(motion_groups, dict):
            raise ValueError(f"{path} must contain a mapping named 'motion_group'.")
        return cls(motion_groups, base_dir=path.parent)

    @classmethod
    def from_direct_inputs(
        cls,
        *,
        files: Iterable[str | Path] = (),
        dirs: Iterable[str | Path] = (),
        exclude_files: Iterable[str | Path] = (),
        exclude_dirs: Iterable[str | Path] = (),
        group: str = "default",
        base_dir: str | Path | None = None,
    ) -> "MotionSourceResolver":
        return cls(
            {
                group: {
                    "file_name": list(files),
                    "folder_name": list(dirs),
                    "wo_file_name": list(exclude_files),
                    "wo_folder_name": list(exclude_dirs),
                }
            },
            base_dir=Path(base_dir) if base_dir is not None else None,
        )

    def resolve(self, groups: Iterable[str] | None = None) -> ResolvedMotionSources:
        """将配置中的 group 解析成具体 npz 路径。

        `groups=None` 表示训练 YAML 中的全部 motion_group。
        """
        selected = set(groups) if groups else None
        by_group: dict[str, list[Path]] = {}

        for group_name, spec in self._by_group_spec.items():
            if selected is not None and group_name not in selected:
                continue
            by_group[group_name] = self._resolve_group(spec or {})

        if selected is not None:
            missing = selected.difference(by_group)
            if missing:
                raise ValueError(f"Unknown motion groups: {sorted(missing)}")
        if not by_group:
            raise ValueError("No motion groups were resolved.")
        return ResolvedMotionSources(by_group=by_group)

    def _resolve_group(self, spec: dict) -> list[Path]:
        files = _as_list(spec.get("file_name"))
        folders = _as_list(spec.get("folder_name"))
        exclude_files = {_normalize_path(path, self._base_dir) for path in _as_list(spec.get("wo_file_name"))}
        exclude_dirs = [_normalize_path(path, self._base_dir) for path in _as_list(spec.get("wo_folder_name"))]

        paths: set[Path] = set()
        seen_basenames: set[str] = set()

        for item in files:
            path = _normalize_path(item, self._base_dir)
            if path.suffix == ".npz" and path.exists():
                paths.add(path)
                seen_basenames.add(path.name)

        for folder_item in folders:
            folder = _normalize_path(folder_item, self._base_dir)
            if not folder.is_dir():
                continue
            pattern = str(folder / "**" / "*.npz")
            for match in glob.glob(pattern, recursive=True):
                path = Path(match).resolve()
                # 中文：沿用旧脚本逻辑，目录扫描遇到与显式文件同 basename 的 npz 时跳过。
                if path.name in seen_basenames:
                    continue
                paths.add(path)
                seen_basenames.add(path.name)

        for path in exclude_files:
            paths.discard(path)
        for folder in exclude_dirs:
            if not folder.is_dir():
                continue
            for match in glob.glob(str(folder / "**" / "*.npz"), recursive=True):
                paths.discard(Path(match).resolve())

        return sorted(paths)


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [value]
    return list(value)


def _normalize_path(path: str | Path, base_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (base_dir / candidate).resolve()
