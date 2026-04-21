"""YAML 配置加载。

将 YAML 映射递归加载成 dataclass。未知字段目前会被忽略，保持第一版简单。
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .schema import MotionReconstructionConfig

T = TypeVar("T")


def load_config(path: str | Path) -> MotionReconstructionConfig:
    """将 YAML 配置加载为 `MotionReconstructionConfig`。"""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping.")
    return _from_dict(MotionReconstructionConfig, raw)


def _from_dict(cls: type[T], values: dict[str, Any]) -> T:
    """从 mapping 递归填充嵌套 dataclass 字段。"""
    kwargs: dict[str, Any] = {}
    for field_info in fields(cls):
        if field_info.name not in values:
            continue
        value = values[field_info.name]
        default_obj = getattr(cls(), field_info.name)
        if is_dataclass(default_obj) and isinstance(value, dict):
            kwargs[field_info.name] = _from_dict(type(default_obj), value)
        else:
            kwargs[field_info.name] = value
    return cls(**kwargs)
