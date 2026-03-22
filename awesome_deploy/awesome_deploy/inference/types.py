from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


ArrayDict = dict[str, np.ndarray]


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int | None, ...]
    dtype: str


@dataclass(frozen=True)
class ModelSignature:
    inputs: dict[str, TensorSpec]
    outputs: dict[str, TensorSpec]


@dataclass
class InferenceContext:
    obs: np.ndarray
    time_step: int
    command: np.ndarray | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    outputs: ArrayDict
    primary_action: np.ndarray | None = None
    state_updates: dict[str, Any] = field(default_factory=dict)
