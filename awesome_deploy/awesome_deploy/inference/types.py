"""Shared datatypes for backend-agnostic policy inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

ArrayDict = dict[str, np.ndarray]


@dataclass(frozen=True)
class TensorSpec:
    """Describes one named tensor declared by a serialized model.

    Attributes:
        name: Exported tensor name used by the backend runtime.
        shape: Declared tensor shape. Unknown dimensions are represented as
            ``None``.
        dtype: Runtime-reported tensor dtype string.
    """

    name: str
    shape: tuple[int | None, ...]
    dtype: str


@dataclass(frozen=True)
class ModelSignature:
    """Collects named input and output tensor metadata for one model."""

    inputs: dict[str, TensorSpec]
    outputs: dict[str, TensorSpec]


@dataclass
class InferenceContext:
    """Semantic runtime context consumed by ``InferenceEngine.step``.

    Attributes:
        obs: Flattened observation vector for the current policy step.
        time_step: Logical rollout step index consumed by some exported models.
        command: Optional high-level command vector associated with the step.
        extras: Arbitrary named runtime state for adapters that require more
            than the default observation and step counter.
    """

    obs: np.ndarray
    time_step: int
    command: np.ndarray | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Semantic result returned by one inference step.

    Attributes:
        outputs: All backend outputs keyed by model output name.
        primary_action: Main action vector to be consumed by the simulator when
            the model exposes one.
        state_updates: Buffer mutations that should be committed after the step
            succeeds.
    """

    outputs: ArrayDict
    primary_action: np.ndarray | None = None
    state_updates: dict[str, Any] = field(default_factory=dict)
