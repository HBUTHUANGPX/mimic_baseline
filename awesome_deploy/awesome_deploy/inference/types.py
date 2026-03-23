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
class RuntimeState:
    """Opaque runtime resource container consumed by the inference engine.

    Attributes:
        values: Named runtime resources collected from the simulator, controller,
            or task layer. The engine does not interpret their semantic meaning;
            it only exposes them to protocol bindings.
    """

    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Semantic result returned by one inference step.

    Attributes:
        outputs: Named outputs promoted by the protocol.
        primary_action: Main action vector to be consumed by the simulator when
            the protocol marks one output as the primary action.
        state_updates: Buffer mutations that should be committed after the step
            succeeds.
    """

    outputs: ArrayDict
    primary_action: np.ndarray | None = None
    state_updates: dict[str, Any] = field(default_factory=dict)
