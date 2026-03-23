"""Protocol description objects for generic model IO binding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


Transform = Callable[[Any], Any]


@dataclass(frozen=True)
class InputBinding:
    """Describes how one model input or buffer update obtains its value.

    Attributes:
        source_kind: Value origin. Supported values are ``"state"``,
            ``"buffer"``, ``"result"``, and ``"constant"``.
        source_key: Key used to fetch the value from the selected source.
        value: Constant payload used when ``source_kind == "constant"``.
        transform: Optional callable applied to the resolved value before it is
            passed forward.
    """

    source_kind: str
    source_key: str | None = None
    value: Any = None
    transform: Transform | None = None


@dataclass(frozen=True)
class OutputBinding:
    """Describes how one raw model output should be interpreted.

    Attributes:
        target_kind: Interpretation target. Supported values are ``"output"``
            for named passthrough outputs and ``"primary"`` for the main action.
        target_key: Semantic name assigned to the parsed output.
        transform: Optional callable applied to the raw output tensor.
    """

    target_kind: str
    target_key: str
    transform: Transform | None = None


@dataclass(frozen=True)
class BufferInitializer:
    """Describes how one persistent buffer should be initialized.

    Attributes:
        init_kind: Initialization strategy. Supported values are
            ``"constant"`` and ``"zeros_from_output"``.
        value: Constant initialization payload for ``"constant"`` buffers.
        tensor_name: Output tensor name used to infer buffer size when
            ``init_kind == "zeros_from_output"``.
        axis: Axis within the referenced tensor whose dimension is used to
            allocate a zero buffer.
    """

    init_kind: str
    value: Any = None
    tensor_name: str | None = None
    axis: int = 0


@dataclass(frozen=True)
class ModelProtocol:
    """Declarative description of model IO semantics.

    Attributes:
        input_bindings: Mapping from backend input tensor name to value binding.
        output_bindings: Mapping from backend output tensor name to output
            interpretation.
        buffer_initializers: Named buffer initialization rules applied after the
            backend signature becomes available.
        per_step_buffer_updates: Named buffer updates evaluated after each
            successful inference step.
    """

    input_bindings: dict[str, InputBinding]
    output_bindings: dict[str, OutputBinding]
    buffer_initializers: dict[str, BufferInitializer] = field(default_factory=dict)
    per_step_buffer_updates: dict[str, InputBinding] = field(default_factory=dict)
