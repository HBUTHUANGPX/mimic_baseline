"""Generic IO adapter driven entirely by a declarative model protocol."""

from __future__ import annotations

import numpy as np

from awesome_deploy.inference.buffers import BufferManager
from awesome_deploy.inference.protocol import BufferInitializer, InputBinding, ModelProtocol
from awesome_deploy.inference.types import InferenceResult, ModelSignature, RuntimeState


class ProtocolAdapter:
    """Executes protocol bindings without embedding model-specific names."""

    def __init__(self, protocol: ModelProtocol) -> None:
        """Stores the protocol used to interpret this model.

        Args:
            protocol: Declarative protocol describing all IO bindings.
        """
        self.protocol = protocol

    def initialize(self, signature: ModelSignature, buffers: BufferManager) -> None:
        """Initializes persistent buffers from declarative rules.

        Args:
            signature: Backend-reported model signature.
            buffers: Buffer manager owned by the inference engine.
        """
        for buffer_name, initializer in self.protocol.buffer_initializers.items():
            buffers.set(buffer_name, self._initialize_value(initializer, signature))

    def build_inputs(
        self,
        runtime_state: RuntimeState,
        buffers: BufferManager,
    ) -> dict[str, np.ndarray]:
        """Builds backend inputs by resolving protocol-defined sources."""
        inputs = {}
        for input_name, binding in self.protocol.input_bindings.items():
            value = self._resolve_binding(
                binding=binding,
                runtime_state=runtime_state,
                buffers=buffers,
                result=None,
            )
            inputs[input_name] = np.asarray(value)
        return inputs

    def parse_outputs(
        self,
        raw_outputs: dict[str, np.ndarray],
        buffers: BufferManager,
    ) -> InferenceResult:
        """Parses backend outputs and computes post-step buffer updates."""
        outputs = {}
        primary_action = None
        for output_name, binding in self.protocol.output_bindings.items():
            value = raw_outputs[output_name]
            if binding.transform is not None:
                value = binding.transform(value)
            else:
                value = np.asarray(value)
            if binding.target_kind == "primary":
                primary_action = np.asarray(value)
            elif binding.target_kind == "output":
                outputs[binding.target_key] = np.asarray(value)
            else:
                raise ValueError(f"Unsupported output binding target_kind: {binding.target_kind}")

        partial_result = InferenceResult(outputs=outputs, primary_action=primary_action)
        state_updates = {}
        for buffer_name, binding in self.protocol.per_step_buffer_updates.items():
            state_updates[buffer_name] = self._resolve_binding(
                binding=binding,
                runtime_state=None,
                buffers=buffers,
                result=partial_result,
            )
        partial_result.state_updates = state_updates
        return partial_result

    def _initialize_value(
        self,
        initializer: BufferInitializer,
        signature: ModelSignature,
    ):
        """Computes one initial buffer value from signature metadata."""
        if initializer.init_kind == "constant":
            return initializer.value
        if initializer.init_kind == "zeros_from_output":
            if initializer.tensor_name is None:
                raise ValueError("tensor_name is required for zeros_from_output initializers.")
            tensor_spec = signature.outputs[initializer.tensor_name]
            axis_size = tensor_spec.shape[initializer.axis]
            if axis_size is None:
                raise ValueError(
                    f"Output tensor '{initializer.tensor_name}' axis {initializer.axis} must be static."
                )
            return np.zeros(int(axis_size), dtype=np.float32)
        raise ValueError(f"Unsupported buffer initializer kind: {initializer.init_kind}")

    def _resolve_binding(
        self,
        binding: InputBinding,
        runtime_state: RuntimeState | None,
        buffers: BufferManager,
        result: InferenceResult | None,
    ):
        """Resolves one declarative binding against runtime state, buffers, or results."""
        if binding.source_kind == "state":
            if runtime_state is None or binding.source_key is None:
                raise ValueError("State binding requires runtime_state and source_key.")
            value = runtime_state.values[binding.source_key]
        elif binding.source_kind == "buffer":
            if binding.source_key is None:
                raise ValueError("Buffer binding requires source_key.")
            value = buffers.get(binding.source_key)
        elif binding.source_kind == "result":
            if result is None or binding.source_key is None:
                raise ValueError("Result binding requires parsed result and source_key.")
            if (
                binding.source_key == "primary_action"
                or binding.source_key == self._primary_target_key()
            ):
                value = result.primary_action
            else:
                value = result.outputs[binding.source_key]
        elif binding.source_kind == "constant":
            value = binding.value
        else:
            raise ValueError(f"Unsupported binding source_kind: {binding.source_kind}")

        if binding.transform is not None:
            value = binding.transform(value)
        return value

    def _primary_target_key(self) -> str | None:
        """Returns the semantic key assigned to the protocol's primary output."""
        for binding in self.protocol.output_bindings.values():
            if binding.target_kind == "primary":
                return binding.target_key
        return None
