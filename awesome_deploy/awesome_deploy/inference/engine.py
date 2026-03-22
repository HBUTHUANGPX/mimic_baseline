"""Inference engine that composes backend execution, IO adapters, and buffers."""

from __future__ import annotations

from awesome_deploy.inference.buffers import BufferManager
from awesome_deploy.inference.types import (
    InferenceContext,
    InferenceResult,
    ModelSignature,
)


class InferenceEngine:
    """Coordinates one complete inference pipeline.

    The engine is intentionally backend-agnostic. A backend knows how to run a
    model, an IO adapter knows how to translate semantic context into tensors
    and back, and the buffer manager holds any persistent state needed across
    timesteps.
    """

    def __init__(
        self, backend, io_adapter, model_path: str, device: str = "cpu"
    ) -> None:
        """Builds an engine around one backend and one IO adapter.

        Args:
            backend: Concrete backend implementation such as ``OnnxBackend``.
            io_adapter: Adapter that maps runtime context to backend tensors.
            model_path: Path to the serialized model file.
            device: Device identifier forwarded to the backend loader.
        """
        self.backend = backend
        self.io_adapter = io_adapter
        self.model_path = model_path
        self.device = device
        self.buffers = BufferManager()
        self.signature: ModelSignature | None = None

    def load(self) -> None:
        """Loads the model and initializes adapter-managed buffers."""
        self.backend.load(self.model_path, device=self.device)
        self.signature = self.backend.get_signature()
        self.io_adapter.initialize(self.signature, self.buffers)

    def reset(self) -> None:
        """Resets runtime buffers while keeping the loaded model resident."""
        self.buffers.reset()
        if self.signature is not None:
            self.io_adapter.initialize(self.signature, self.buffers)

    def step(self, context: InferenceContext) -> InferenceResult:
        """Executes one inference step from semantic context to semantic result.

        Args:
            context: Per-step runtime information collected from the simulator.

        Returns:
            Parsed inference result with named outputs and buffer updates.
        """
        # Convert simulator-side semantic state into backend tensor inputs.
        raw_inputs = self.io_adapter.build_inputs(context, self.buffers)
        # Execute the concrete backend without exposing backend details to the
        # caller.
        raw_outputs = self.backend.infer(raw_inputs)
        # Parse raw tensors back into named semantic outputs.
        result = self.io_adapter.parse_outputs(raw_outputs, self.buffers)
        # Apply state updates last so the caller observes a consistent result
        # from the current step.
        self.buffers.update(result.state_updates)
        return result
