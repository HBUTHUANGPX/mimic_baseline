from __future__ import annotations

from awesome_deploy.inference.buffers import BufferManager
from awesome_deploy.inference.types import InferenceContext, InferenceResult, ModelSignature


class InferenceEngine:
    def __init__(self, backend, io_adapter, model_path: str, device: str = "cpu") -> None:
        self.backend = backend
        self.io_adapter = io_adapter
        self.model_path = model_path
        self.device = device
        self.buffers = BufferManager()
        self.signature: ModelSignature | None = None

    def load(self) -> None:
        self.backend.load(self.model_path, device=self.device)
        self.signature = self.backend.get_signature()
        self.io_adapter.initialize(self.signature, self.buffers)

    def reset(self) -> None:
        self.buffers.reset()
        if self.signature is not None:
            self.io_adapter.initialize(self.signature, self.buffers)

    def step(self, context: InferenceContext) -> InferenceResult:
        raw_inputs = self.io_adapter.build_inputs(context, self.buffers)
        raw_outputs = self.backend.infer(raw_inputs)
        result = self.io_adapter.parse_outputs(raw_outputs, self.buffers)
        self.buffers.update(result.state_updates)
        return result
