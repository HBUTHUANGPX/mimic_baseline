from __future__ import annotations

import onnxruntime as ort

from awesome_deploy.inference.types import ArrayDict, ModelSignature, TensorSpec


class OnnxBackend:
    def __init__(self) -> None:
        self.session: ort.InferenceSession | None = None

    def load(self, model_path: str, device: str = "cpu") -> None:
        providers = (
            ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]
        )
        self.session = ort.InferenceSession(model_path, providers=providers)

    def get_signature(self) -> ModelSignature:
        if self.session is None:
            raise RuntimeError("Backend session is not loaded.")
        return ModelSignature(
            inputs={
                meta.name: TensorSpec(
                    name=meta.name,
                    shape=tuple(meta.shape),
                    dtype=str(meta.type),
                )
                for meta in self.session.get_inputs()
            },
            outputs={
                meta.name: TensorSpec(
                    name=meta.name,
                    shape=tuple(meta.shape),
                    dtype=str(meta.type),
                )
                for meta in self.session.get_outputs()
            },
        )

    def infer(self, inputs: ArrayDict) -> ArrayDict:
        if self.session is None:
            raise RuntimeError("Backend session is not loaded.")
        output_names = [meta.name for meta in self.session.get_outputs()]
        output_values = self.session.run(output_names, inputs)
        return dict(zip(output_names, output_values))
