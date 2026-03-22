"""ONNX Runtime backend for deployment-time neural policy inference."""

from __future__ import annotations

import onnxruntime as ort

from awesome_deploy.inference.types import ArrayDict, ModelSignature, TensorSpec


class OnnxBackend:
    """Wraps ``onnxruntime.InferenceSession`` behind a uniform backend API.

    The backend is intentionally narrow: it only knows how to load a model,
    expose static signature metadata, and execute one inference call. Input and
    output semantics are handled by higher-level IO adapters.
    """

    def __init__(self) -> None:
        """Initializes an unloaded ONNX backend."""
        self.session: ort.InferenceSession | None = None

    def load(self, model_path: str, device: str = "cpu") -> None:
        """Loads an ONNX model from disk.

        Args:
            model_path: Absolute or package-relative path to ``.onnx`` file.
            device: Target execution device. ``"cpu"`` selects
                ``CPUExecutionProvider``. Any other value currently maps to
                ``CUDAExecutionProvider``.
        """
        providers = (
            ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]
        )
        self.session = ort.InferenceSession(model_path, providers=providers)

    def get_signature(self) -> ModelSignature:
        """Builds a name-based input/output signature from the loaded session.

        Returns:
            A ``ModelSignature`` object containing every named input and output
            tensor together with its declared shape and dtype string.

        Raises:
            RuntimeError: If the backend has not been loaded yet.
        """
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
        """Runs one forward pass and returns outputs keyed by tensor name.

        Args:
            inputs: Mapping from ONNX input tensor name to numpy array.

        Returns:
            A mapping from ONNX output tensor name to numpy array.

        Raises:
            RuntimeError: If the backend has not been loaded yet.
        """
        if self.session is None:
            raise RuntimeError("Backend session is not loaded.")
        # Request outputs by declared names so downstream code never depends on
        # positional ordering in the exported model.
        output_names = [meta.name for meta in self.session.get_outputs()]
        output_values = self.session.run(output_names, inputs)
        return dict(zip(output_names, output_values))
