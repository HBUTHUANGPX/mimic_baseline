import numpy as np

from awesome_deploy.inference.engine import InferenceEngine
from awesome_deploy.inference.io_adapters.default_mimo import DefaultMimoAdapter
from awesome_deploy.inference.types import InferenceContext, ModelSignature, TensorSpec


class FakeBackend:
    def __init__(self):
        self.loaded = None
        self.last_inputs = None

    def load(self, model_path: str, device: str = "cpu") -> None:
        self.loaded = (model_path, device)

    def get_signature(self) -> ModelSignature:
        return ModelSignature(
            inputs={
                "obs": TensorSpec(name="obs", shape=(1, 3), dtype="tensor(float)"),
                "time_step": TensorSpec(
                    name="time_step", shape=(1, 1), dtype="tensor(float)"
                ),
            },
            outputs={
                "actions": TensorSpec(
                    name="actions", shape=(1, 2), dtype="tensor(float)"
                ),
                "joint_pos": TensorSpec(
                    name="joint_pos", shape=(1, 2), dtype="tensor(float)"
                ),
            },
        )

    def infer(self, inputs):
        self.last_inputs = inputs
        return {
            "actions": np.asarray([[1.0, -1.0]], dtype=np.float32),
            "joint_pos": np.asarray([[0.2, 0.4]], dtype=np.float32),
        }


def test_inference_engine_updates_buffers_and_parses_mimo_outputs():
    backend = FakeBackend()
    engine = InferenceEngine(
        backend=backend,
        io_adapter=DefaultMimoAdapter(),
        model_path="fake.onnx",
        device="cpu",
    )

    engine.load()
    result = engine.step(
        InferenceContext(
            obs=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            time_step=1,
        )
    )

    assert backend.loaded == ("fake.onnx", "cpu")
    assert backend.last_inputs["obs"].shape == (1, 3)
    assert backend.last_inputs["time_step"].shape == (1, 1)
    assert np.allclose(result.primary_action, np.asarray([1.0, -1.0], dtype=np.float32))
    assert "joint_pos" in result.outputs
    assert engine.buffers.get("time_step") == 2
    assert np.allclose(
        engine.buffers.get("action"), np.asarray([1.0, -1.0], dtype=np.float32)
    )
    assert np.allclose(engine.buffers.get("prev_action"), np.zeros(2, dtype=np.float32))
