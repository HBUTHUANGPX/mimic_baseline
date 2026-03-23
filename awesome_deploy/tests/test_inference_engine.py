import numpy as np

from awesome_deploy.inference.engine import InferenceEngine
from awesome_deploy.inference.io_adapters.protocol_adapter import ProtocolAdapter
from awesome_deploy.inference.protocol import (
    BufferInitializer,
    InputBinding,
    ModelProtocol,
    OutputBinding,
)
from awesome_deploy.inference.types import ModelSignature, RuntimeState, TensorSpec


def _batch_feature(value):
    return np.asarray(value, dtype=np.float32).reshape(1, -1)


def _batch_scalar(value):
    return np.asarray([[value]], dtype=np.float32)


def _flat(value):
    return np.asarray(value, dtype=np.float32).reshape(-1)


class FakeBackend:
    def __init__(self):
        self.loaded = None
        self.last_inputs = None

    def load(self, model_path: str, device: str = "cpu") -> None:
        self.loaded = (model_path, device)

    def get_signature(self) -> ModelSignature:
        return ModelSignature(
            inputs={
                "obs_tensor": TensorSpec(
                    name="obs_tensor", shape=(1, 3), dtype="tensor(float)"
                ),
                "step_tensor": TensorSpec(
                    name="step_tensor", shape=(1, 1), dtype="tensor(float)"
                ),
            },
            outputs={
                "policy_head": TensorSpec(
                    name="policy_head", shape=(1, 2), dtype="tensor(float)"
                ),
                "joint_pos_head": TensorSpec(
                    name="joint_pos_head", shape=(1, 2), dtype="tensor(float)"
                ),
            },
        )

    def infer(self, inputs):
        self.last_inputs = inputs
        return {
            "policy_head": np.asarray([[1.0, -1.0]], dtype=np.float32),
            "joint_pos_head": np.asarray([[0.2, 0.4]], dtype=np.float32),
        }


def test_inference_engine_uses_protocol_bindings_to_build_inputs_and_outputs():
    protocol = ModelProtocol(
        input_bindings={
            "obs_tensor": InputBinding(
                source_kind="state",
                source_key="policy_obs",
                transform=_batch_feature,
            ),
            "step_tensor": InputBinding(
                source_kind="buffer",
                source_key="time_step",
                transform=_batch_scalar,
            ),
        },
        output_bindings={
            "policy_head": OutputBinding(
                target_kind="primary",
                target_key="policy_action",
                transform=_flat,
            ),
            "joint_pos_head": OutputBinding(
                target_kind="output",
                target_key="joint_pos",
            ),
        },
        buffer_initializers={
            "time_step": BufferInitializer(init_kind="constant", value=1),
            "action": BufferInitializer(
                init_kind="zeros_from_output",
                tensor_name="policy_head",
                axis=1,
            ),
            "prev_action": BufferInitializer(
                init_kind="zeros_from_output",
                tensor_name="policy_head",
                axis=1,
            ),
        },
        per_step_buffer_updates={
            "time_step": InputBinding(
                source_kind="buffer",
                source_key="time_step",
                transform=lambda value: int(value) + 1,
            ),
            "prev_action": InputBinding(
                source_kind="buffer",
                source_key="action",
            ),
            "action": InputBinding(
                source_kind="result",
                source_key="policy_action",
            ),
        },
    )

    backend = FakeBackend()
    engine = InferenceEngine(
        backend=backend,
        io_adapter=ProtocolAdapter(protocol),
        model_path="fake.onnx",
        device="cpu",
    )

    engine.load()
    result = engine.step(
        RuntimeState(values={"policy_obs": np.asarray([0.1, 0.2, 0.3], dtype=np.float32)})
    )

    assert backend.loaded == ("fake.onnx", "cpu")
    assert backend.last_inputs["obs_tensor"].shape == (1, 3)
    assert backend.last_inputs["step_tensor"].shape == (1, 1)
    assert np.allclose(result.primary_action, np.asarray([1.0, -1.0], dtype=np.float32))
    assert "joint_pos" in result.outputs
    assert engine.buffers.get("time_step") == 2
    assert np.allclose(
        engine.buffers.get("action"), np.asarray([1.0, -1.0], dtype=np.float32)
    )
    assert np.allclose(engine.buffers.get("prev_action"), np.zeros(2, dtype=np.float32))
