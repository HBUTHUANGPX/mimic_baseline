import numpy as np
import pytest

from awesome_deploy.inference import InputBinding, ModelProtocol
from awesome_deploy.inference.types import InferenceResult, ModelSignature, TensorSpec
from awesome_deploy.utils.cfg import G1RobotCfg
from awesome_deploy.utils import infer as infer_module
from awesome_deploy.utils.infer import infere


class FakeBuffers:
    def __init__(self):
        self.values = {"time_step": 3}

    def get(self, name, default=None):
        return self.values.get(name, default)


class FakeEngine:
    def __init__(self):
        self.buffers = FakeBuffers()
        self.calls = []

    def step(self, context):
        self.calls.append(context)
        return InferenceResult(
            outputs={"actions": np.asarray([[0.5, -0.25]], dtype=np.float32)},
            primary_action=np.asarray([0.5, -0.25], dtype=np.float32),
            state_updates={},
        )


def test_minimum_infer_uses_inference_engine_and_updates_target_action():
    class FakeBuilder:
        def build(self, runner):
            return infer_module.RuntimeState(
                values={
                    "policy_obs": np.asarray([0.1, 0.2], dtype=np.float32),
                    "time_step": 3,
                }
            )

    runner = infere.__new__(infere)
    runner.runtime_state_builder = FakeBuilder()
    runner.inference_engine = FakeEngine()
    runner.cmd = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    runner.tq_max = np.asarray([10.0, 10.0], dtype=np.float32)
    runner.P_gains = np.asarray([2.0, 2.0], dtype=np.float32)
    runner.default_pos = np.asarray([0.0, 0.0], dtype=np.float32)
    runner.isaac_sim2mujoco_index = [0, 1]
    runner.action_clip = 10.0
    runner.action_scale = 0.25
    runner.latest_inference = None
    runner.action = np.zeros(2, dtype=np.float32)

    runner.minimum_infer()

    assert len(runner.inference_engine.calls) == 1
    runtime_state = runner.inference_engine.calls[0]
    assert runtime_state.values["time_step"] == 3
    assert np.allclose(
        runtime_state.values["policy_obs"], np.asarray([0.1, 0.2], dtype=np.float32)
    )
    assert np.allclose(runner.action, np.asarray([0.5, -0.25], dtype=np.float32))
    assert np.allclose(
        runner.target_dof_pos, np.asarray([0.625, -0.3125], dtype=np.float32)
    )


def test_robot_cfg_exposes_default_protocol_path():
    robot_cfg = G1RobotCfg()

    assert robot_cfg.protocol_path.endswith("/policy.protocol.yaml")
    assert robot_cfg.protocol_path.startswith(robot_cfg.policy_dir)


def test_load_model_protocol_uses_cfg_protocol_path(monkeypatch):
    captured = {}
    expected_protocol = ModelProtocol(input_bindings={}, output_bindings={})

    def fake_loader(protocol_path, transform_registry):
        captured["protocol_path"] = protocol_path
        captured["transform_registry"] = transform_registry
        return expected_protocol

    monkeypatch.setattr(infer_module, "load_protocol_from_file", fake_loader)

    runner = infere.__new__(infere)
    protocol = runner._load_model_protocol()

    assert protocol is expected_protocol
    assert captured["protocol_path"] == infer_module.cfg.protocol_path
    assert captured["transform_registry"] is infer_module.TRANSFORM_REGISTRY


def test_runtime_state_builder_primary_observation_dim_uses_first_state_input():
    protocol = ModelProtocol(
        input_bindings={
            "actor_obs": InputBinding(source_kind="state", source_key="actor_obs"),
            "time_step": InputBinding(source_kind="buffer", source_key="time_step"),
            "actor_fsq_obs": InputBinding(
                source_kind="state",
                source_key="actor_fsq_obs",
            ),
        },
        output_bindings={},
    )
    signature = ModelSignature(
        inputs={
            "actor_obs": TensorSpec("actor_obs", (1, 581), "tensor(float)"),
            "time_step": TensorSpec("time_step", (1, 1), "tensor(float)"),
            "actor_fsq_obs": TensorSpec("actor_fsq_obs", (1, 704), "tensor(float)"),
        },
        outputs={},
    )

    builder = infer_module.RuntimeStateBuilder(protocol=protocol, signature=signature)

    assert builder.get_primary_observation_dim() == 581


def test_runtime_state_builder_reuses_policy_obs_for_multiple_state_inputs():
    protocol = ModelProtocol(
        input_bindings={
            "actor_obs": InputBinding(source_kind="state", source_key="actor_obs"),
            "actor_fsq_obs": InputBinding(
                source_kind="state",
                source_key="actor_fsq_obs",
            ),
            "time_step": InputBinding(source_kind="buffer", source_key="time_step"),
        },
        output_bindings={},
    )
    signature = ModelSignature(
        inputs={
            "actor_obs": TensorSpec("actor_obs", (1, 3), "tensor(float)"),
            "actor_fsq_obs": TensorSpec("actor_fsq_obs", (1, 5), "tensor(float)"),
            "time_step": TensorSpec("time_step", (1, 1), "tensor(float)"),
        },
        outputs={},
    )
    builder = infer_module.RuntimeStateBuilder(protocol=protocol, signature=signature)
    runner = infere.__new__(infere)
    runner.inference_engine = FakeEngine()
    runner.inference_engine.buffers.values["time_step"] = 7
    runner.cmd = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    runner.motion = "motion"
    runner.update_obs = lambda: np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    runtime_state = builder.build(runner)

    assert np.allclose(runtime_state.values["actor_obs"], np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.allclose(
        runtime_state.values["actor_fsq_obs"],
        np.asarray([1.0, 2.0, 3.0, 4.0, 0.0], dtype=np.float32),
    )
    assert runtime_state.values["time_step"] == 7
    assert np.allclose(runtime_state.values["command"], runner.cmd)
    assert runtime_state.values["motion"] == "motion"


def test_minimum_infer_uses_runtime_state_builder(monkeypatch):
    class FakeBuilder:
        def __init__(self):
            self.calls = []

        def build(self, runner):
            self.calls.append(runner)
            return infer_module.RuntimeState(
                values={
                    "actor_obs": np.asarray([0.1, 0.2], dtype=np.float32),
                    "time_step": 3,
                }
            )

    runner = infere.__new__(infere)
    runner.runtime_state_builder = FakeBuilder()
    runner.inference_engine = FakeEngine()
    runner.cmd = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    runner.tq_max = np.asarray([10.0, 10.0], dtype=np.float32)
    runner.P_gains = np.asarray([2.0, 2.0], dtype=np.float32)
    runner.default_pos = np.asarray([0.0, 0.0], dtype=np.float32)
    runner.isaac_sim2mujoco_index = [0, 1]
    runner.action_clip = 10.0
    runner.action_scale = 0.25
    runner.latest_inference = None
    runner.action = np.zeros(2, dtype=np.float32)

    runner.minimum_infer()

    assert runner.runtime_state_builder.calls == [runner]
    runtime_state = runner.inference_engine.calls[0]
    assert "policy_obs" not in runtime_state.values
    assert np.allclose(
        runtime_state.values["actor_obs"], np.asarray([0.1, 0.2], dtype=np.float32)
    )
