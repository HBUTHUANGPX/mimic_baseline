import numpy as np
import pytest

from awesome_deploy.inference import InputBinding, ModelProtocol, OutputBinding
from awesome_deploy.inference.types import (
    InferenceResult,
    ModelSignature,
    RuntimeState,
    TensorSpec,
)
from awesome_deploy.utils.cfg import G1RobotCfg
from awesome_deploy.utils import infer as infer_module
from awesome_deploy.utils import obscfg as obscfg_module
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
            return RuntimeState(
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


def test_get_obs_cfg_can_select_q1_independently():
    obs_cfg = obscfg_module.get_obs_cfg("q1")

    assert isinstance(obs_cfg, obscfg_module.Q1ObsCfg)
    assert obs_cfg.input_group_map["actor_obs"] == "policy"
    assert obs_cfg.input_group_map["actor_fsq_obs"] == "policy_window"


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


def test_runtime_state_builder_uses_obs_group_mapping_for_multiple_state_inputs():
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
    runner.obs_cfg = obscfg_module.Q1ObsCfg()
    runner.compute_obs_group = lambda group_name: {
        "policy": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "policy_window": np.asarray([4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32),
    }[group_name]

    runtime_state = builder.build(runner)

    assert np.allclose(
        runtime_state.values["actor_obs"],
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    )
    assert np.allclose(
        runtime_state.values["actor_fsq_obs"],
        np.asarray([4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32),
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
            return RuntimeState(
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


def test_init_inference_builds_motion_via_factory(monkeypatch):
    """Inference initialization should delegate motion construction to the factory."""
    fake_motion = type(
        "FakeMotion",
        (),
        {
            "joint_pos": np.zeros((2, 3), dtype=np.float32),
            "joint_vel": np.zeros((2, 3), dtype=np.float32),
            "fps": np.asarray([50.0], dtype=np.float32),
            "time_step_total": 2,
        },
    )()
    captured = {}

    def fake_build_motion_source(cfg, body_indexes, device, body_names=None):
        captured["cfg"] = cfg
        captured["body_indexes"] = body_indexes
        captured["device"] = device
        captured["body_names"] = body_names
        return fake_motion

    protocol = ModelProtocol(
        input_bindings={},
        output_bindings={
            "actions": OutputBinding(
                target_kind="primary",
                target_key="actions",
            )
        },
    )
    signature = ModelSignature(
        inputs={"actor_obs": TensorSpec("actor_obs", (1, 3), "tensor(float)")},
        outputs={"actions": TensorSpec("actions", (1, 2), "tensor(float)")},
    )

    class FakeAdapter:
        def __init__(self, protocol):
            self.protocol = protocol

    class FakeInferenceEngine:
        def __init__(self, backend, io_adapter, model_path, device):
            self.backend = backend
            self.io_adapter = io_adapter
            self.model_path = model_path
            self.device = device
            self.signature = signature
            self.buffers = FakeBuffers()

        def load(self):
            return None

    class FakeBuilder:
        def __init__(self, protocol, signature):
            self.protocol = protocol
            self.signature = signature

        def get_primary_observation_dim(self):
            return 3

    monkeypatch.setattr(infer_module, "build_motion_source", fake_build_motion_source)
    monkeypatch.setattr(infer_module, "ProtocolAdapter", FakeAdapter)
    monkeypatch.setattr(infer_module, "InferenceEngine", FakeInferenceEngine)
    monkeypatch.setattr(infer_module, "RuntimeStateBuilder", FakeBuilder)
    monkeypatch.setattr(infer_module, "OnnxBackend", lambda: "backend")

    runner = infere.__new__(infere)
    runner.motion_body_names_in_isaacsim_index = [4, 2, 0]
    runner._load_model_protocol = lambda: protocol

    runner._init_inference()

    assert runner.motion is fake_motion
    assert captured["cfg"] is infer_module.cfg
    assert captured["body_indexes"] == [4, 2, 0]
    assert captured["device"] == "cpu"
    assert captured["body_names"] == infer_module.cfg.motion_body_names


def test_minimum_infer_advances_realtime_motion_before_building_runtime_state():
    """Realtime motion sources should advance once per policy step before obs build."""
    calls = []

    class FakeMotion:
        def advance(self):
            calls.append("advance")

    class FakeBuilder:
        def build(self, runner):
            calls.append("build")
            return RuntimeState(
                values={
                    "actor_obs": np.asarray([0.1, 0.2], dtype=np.float32),
                    "time_step": 3,
                }
            )

    runner = infere.__new__(infere)
    runner.motion = FakeMotion()
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

    assert calls[:2] == ["advance", "build"]


def test_init_inference_supports_motion_play_with_realtime_source(monkeypatch):
    """Realtime sources should support direct motion-play mode."""
    fake_motion = type(
        "FakeRealtimeMotion",
        (),
        {
            "joint_pos": np.zeros((2, 3), dtype=np.float32),
            "joint_vel": np.zeros((2, 3), dtype=np.float32),
            "fps": np.asarray([0.0], dtype=np.float32),
            "time_step_total": 2,
            "is_realtime": True,
            "joint_order_space": "mujoco",
            "body_order_space": "policy",
        },
    )()

    monkeypatch.setattr(
        infer_module,
        "build_motion_source",
        lambda cfg, body_indexes, device, body_names=None: fake_motion,
    )
    monkeypatch.setattr(infer_module.cfg, "motion_play", True)

    protocol = ModelProtocol(
        input_bindings={},
        output_bindings={
            "actions": OutputBinding(
                target_kind="primary",
                target_key="actions",
            )
        },
    )
    signature = ModelSignature(
        inputs={"actor_obs": TensorSpec("actor_obs", (1, 3), "tensor(float)")},
        outputs={"actions": TensorSpec("actions", (1, 2), "tensor(float)")},
    )

    class FakeAdapter:
        def __init__(self, protocol):
            self.protocol = protocol

    class FakeInferenceEngine:
        def __init__(self, backend, io_adapter, model_path, device):
            self.signature = signature
            self.buffers = FakeBuffers()

        def load(self):
            return None

    class FakeBuilder:
        def __init__(self, protocol, signature):
            self.protocol = protocol
            self.signature = signature

        def get_primary_observation_dim(self):
            return 3

    monkeypatch.setattr(infer_module, "ProtocolAdapter", FakeAdapter)
    monkeypatch.setattr(infer_module, "InferenceEngine", FakeInferenceEngine)
    monkeypatch.setattr(infer_module, "RuntimeStateBuilder", FakeBuilder)
    monkeypatch.setattr(infer_module, "OnnxBackend", lambda: "backend")

    runner = infere.__new__(infere)
    runner.motion_body_names_in_isaacsim_index = [0]
    runner._load_model_protocol = lambda: protocol

    runner._init_inference()

    assert runner.motion is fake_motion
    assert runner.policy_dt == infer_module.cfg.policy_dt

    monkeypatch.setattr(infer_module.cfg, "motion_play", False)


def test_motion_joint_pos_for_policy_order_passthrough_for_isaac_motion():
    runner = infere.__new__(infere)
    runner.motion = type(
        "OfflineMotion",
        (),
        {
            "joint_pos": np.asarray([[10.0, 20.0, 30.0]], dtype=np.float32),
            "joint_order_space": "isaac",
        },
    )()
    runner.mujoco2isaac_sim_index = [2, 0, 1]

    joint_pos = runner._motion_joint_pos_for_policy(0)

    assert np.allclose(joint_pos, np.asarray([10.0, 20.0, 30.0], dtype=np.float32))


def test_motion_joint_pos_for_policy_order_reorders_mujoco_motion():
    runner = infere.__new__(infere)
    runner.motion = type(
        "RealtimeMotion",
        (),
        {
            "joint_pos": np.asarray([[100.0, 200.0, 300.0]], dtype=np.float32),
            "joint_order_space": "mujoco",
        },
    )()
    runner.mujoco2isaac_sim_index = [2, 0, 1]

    joint_pos = runner._motion_joint_pos_for_policy(0)

    assert np.allclose(joint_pos, np.asarray([300.0, 100.0, 200.0], dtype=np.float32))


def test_motion_joint_pos_for_mujoco_order_reorders_isaac_motion():
    runner = infere.__new__(infere)
    runner.motion = type(
        "OfflineMotion",
        (),
        {
            "joint_pos": np.asarray([[10.0, 20.0, 30.0]], dtype=np.float32),
            "joint_order_space": "isaac",
        },
    )()
    runner.isaac_sim2mujoco_index = [1, 2, 0]

    joint_pos = runner._motion_joint_pos_for_mujoco(0)

    assert np.allclose(joint_pos, np.asarray([20.0, 30.0, 10.0], dtype=np.float32))


def test_motion_body_quat_for_policy_passthrough():
    runner = infere.__new__(infere)
    runner.motion = type(
        "Motion",
        (),
        {
            "body_quat_w": np.asarray(
                [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
                dtype=np.float32,
            ),
            "body_order_space": "policy",
        },
    )()

    quat = runner._motion_body_quat_for_policy(0, 1)

    assert np.allclose(quat, np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32))


def test_motion_body_pos_for_policy_window_passthrough():
    runner = infere.__new__(infere)
    runner.motion = type(
        "Motion",
        (),
        {
            "body_pos_w": np.asarray(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                ],
                dtype=np.float32,
            ),
            "body_order_space": "policy",
        },
    )()

    body_pos = runner._motion_body_pos_for_policy(np.asarray([0, 1], dtype=np.int64))

    assert np.allclose(
        body_pos,
        np.asarray(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
            dtype=np.float32,
        ),
    )
