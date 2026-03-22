import numpy as np

from awesome_deploy.utils.infer import infere
from awesome_deploy.inference.types import InferenceResult


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
    runner = infere.__new__(infere)
    runner.inference_engine = FakeEngine()
    runner.update_obs = lambda: np.asarray([0.1, 0.2], dtype=np.float32)
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
    context = runner.inference_engine.calls[0]
    assert context.time_step == 3
    assert np.allclose(context.obs, np.asarray([0.1, 0.2], dtype=np.float32))
    assert np.allclose(runner.action, np.asarray([0.5, -0.25], dtype=np.float32))
    assert np.allclose(
        runner.target_dof_pos, np.asarray([0.625, -0.3125], dtype=np.float32)
    )
