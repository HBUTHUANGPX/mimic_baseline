"""Tests for window observations exposed by the MuJoCo deployment simulator."""

from types import SimpleNamespace

import numpy as np

from awesome_deploy.scripts.deploy_g1_mujoco import simulator


class _FakeBuffers:
    """Minimal buffer stub that only serves ``time_step`` lookups."""

    def __init__(self, time_step):
        """Stores the single logical rollout step used by tests."""
        self._time_step = time_step

    def get(self, name, default=None):
        """Returns the stored time step for the expected buffer key."""
        if name == "time_step":
            return self._time_step
        return default


class _FakeEngine:
    """Minimal inference-engine stub exposing only persistent buffers."""

    def __init__(self, time_step):
        """Initializes the fake engine with a fake time-step buffer."""
        self.buffers = _FakeBuffers(time_step)


def test_get_motion_window_indices_clips_to_available_frames():
    """Window indices should stay inside the motion timeline."""
    sim = simulator.__new__(simulator)
    sim.inference_engine = _FakeEngine(0)
    sim.command_window_offsets = np.asarray([-2, -1, 0, 1, 2], dtype=np.int64)
    sim.motion = SimpleNamespace(time_step_total=4)

    window_indices = sim._get_motion_window_indices()

    assert np.array_equal(
        window_indices,
        np.asarray([0, 0, 0, 1, 2], dtype=np.int64),
    )


def test_obs_joint_pos_delta_window_returns_flattened_window():
    """Joint-position delta window should flatten time and joint dimensions."""
    sim = simulator.__new__(simulator)
    sim.inference_engine = _FakeEngine(1)
    sim.command_window_offsets = np.asarray([0, 1], dtype=np.int64)
    sim.motion = SimpleNamespace(
        time_step_total=3,
        joint_pos=np.asarray(
            [
                [0.0, 0.0],
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=np.float32,
        ),
    )
    sim._obs_joint_pos = lambda: np.asarray([0.5, 1.5], dtype=np.float32)

    obs = sim._obs_joint_pos_delta_window()

    assert np.allclose(obs, np.asarray([0.5, 0.5, 2.5, 2.5], dtype=np.float32))


def test_obs_motion_ref_ori_b_window_returns_flattened_6d_rotations():
    """Reference orientation window should return flattened 6D rotations."""
    sim = simulator.__new__(simulator)
    sim.inference_engine = _FakeEngine(0)
    sim.command_window_offsets = np.asarray([0, 1], dtype=np.int64)
    sim.motion = SimpleNamespace(
        time_step_total=2,
        body_quat_w=np.asarray(
            [
                [[1.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        ),
    )
    sim.pin = SimpleNamespace(
        mujoco_to_pinocchio=lambda *args, **kwargs: None,
        get_link_quaternion=lambda link_name: np.asarray(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        ),
    )
    sim.d = SimpleNamespace(qpos=np.zeros(9, dtype=np.float32))
    sim.robot_ref_quat_w = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    sim.motion_reference_body_index = 0

    obs = sim._obs_motion_ref_ori_b_window()

    expected = np.asarray(
        [
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    assert np.allclose(obs, expected)
