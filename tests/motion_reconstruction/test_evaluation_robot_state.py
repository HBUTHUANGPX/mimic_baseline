import numpy as np

from motion_reconstruction.evaluation.robot_state import (
    human_skeleton_edges,
    robot_feature_to_qpos,
    rot6d_to_quat_wxyz_numpy,
)


def test_rot6d_to_quat_wxyz_numpy_converts_identity_rotation():
    quat = rot6d_to_quat_wxyz_numpy(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32))

    assert np.allclose(quat, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), atol=1e-5)


def test_robot_feature_to_qpos_supports_free_and_fixed_base_models():
    feature = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.25, -0.5], dtype=np.float32)
    anchor_pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    free_qpos = robot_feature_to_qpos(feature, anchor_pos_w=anchor_pos, expected_nq=9)
    fixed_qpos = robot_feature_to_qpos(feature, anchor_pos_w=anchor_pos, expected_nq=2)

    assert np.allclose(free_qpos[:3], anchor_pos)
    assert np.allclose(free_qpos[3:7], np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), atol=1e-5)
    assert np.allclose(free_qpos[7:], np.array([0.25, -0.5], dtype=np.float32))
    assert np.allclose(fixed_qpos, np.array([0.25, -0.5], dtype=np.float32))


def test_human_skeleton_edges_only_uses_available_names():
    names = ["Hips", "Spine1", "Spine2", "LeftLeg", "LeftShin"]

    edges = human_skeleton_edges(names)

    assert edges == [(0, 1), (1, 2), (0, 3), (3, 4)]
