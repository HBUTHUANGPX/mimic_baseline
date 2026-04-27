from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "hdf5_parse" / "scripts" / "full_body_mocap_mujoco_viewer.py"
HDF5_PATH = next(
    iter(sorted((REPO_ROOT / "hdf5_parse" / "test_data").glob("*/*/annotation.hdf5"))),
    REPO_ROOT / "hdf5_parse" / "hdf5" / "annotation.hdf5",
)


def load_module():
    spec = importlib.util.spec_from_file_location("full_body_mocap_mujoco_viewer", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_split_pose7_interprets_annotation_layout_as_qwxyz_plus_xyz():
    module = load_module()

    pose7 = np.array([1.0, 0.0, 0.0, 0.0, 0.1, -0.2, 0.3], dtype=np.float32)
    quat_wxyz, translation_xyz = module.split_pose7_qwxyz_xyz(pose7)

    np.testing.assert_allclose(quat_wxyz, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(translation_xyz, [0.1, -0.2, 0.3])


def test_build_body_bone_segments_uses_official_parent_indices():
    module = load_module()

    joints = np.arange(52 * 3, dtype=np.float32).reshape(52, 3)
    segments = module.build_body_bone_segments(joints)

    assert segments.shape == (51, 2, 3)
    np.testing.assert_allclose(segments[0, 0], joints[0])
    np.testing.assert_allclose(segments[0, 1], joints[1])


def test_load_motion_clip_reads_expected_shapes_from_annotation_file():
    module = load_module()

    clip = module.load_motion_clip(HDF5_PATH, start_frame=0, end_frame=4, stride=1)

    assert clip.keypoints.shape == (4, 52, 3)
    assert clip.left_hand_joints.shape == (4, 16, 3)
    assert clip.right_hand_joints.shape == (4, 16, 3)
    assert clip.root_quat_wxyz.shape == (4, 4)
    assert clip.root_translation.shape == (4, 3)
    assert clip.body_quats.shape == (4, 21, 4)
    assert clip.fps == 20.0
    np.testing.assert_allclose(clip.root_translation[0], clip.keypoints[0, 0], atol=1e-6)


def test_summarize_caption_prefers_main_task_from_json_payload():
    module = load_module()

    caption = '{"config": {"Main Task": "Sorting colorful star-shaped paper origami"}, "segments": []}'

    assert module.summarize_caption(caption) == "Sorting colorful star-shaped paper origami"


def test_extract_visual_hand_keypoints_uses_full_body_hand_branches():
    module = load_module()

    keypoints = np.arange(52 * 3, dtype=np.float32).reshape(52, 3)
    left_hand, right_hand = module.extract_visual_hand_keypoints(keypoints)

    np.testing.assert_allclose(left_hand[0], keypoints[20])
    np.testing.assert_allclose(left_hand[1:], keypoints[22:37])
    np.testing.assert_allclose(right_hand[0], keypoints[21])
    np.testing.assert_allclose(right_hand[1:], keypoints[37:52])


def test_extract_body_visual_keypoints_excludes_hand_branches():
    module = load_module()

    keypoints = np.arange(52 * 3, dtype=np.float32).reshape(52, 3)
    body_only = module.extract_body_visual_keypoints(keypoints)

    assert body_only.shape == (20, 3)
    np.testing.assert_allclose(body_only, keypoints[:20])


def test_load_motion_clip_drops_invalid_frames_by_default():
    module = load_module()

    clip = module.load_motion_clip(HDF5_PATH, start_frame=296, end_frame=299, stride=1)

    assert clip.keypoints.shape == (1, 52, 3)
    np.testing.assert_array_equal(clip.frame_nums, [296])


def test_parse_args_uses_streamlined_defaults():
    module = load_module()

    args = module.parse_args([])

    assert args.hdf5_path == HDF5_PATH
    assert args.start == 0
    assert args.end == -1
    assert args.stride == 1
    assert args.loop is False
    assert args.hands is True
    assert args.root_frame is False
    assert args.slam_points == 0


def test_parse_args_removes_legacy_tuning_flags():
    module = load_module()

    with pytest.raises(SystemExit):
        module.parse_args(["--body-joint-radius", "0.03"])

    with pytest.raises(SystemExit):
        module.parse_args(["--show-cpf-frame"])


def test_build_body_visual_bone_segments_excludes_hand_finger_chains():
    module = load_module()

    joints = np.arange(52 * 3, dtype=np.float32).reshape(52, 3)
    segments = module.build_body_visual_bone_segments(joints)

    assert segments.shape[0] < module.build_body_bone_segments(joints).shape[0]
    flattened = segments.reshape(-1, 3)
    for hand_joint in joints[22:]:
        assert not np.any(np.all(np.isclose(flattened, hand_joint), axis=1))
