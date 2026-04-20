from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "hdf5_parse" / "smpl_motion_tools.py"
HDF5_PATH = REPO_ROOT / "hdf5_parse" / "hdf5" / "annotation.hdf5"


def load_module():
    spec = importlib.util.spec_from_file_location("smpl_motion_tools", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_quat_wxyz_to_rotvec_handles_identity_and_half_turn():
    module = load_module()

    quats = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    rotvec = module.quat_wxyz_to_rotvec(quats)

    np.testing.assert_allclose(rotvec[0], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(rotvec[1], [np.pi, 0.0, 0.0], atol=1e-6)


def test_build_smplh_motion_clip_outputs_expected_shapes():
    module = load_module()

    root_pose7 = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.1, -0.2, 0.3],
            [1.0, 0.0, 0.0, 0.0, 0.4, -0.5, 0.6],
        ],
        dtype=np.float32,
    )
    body_quats = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 21, 1))
    left_hand_quats = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 15, 1))
    right_hand_quats = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 15, 1))
    betas = np.zeros((2, 16), dtype=np.float32)
    frame_nums = np.array([10, 11], dtype=np.int32)

    clip = module.build_smplh_motion_clip(
        root_pose7=root_pose7,
        body_quats=body_quats,
        left_hand_quats=left_hand_quats,
        right_hand_quats=right_hand_quats,
        betas=betas,
        frame_nums=frame_nums,
        fps=20.0,
    )

    assert clip.model_type == "smplh"
    assert clip.global_orient.shape == (2, 3)
    assert clip.body_pose.shape == (2, 63)
    assert clip.left_hand_pose.shape == (2, 45)
    assert clip.right_hand_pose.shape == (2, 45)
    assert clip.transl.shape == (2, 3)
    assert clip.betas.shape == (2, 16)
    np.testing.assert_allclose(clip.transl[:, :], root_pose7[:, 4:])
    np.testing.assert_array_equal(clip.frame_nums, frame_nums)


def test_convert_smplh_motion_clip_to_smpl_pads_two_missing_hand_body_joints():
    module = load_module()

    smplh_clip = module.SMPLMotionClip(
        model_type="smplh",
        global_orient=np.zeros((3, 3), dtype=np.float32),
        body_pose=np.arange(3 * 63, dtype=np.float32).reshape(3, 63),
        transl=np.ones((3, 3), dtype=np.float32),
        betas=np.zeros((3, 16), dtype=np.float32),
        fps=20.0,
        frame_nums=np.array([1, 2, 3], dtype=np.int32),
        left_hand_pose=np.full((3, 45), 7.0, dtype=np.float32),
        right_hand_pose=np.full((3, 45), 9.0, dtype=np.float32),
    )

    smpl_clip = module.convert_smplh_motion_clip_to_smpl(smplh_clip)

    assert smpl_clip.model_type == "smpl"
    assert smpl_clip.body_pose.shape == (3, 69)
    np.testing.assert_allclose(smpl_clip.body_pose[:, :63], smplh_clip.body_pose)
    np.testing.assert_allclose(smpl_clip.body_pose[:, 63:], 0.0)
    assert smpl_clip.left_hand_pose is None
    assert smpl_clip.right_hand_pose is None


def test_resolve_body_model_path_prefers_explicit_path(tmp_path: Path):
    module = load_module()

    fake_model = tmp_path / "SMPLH_NEUTRAL.pkl"
    fake_model.write_bytes(b"model")

    resolved = module.resolve_body_model_path("smplh", explicit_path=fake_model)

    assert resolved == fake_model


def test_resolve_body_model_path_prefers_npz_for_default_smpl_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = load_module()

    pkl_model = tmp_path / "SMPL_NEUTRAL.pkl"
    npz_model = tmp_path / "SMPL_NEUTRAL.npz"
    pkl_model.write_bytes(b"pkl")
    npz_model.write_bytes(b"npz")

    monkeypatch.setattr(module, "DEFAULT_SMPL_MODEL_CANDIDATES", (pkl_model, npz_model))

    resolved = module.resolve_body_model_path("smpl")

    assert resolved == npz_model


def test_resolve_body_model_path_raises_when_no_candidate_exists(monkeypatch: pytest.MonkeyPatch):
    module = load_module()

    monkeypatch.setattr(module, "DEFAULT_SMPLH_MODEL_CANDIDATES", ())
    monkeypatch.setattr(module, "DEFAULT_SMPL_MODEL_CANDIDATES", ())

    with pytest.raises(FileNotFoundError):
        module.resolve_body_model_path("smplh")


def test_load_smplh_motion_clip_from_annotation_file():
    module = load_module()

    clip = module.load_smplh_motion_clip(HDF5_PATH, start_frame=0, end_frame=4, stride=1)

    assert clip.model_type == "smplh"
    assert clip.global_orient.shape == (4, 3)
    assert clip.body_pose.shape == (4, 63)
    assert clip.left_hand_pose.shape == (4, 45)
    assert clip.right_hand_pose.shape == (4, 45)
    assert clip.transl.shape == (4, 3)
    assert clip.betas.shape == (4, 16)
    assert clip.fps == 20.0


def test_load_smplh_motion_clip_drops_invalid_frames_by_default():
    module = load_module()

    clip = module.load_smplh_motion_clip(HDF5_PATH, start_frame=296, end_frame=299, stride=1)

    assert clip.body_pose.shape[0] == 1
    np.testing.assert_array_equal(clip.frame_nums, [296])
