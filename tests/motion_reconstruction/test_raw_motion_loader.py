from pathlib import Path

import numpy as np
import pytest
import torch

from motion_reconstruction.data.raw_motion import RawMotionLoader


def _write_npz(path: Path, *, joint_names=("hip", "knee"), scalar_first=False):
    frames = 4
    robot_bodies = np.array(["torso_link", "left_foot"], dtype=object)
    human_names = np.array(["Hips", "Spine1", "Head"], dtype=object)
    quat_xyzw = np.zeros((frames, 2, 4), dtype=np.float32)
    quat_xyzw[..., 3] = 1.0
    human_quat_xyzw = np.zeros((frames, 3, 4), dtype=np.float32)
    human_quat_xyzw[..., 3] = 1.0

    if scalar_first:
        body_quat = quat_xyzw[..., [3, 0, 1, 2]]
        human_quat = human_quat_xyzw[..., [3, 0, 1, 2]]
    else:
        body_quat = quat_xyzw
        human_quat = human_quat_xyzw

    np.savez(
        path,
        fps=np.array(120),
        scalar_first=np.array(scalar_first),
        robot_joint_names=np.asarray(joint_names, dtype=object),
        robot_body_names=robot_bodies,
        human_joint_names=human_names,
        robot_joint_pos=np.arange(frames * len(joint_names), dtype=np.float32).reshape(
            frames, len(joint_names)
        ),
        robot_joint_vel=np.ones((frames, len(joint_names)), dtype=np.float32),
        robot_body_pos=np.zeros((frames, 2, 3), dtype=np.float32),
        robot_body_quat=body_quat,
        robot_body_lin_vel=np.zeros((frames, 2, 3), dtype=np.float32),
        robot_body_ang_vel=np.zeros((frames, 2, 3), dtype=np.float32),
        human_global_pos=np.zeros((frames, 3, 3), dtype=np.float32),
        human_global_quat=human_quat,
    )


def test_raw_loader_concatenates_alias_fields_and_normalizes_quat_order(tmp_path: Path):
    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    _write_npz(first, scalar_first=False)
    _write_npz(second, scalar_first=True)

    dataset = RawMotionLoader([first, second], groups=["g0", "g1"]).load(device="cpu")

    assert dataset.fps == 120
    assert dataset.joint_pos.shape == (8, 2)
    assert dataset.body_quat_w.shape == (8, 2, 4)
    assert torch.allclose(dataset.body_quat_w[..., 0], torch.ones(8, 2))
    assert torch.allclose(dataset.body_quat_w[..., 1:], torch.zeros(8, 2, 3))
    assert dataset.motion_lengths.tolist() == [4, 4]
    assert dataset.motion_groups == ["g0", "g1"]
    assert dataset.robot_joint_names == ["hip", "knee"]
    assert dataset.human_body_names == ["Hips", "Spine1", "Head"]


def test_raw_loader_rejects_inconsistent_schema(tmp_path: Path):
    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    _write_npz(first, joint_names=("hip", "knee"))
    _write_npz(second, joint_names=("knee", "hip"))

    with pytest.raises(ValueError, match="robot_joint_names"):
        RawMotionLoader([first, second]).load(device="cpu")
