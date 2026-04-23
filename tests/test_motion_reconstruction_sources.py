from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_reconstruction.config.schema import MotionReconstructionConfig


def _identity_quat_array(num_frames: int, num_joints: int) -> np.ndarray:
    quat = np.zeros((num_frames, num_joints, 4), dtype=np.float32)
    quat[..., 0] = 1.0
    return quat


def _make_hdf5_human_npz(path: Path) -> Path:
    human_joint_names = np.asarray(["Hips", "Spine1", "Head"], dtype=object)
    human_global_pos = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
            [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 2.0]],
            [[2.0, 0.0, 0.0], [2.0, 0.0, 1.0], [2.0, 0.0, 2.0]],
            [[3.0, 0.0, 0.0], [3.0, 0.0, 1.0], [3.0, 0.0, 2.0]],
        ],
        dtype=np.float32,
    )
    np.savez(
        path,
        fps=np.array(20, dtype=np.int32),
        scalar_first=np.array(True),
        human_joint_names=human_joint_names,
        human_global_pos=human_global_pos,
        human_global_quat=_identity_quat_array(num_frames=4, num_joints=3),
        timeline_frame_indices=np.array([10, 11, 12, 13], dtype=np.int32),
    )
    return path


def test_build_hdf5_human_source_uses_anchor_positions_and_windows(tmp_path: Path) -> None:
    from motion_reconstruction.inference.sources import build_hdf5_human_source

    motion_path = _make_hdf5_human_npz(tmp_path / "human_only.npz")
    config = MotionReconstructionConfig()
    config.train.history = 1
    config.train.future = 1
    config.features.human_anchor_body = "Hips"
    config.features.human_body_names = ["Spine1", "Head"]
    feature_schema = {
        "robot_joint_names": ["left_hip", "right_hip"],
        "robot_body_names": ["base", "torso_link"],
    }

    bundle = build_hdf5_human_source(
        motion_npz=motion_path,
        config=config,
        feature_schema=feature_schema,
        device="cpu",
    )

    np.testing.assert_array_equal(bundle.center_indices.cpu().numpy(), np.array([1, 2], dtype=np.int64))
    np.testing.assert_array_equal(bundle.window_offsets.cpu().numpy(), np.array([-1, 0, 1], dtype=np.int64))
    assert bundle.robot_features is None
    assert bundle.human_features.shape == (4, 12)
    np.testing.assert_allclose(
        bundle.robot_anchor_pos_w.cpu().numpy(),
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert bundle.robot_joint_names == ["left_hip", "right_hip"]
    assert bundle.robot_body_names == ["base", "torso_link"]
    assert bundle.human_body_names == ["Hips", "Spine1", "Head"]
    assert bundle.display_human_body_names == ["Hips", "Spine1", "Head"]


def test_build_hdf5_human_source_requires_motion_npz(tmp_path: Path) -> None:
    from motion_reconstruction.inference.sources import build_inference_source

    config = MotionReconstructionConfig()
    with pytest.raises(ValueError, match="motion_npz"):
        build_inference_source(
            source="hdf5-human",
            config=config,
            device="cpu",
            feature_schema={"robot_joint_names": [], "robot_body_names": []},
            motion_npz=None,
        )


def test_build_raw_source_dispatches_to_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    from motion_reconstruction.inference.sources import build_inference_source
    from motion_reconstruction.pipeline import MotionRuntimeBundle
    from motion_reconstruction.features.builder import FeatureBundle, FeatureSchema

    config = MotionReconstructionConfig()
    config.features.robot_anchor_body = "torso_link"
    config.features.human_anchor_body = "Hips"
    config.features.human_body_names = ["Spine1"]

    class FakeBuffer:
        valid_center_indices = torch.tensor([2, 3], dtype=torch.long)
        window_offsets = torch.tensor([-1, 0, 1], dtype=torch.long)
        window_size = 3
        robot_features = torch.zeros((6, 8), dtype=torch.float32)
        human_features = torch.ones((6, 9), dtype=torch.float32)

    class FakeRaw:
        fps = 30
        robot_body_names = ["base", "torso_link"]
        robot_joint_names = ["joint_a", "joint_b"]
        human_body_names = ["Hips", "Spine1"]
        body_pos_w = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.5]],
                [[1.0, 0.0, 0.0], [1.1, 0.0, 0.5]],
                [[2.0, 0.0, 0.0], [2.1, 0.0, 0.5]],
                [[3.0, 0.0, 0.0], [3.1, 0.0, 0.5]],
                [[4.0, 0.0, 0.0], [4.1, 0.0, 0.5]],
                [[5.0, 0.0, 0.0], [5.1, 0.0, 0.5]],
            ],
            dtype=torch.float32,
        )
        human_body_pos_w = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
                [[2.0, 0.0, 0.0], [2.0, 0.0, 1.0]],
                [[3.0, 0.0, 0.0], [3.0, 0.0, 1.0]],
                [[4.0, 0.0, 0.0], [4.0, 0.0, 1.0]],
                [[5.0, 0.0, 0.0], [5.0, 0.0, 1.0]],
            ],
            dtype=torch.float32,
        )

    runtime = MotionRuntimeBundle(
        raw=FakeRaw(),
        features=FeatureBundle(
            robot=FakeBuffer.robot_features,
            human=FakeBuffer.human_features,
            schema=FeatureSchema(
                robot_anchor_body="torso_link",
                human_anchor_body="Hips",
                human_body_names=["Spine1"],
                robot_joint_names=["joint_a", "joint_b"],
                robot_body_names=["base", "torso_link"],
                source_human_body_names=["Hips", "Spine1"],
                robot_feature_dim=8,
                human_feature_dim=9,
            ),
        ),
        buffer=FakeBuffer(),
    )

    monkeypatch.setattr("motion_reconstruction.inference.sources.build_motion_runtime", lambda *args, **kwargs: runtime)

    bundle = build_inference_source(
        source="raw",
        config=config,
        device="cpu",
        feature_schema={},
    )

    np.testing.assert_array_equal(bundle.center_indices.cpu().numpy(), np.array([2, 3], dtype=np.int64))
    np.testing.assert_allclose(bundle.robot_anchor_pos_w.cpu().numpy(), FakeRaw.body_pos_w[:, 1].numpy())
    assert bundle.display_human_body_names == ["Hips", "Spine1"]
