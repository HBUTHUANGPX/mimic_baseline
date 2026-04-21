import torch

from motion_reconstruction.data.raw_motion import RawMotionDataset
from motion_reconstruction.features.builder import FeatureBuilder, FeatureBuilderConfig


def test_feature_builder_creates_robot_and_human_frame_features():
    raw = RawMotionDataset(
        fps=120,
        joint_pos=torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
        joint_vel=torch.zeros(2, 2),
        body_pos_w=torch.zeros(2, 2, 3),
        body_quat_w=torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            ]
        ),
        body_lin_vel_w=torch.zeros(2, 2, 3),
        body_ang_vel_w=torch.zeros(2, 2, 3),
        human_body_pos_w=torch.tensor(
            [
                [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [1.0, 4.0, 3.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]],
            ]
        ),
        human_body_quat_w=torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            ]
        ),
        robot_joint_names=["hip", "knee"],
        robot_body_names=["torso_link", "left_foot"],
        human_body_names=["Hips", "Spine1", "Head"],
        motion_lengths=torch.tensor([2]),
        motion_start_indices=torch.tensor([0]),
        motion_groups=["group"],
        motion_paths=["motion.npz"],
    )
    builder = FeatureBuilder(
        FeatureBuilderConfig(
            robot_anchor_body="torso_link",
            human_anchor_body="Hips",
            human_body_names=["Spine1", "Head"],
        )
    )

    features = builder.build(raw)

    assert features.robot.shape == (2, 8)
    assert features.human.shape == (2, 12)
    assert torch.allclose(features.robot[0, :6], torch.tensor([1, 0, 0, 0, 1, 0.0]))
    assert torch.allclose(features.robot[:, 6:], raw.joint_pos)
    assert torch.allclose(features.human[0, :6], torch.tensor([1, 0, 0, 0, 1, 0.0]))
    assert torch.allclose(
        features.human[0, 6:],
        torch.tensor([1.0, 0.0, 0.0, 0.0, 2.0, 0.0]),
    )
    assert features.schema.robot_feature_dim == 8
    assert features.schema.human_feature_dim == 12
