from pathlib import Path

import numpy as np

from motion_reconstruction.config.schema import (
    DataConfig,
    FeatureConfig,
    LossConfig,
    ModelConfig,
    MotionReconstructionConfig,
    OutputConfig,
    QuantizerConfig,
    TrainConfig,
)
from motion_reconstruction.evaluation.reconstruct import reconstruct_motion
from motion_reconstruction.training.trainer import MotionReconstructionTrainer


def test_reconstruct_motion_loads_checkpoint_and_returns_current_frame_features(tmp_path: Path):
    motion_file = tmp_path / "motion.npz"
    frames = 6
    robot_quat = np.zeros((frames, 1, 4), dtype=np.float32)
    robot_quat[..., 0] = 1.0
    human_quat = np.zeros((frames, 3, 4), dtype=np.float32)
    human_quat[..., 0] = 1.0
    np.savez(
        motion_file,
        fps=np.array(120),
        scalar_first=np.array(True),
        robot_joint_names=np.array(["hip", "knee"], dtype=object),
        robot_body_names=np.array(["torso_link"], dtype=object),
        human_joint_names=np.array(["Hips", "Spine1", "Head"], dtype=object),
        robot_joint_pos=np.random.randn(frames, 2).astype(np.float32),
        robot_joint_vel=np.zeros((frames, 2), dtype=np.float32),
        robot_body_pos=np.zeros((frames, 1, 3), dtype=np.float32),
        robot_body_quat=robot_quat,
        robot_body_lin_vel=np.zeros((frames, 1, 3), dtype=np.float32),
        robot_body_ang_vel=np.zeros((frames, 1, 3), dtype=np.float32),
        human_global_pos=np.random.randn(frames, 3, 3).astype(np.float32),
        human_global_quat=human_quat,
    )
    config = MotionReconstructionConfig(
        data=DataConfig(files=[str(motion_file)]),
        features=FeatureConfig(human_body_names=["Spine1", "Head"]),
        model=ModelConfig(
            latent_dim=2,
            robot_encoder_hidden_dims=[8],
            human_encoder_hidden_dims=[8],
            decoder_hidden_dims=[8],
            quantizer=QuantizerConfig(type="fsq", levels=[5, 5]),
        ),
        loss=LossConfig(),
        train=TrainConfig(
            device="cpu",
            epochs=1,
            batch_size=2,
            history=1,
            future=1,
            log_every_steps=100,
            log_histograms=False,
            checkpoint_interval_epochs=1,
            progress=False,
        ),
        output=OutputConfig(root_dir=str(tmp_path / "runs"), run_name="smoke"),
    )
    trainer = MotionReconstructionTrainer(config)
    trainer.train()

    result = reconstruct_motion(
        config=config,
        checkpoint_path=tmp_path / "runs" / "smoke" / "checkpoints" / "latest.pt",
        device="cpu",
        batch_size=2,
    )

    assert result.original_robot_feature.shape == (4, 8)
    assert result.recon_from_robot_feature.shape == (4, 8)
    assert result.recon_from_human_feature.shape == (4, 8)
    assert result.human_body_pos_w.shape == (4, 3, 3)
    assert result.center_indices.tolist() == [1, 2, 3, 4]
    metrics = result.metrics()
    assert set(metrics) == {
        "robot_from_robot_mse",
        "robot_from_human_mse",
        "joint_from_robot_mse",
        "joint_from_human_mse",
    }
