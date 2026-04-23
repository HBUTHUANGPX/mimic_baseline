import os
import socket
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

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
from motion_reconstruction.training.trainer import MotionReconstructionTrainer


def _write_motion_file(path: Path, joint_value: float, frames: int = 4) -> None:
    robot_quat = np.zeros((frames, 1, 4), dtype=np.float32)
    robot_quat[..., 0] = 1.0
    human_quat = np.zeros((frames, 3, 4), dtype=np.float32)
    human_quat[..., 0] = 1.0
    joint_pos = np.full((frames, 2), joint_value, dtype=np.float32)
    np.savez(
        path,
        fps=np.array(120),
        scalar_first=np.array(True),
        robot_joint_names=np.array(["hip", "knee"], dtype=object),
        robot_body_names=np.array(["torso_link"], dtype=object),
        human_joint_names=np.array(["Hips", "Spine1", "Head"], dtype=object),
        robot_joint_pos=joint_pos,
        robot_joint_vel=np.zeros((frames, 2), dtype=np.float32),
        robot_body_pos=np.zeros((frames, 1, 3), dtype=np.float32),
        robot_body_quat=robot_quat,
        robot_body_lin_vel=np.zeros((frames, 1, 3), dtype=np.float32),
        robot_body_ang_vel=np.zeros((frames, 1, 3), dtype=np.float32),
        human_global_pos=np.full((frames, 3, 3), joint_value, dtype=np.float32),
        human_global_quat=human_quat,
    )


def _build_config(tmp_path: Path, files: list[Path]) -> MotionReconstructionConfig:
    return MotionReconstructionConfig(
        data=DataConfig(files=[str(path) for path in files]),
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
            history=0,
            future=0,
            seed=7,
            log_every_steps=100,
            log_histograms=False,
            progress=False,
            checkpoint_interval_epochs=1,
        ),
        output=OutputConfig(root_dir=str(tmp_path / "runs"), run_name="ddp_smoke"),
    )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _dist_worker(rank: int, world_size: int, port: int, tmp_path_str: str, files: list[str]) -> None:
    tmp_path = Path(tmp_path_str)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    try:
        trainer = MotionReconstructionTrainer(_build_config(tmp_path, [Path(path) for path in files]))
        trainer.train()
        result = {
            "rank": rank,
            "motion_paths": list(trainer.runtime.raw.motion_paths),
            "local_centers": int(trainer.buffer.valid_center_indices.numel()),
            "robot_mean_tail": trainer.normalizers["robot"].mean[-2:].detach().cpu(),
        }
        torch.save(result, tmp_path / f"rank_{rank}.pt")
    finally:
        for name in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
            os.environ.pop(name, None)


def test_distributed_trainer_shards_files_and_uses_global_normalizer(tmp_path: Path):
    values = [1.0, 5.0, 9.0, 13.0]
    files: list[Path] = []
    for index, value in enumerate(values):
        path = tmp_path / f"motion_{index}.npz"
        _write_motion_file(path, value)
        files.append(path)

    port = _find_free_port()
    world_size = 2
    mp.spawn(
        _dist_worker,
        args=(world_size, port, str(tmp_path), [str(path) for path in files]),
        nprocs=world_size,
        join=True,
    )

    rank0 = torch.load(tmp_path / "rank_0.pt", map_location="cpu")
    rank1 = torch.load(tmp_path / "rank_1.pt", map_location="cpu")

    all_paths = {str(path) for path in files}
    rank0_paths = set(rank0["motion_paths"])
    rank1_paths = set(rank1["motion_paths"])

    assert rank0_paths
    assert rank1_paths
    assert rank0_paths.isdisjoint(rank1_paths)
    assert rank0_paths | rank1_paths == all_paths
    assert torch.allclose(rank0["robot_mean_tail"], torch.tensor([7.0, 7.0]), atol=1e-5)
    assert torch.allclose(rank1["robot_mean_tail"], torch.tensor([7.0, 7.0]), atol=1e-5)
    assert rank0["local_centers"] == 8
    assert rank1["local_centers"] == 8
    assert (tmp_path / "runs" / "ddp_smoke" / "checkpoints" / "latest.pt").is_file()
