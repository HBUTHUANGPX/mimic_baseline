"""可供其它工程直接调用的重构可视化入口。"""

from __future__ import annotations

from pathlib import Path

from motion_reconstruction.config.schema import MotionReconstructionConfig
from motion_reconstruction.evaluation import ReconstructionResult, reconstruct_motion

from .mujoco_viewer import play_reconstruction


def visualize_reconstruction_from_source(
    *,
    config: MotionReconstructionConfig,
    checkpoint_path: str | Path,
    xml_path: str | Path,
    source: str = "raw",
    motion_npz: str | Path | None = None,
    inference_path: str = "both",
    device: str = "cpu",
    batch_size: int = 4096,
    max_frames: int | None = None,
    pair: str = "both",
    fps: int | None = None,
    loop: bool = False,
    keep_world: bool = False,
) -> ReconstructionResult:
    """先重构，再直接启动 MuJoCo viewer。"""
    result = reconstruct_motion(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
        batch_size=batch_size,
        max_frames=max_frames,
        source=source,
        motion_npz=motion_npz,
        inference_path=inference_path,
    )
    play_reconstruction(
        result=result,
        xml_path=xml_path,
        pair=pair,
        fps=fps,
        loop=loop,
        keep_world=keep_world,
    )
    return result


def visualize_hdf5_human_npz(
    *,
    config: MotionReconstructionConfig,
    checkpoint_path: str | Path,
    xml_path: str | Path,
    motion_npz: str | Path,
    device: str = "cpu",
    batch_size: int = 4096,
    max_frames: int | None = None,
    fps: int | None = None,
    loop: bool = False,
    keep_world: bool = False,
) -> ReconstructionResult:
    """专门给 human motion npz 使用的可视化入口。"""
    return visualize_reconstruction_from_source(
        config=config,
        checkpoint_path=checkpoint_path,
        xml_path=xml_path,
        source="hdf5-human",
        motion_npz=motion_npz,
        inference_path="human",
        device=device,
        batch_size=batch_size,
        max_frames=max_frames,
        pair="human",
        fps=fps,
        loop=loop,
        keep_world=keep_world,
    )
