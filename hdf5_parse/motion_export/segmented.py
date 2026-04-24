from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import (
    DEFAULT_HDF5_PATH,
    DEFAULT_SOMA_X_ROOT,
    BodyFrameSelection,
    SMPLBodyMotion,
    ensure_local_transforms_pre_visualization_frame,
    load_body_frame_selection,
    normalize_root_parent_index,
    run_soma_inversion,
    selection_to_smpl_body_motion,
)
from .bvh import (
    canonicalize_motion_local_transforms_for_bvh,
    write_soma_bvh,
)


DEFAULT_SOMA_BVH_OUTPUT_DIR = Path("hdf5_parse/out/soma_bvh")
DEFAULT_SMPL_OUTPUT_DIR = Path("hdf5_parse/out/smpl")
DEFAULT_FILENAME_PREFIX = "annotation"

def split_contiguous_frame_ranges(frame_nums: np.ndarray, *, expected_step: int = 1) -> list[tuple[int, int]]:
    frame_nums = np.asarray(frame_nums, dtype=np.int64).reshape(-1)
    if frame_nums.size == 0:
        return []
    if expected_step <= 0:
        raise ValueError(f"expected_step must be positive, got {expected_step}.")

    split_points = np.flatnonzero(np.diff(frame_nums) != int(expected_step)) + 1
    starts = np.concatenate([np.array([0], dtype=np.int64), split_points])
    ends = np.concatenate([split_points, np.array([frame_nums.shape[0]], dtype=np.int64)])
    return [(int(start), int(end)) for start, end in zip(starts.tolist(), ends.tolist())]


def slice_body_frame_selection(selection: BodyFrameSelection, start_idx: int, end_idx: int) -> BodyFrameSelection:
    return BodyFrameSelection(
        root_pose7=np.asarray(selection.root_pose7[start_idx:end_idx], dtype=np.float32),
        body_quats=np.asarray(selection.body_quats[start_idx:end_idx], dtype=np.float32),
        betas=np.asarray(selection.betas[start_idx:end_idx], dtype=np.float32),
        frame_nums=np.asarray(selection.frame_nums[start_idx:end_idx], dtype=np.int32),
        frame_timestamps=np.asarray(selection.frame_timestamps[start_idx:end_idx], dtype=np.int64),
        fps=float(selection.fps),
    )


def slice_smpl_body_motion(motion: SMPLBodyMotion, start_idx: int, end_idx: int) -> SMPLBodyMotion:
    return SMPLBodyMotion(
        global_orient=np.asarray(motion.global_orient[start_idx:end_idx], dtype=np.float32),
        body_pose=np.asarray(motion.body_pose[start_idx:end_idx], dtype=np.float32),
        transl=np.asarray(motion.transl[start_idx:end_idx], dtype=np.float32),
        betas=np.asarray(motion.betas[start_idx:end_idx], dtype=np.float32),
        frame_nums=np.asarray(motion.frame_nums[start_idx:end_idx], dtype=np.int32),
        frame_timestamps=np.asarray(motion.frame_timestamps[start_idx:end_idx], dtype=np.int64),
        fps=float(motion.fps),
    )


def build_segment_file_stem(frame_timestamps: np.ndarray, *, prefix: str = DEFAULT_FILENAME_PREFIX) -> str:
    frame_timestamps = np.asarray(frame_timestamps, dtype=np.int64).reshape(-1)
    if frame_timestamps.size == 0:
        raise ValueError("frame_timestamps must contain at least one element.")
    return f"{prefix}_{int(frame_timestamps[0])}_{int(frame_timestamps[-1])}"


def save_smpl_motion_npz(motion: SMPLBodyMotion, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fps": np.asarray(int(round(float(motion.fps))), dtype=np.int32),
        "num_frames": np.asarray(motion.num_frames, dtype=np.int32),
        "frame_nums": np.asarray(motion.frame_nums, dtype=np.int32),
        "frame_timestamps": np.asarray(motion.frame_timestamps, dtype=np.int64),
        "smpl_global_orient": np.asarray(motion.global_orient, dtype=np.float32),
        "smpl_body_pose": np.asarray(motion.body_pose, dtype=np.float32),
        "smpl_transl": np.asarray(motion.transl, dtype=np.float32),
        "smpl_betas": np.asarray(motion.betas, dtype=np.float32),
    }
    np.savez(output_path, **payload)
    return output_path


def export_segmented_smpl_and_soma_bvh(
    hdf5_path: str | Path = DEFAULT_HDF5_PATH,
    *,
    smpl_output_dir: str | Path = DEFAULT_SMPL_OUTPUT_DIR,
    soma_bvh_output_dir: str | Path = DEFAULT_SOMA_BVH_OUTPUT_DIR,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    device: str = "cuda",
    batch_size: int | None = None,
    soma_x_root: str | Path = DEFAULT_SOMA_X_ROOT,
    smpl_model_path: str | Path | None = None,
    filename_prefix: str = DEFAULT_FILENAME_PREFIX,
) -> dict[str, list[Path]]:
    selection = load_body_frame_selection(
        hdf5_path=hdf5_path,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
    )
    ranges = split_contiguous_frame_ranges(selection.frame_nums, expected_step=max(1, int(stride)))
    if not ranges:
        return {"smpl_paths": [], "soma_bvh_paths": []}

    motion = selection_to_smpl_body_motion(selection)
    soma_output = run_soma_inversion(
        motion,
        device=device,
        batch_size=batch_size,
        soma_x_root=soma_x_root,
        smpl_model_path=smpl_model_path,
    )

    joint_names = list(soma_output["joint_names"])
    parent_indices = normalize_root_parent_index(soma_output["parent_indices"])
    reference_local_transforms = np.asarray(soma_output["reference_local_transforms"], dtype=np.float32)
    human_local_transforms = canonicalize_motion_local_transforms_for_bvh(
        local_transforms=ensure_local_transforms_pre_visualization_frame(
            local_transforms=np.asarray(soma_output["local_transforms"], dtype=np.float32),
            parent_indices=parent_indices,
            joint_names=joint_names,
        ),
        parent_indices=parent_indices,
    )

    smpl_output_dir = Path(smpl_output_dir)
    soma_bvh_output_dir = Path(soma_bvh_output_dir)
    smpl_output_dir.mkdir(parents=True, exist_ok=True)
    soma_bvh_output_dir.mkdir(parents=True, exist_ok=True)

    smpl_paths: list[Path] = []
    soma_bvh_paths: list[Path] = []
    for start_idx, end_idx in ranges:
        segment_motion = slice_smpl_body_motion(motion, start_idx, end_idx)
        stem = build_segment_file_stem(segment_motion.frame_timestamps, prefix=filename_prefix)
        smpl_path = save_smpl_motion_npz(segment_motion, smpl_output_dir / f"{stem}.npz")
        bvh_path = write_soma_bvh(
            output_path=soma_bvh_output_dir / f"{stem}.bvh",
            joint_names=joint_names,
            parent_indices=parent_indices,
            reference_local_transforms=reference_local_transforms,
            local_transforms=human_local_transforms[start_idx:end_idx],
            fps=float(motion.fps),
        )
        smpl_paths.append(smpl_path)
        soma_bvh_paths.append(bvh_path)

    return {
        "smpl_paths": smpl_paths,
        "soma_bvh_paths": soma_bvh_paths,
    }
