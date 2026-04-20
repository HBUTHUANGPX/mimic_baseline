from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_HDF5_PATH = Path(__file__).resolve().parent / "hdf5" / "annotation.hdf5"
DEFAULT_SMPLH_MODEL_CANDIDATES = (
    Path("/home/hpx/HPX_Loco/loco-mujoco/datasets/smplh/SMPLH_NEUTRAL.pkl"),
    Path("/home/hpx/2025_5_24/loco-mujoco/datasets/smpl/SMPLH_NEUTRAL.pkl"),
)
DEFAULT_SMPL_MODEL_CANDIDATES = (
    Path("/home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz"),
    Path("/home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.pkl"),
)


@dataclass
class SMPLMotionClip:
    model_type: str
    global_orient: np.ndarray
    body_pose: np.ndarray
    transl: np.ndarray
    betas: np.ndarray | None
    fps: float
    frame_nums: np.ndarray
    left_hand_pose: np.ndarray | None = None
    right_hand_pose: np.ndarray | None = None

    @property
    def num_frames(self) -> int:
        return int(self.global_orient.shape[0])


def split_pose7_qwxyz_xyz(pose7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose7 = np.asarray(pose7, dtype=np.float32)
    if pose7.shape[-1] != 7:
        raise ValueError(f"Expected pose7 with trailing dim 7, got {pose7.shape}.")
    return pose7[..., :4], pose7[..., 4:]


def quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    if quat_wxyz.shape[-1] != 4:
        raise ValueError(f"Expected quaternion with trailing dim 4, got {quat_wxyz.shape}.")
    return quat_wxyz[..., [1, 2, 3, 0]]


def quat_wxyz_to_rotvec(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_xyzw = quat_wxyz_to_xyzw(quat_wxyz)
    rot = Rotation.from_quat(quat_xyzw.reshape(-1, 4))
    return rot.as_rotvec().reshape(quat_xyzw.shape[:-1] + (3,)).astype(np.float32)


def compute_fps(num_frames: int, video_length_sec: float | None) -> float:
    if video_length_sec is None or video_length_sec <= 0:
        return 20.0
    return float(num_frames) / float(video_length_sec)


def _array_is_finite(array: np.ndarray | None) -> np.ndarray | None:
    if array is None:
        return None
    return np.all(np.isfinite(array), axis=tuple(range(1, array.ndim)))


def apply_frame_valid_mask(array: np.ndarray | None, valid_mask: np.ndarray) -> np.ndarray | None:
    if array is None:
        return None
    return np.asarray(array[valid_mask])


def build_smplh_motion_clip(
    *,
    root_pose7: np.ndarray,
    body_quats: np.ndarray,
    left_hand_quats: np.ndarray,
    right_hand_quats: np.ndarray,
    betas: np.ndarray | None,
    frame_nums: np.ndarray,
    fps: float,
) -> SMPLMotionClip:
    root_quat_wxyz, transl = split_pose7_qwxyz_xyz(root_pose7)
    global_orient = quat_wxyz_to_rotvec(root_quat_wxyz)
    body_pose = quat_wxyz_to_rotvec(body_quats).reshape(body_quats.shape[0], -1)
    left_hand_pose = quat_wxyz_to_rotvec(left_hand_quats).reshape(left_hand_quats.shape[0], -1)
    right_hand_pose = quat_wxyz_to_rotvec(right_hand_quats).reshape(right_hand_quats.shape[0], -1)
    return SMPLMotionClip(
        model_type="smplh",
        global_orient=global_orient,
        body_pose=body_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        transl=np.asarray(transl, dtype=np.float32),
        betas=None if betas is None else np.asarray(betas, dtype=np.float32),
        fps=float(fps),
        frame_nums=np.asarray(frame_nums, dtype=np.int32),
    )


def convert_smplh_motion_clip_to_smpl(clip: SMPLMotionClip) -> SMPLMotionClip:
    if clip.model_type != "smplh":
        raise ValueError(f"Expected a smplh clip, got {clip.model_type}.")
    if clip.body_pose.shape[-1] != 63:
        raise ValueError(f"Expected SMPL-H body pose with 63 dims, got {clip.body_pose.shape}.")
    padded_body_pose = np.concatenate(
        [
            np.asarray(clip.body_pose, dtype=np.float32),
            np.zeros((clip.num_frames, 6), dtype=np.float32),
        ],
        axis=-1,
    )
    return SMPLMotionClip(
        model_type="smpl",
        global_orient=np.asarray(clip.global_orient, dtype=np.float32),
        body_pose=padded_body_pose,
        transl=np.asarray(clip.transl, dtype=np.float32),
        betas=None if clip.betas is None else np.asarray(clip.betas, dtype=np.float32),
        fps=float(clip.fps),
        frame_nums=np.asarray(clip.frame_nums, dtype=np.int32),
        left_hand_pose=None,
        right_hand_pose=None,
    )


def resolve_body_model_path(model_type: str, explicit_path: str | Path | None = None) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Body model path does not exist: {path}")
        return path

    model_type = str(model_type).lower()
    if model_type == "smplh":
        candidates = DEFAULT_SMPLH_MODEL_CANDIDATES
    elif model_type == "smpl":
        candidates = DEFAULT_SMPL_MODEL_CANDIDATES
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    if model_type == "smpl":
        candidates = tuple(sorted(candidates, key=lambda path: (path.suffix != ".npz", str(path))))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No default {model_type} body model was found in {candidates}.")


def load_smplh_motion_clip(
    hdf5_path: str | Path = DEFAULT_HDF5_PATH,
    *,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    drop_invalid_frames: bool = True,
) -> SMPLMotionClip:
    hdf5_path = Path(hdf5_path)
    with h5py.File(hdf5_path, "r") as h5_file:
        keypoints_ds = h5_file["full_body_mocap/keypoints"]
        num_frames = int(keypoints_ds.shape[0])
        stop = num_frames if end_frame in (-1, None) else min(int(end_frame), num_frames)
        frame_slice = slice(int(start_frame), stop, int(stride))

        root_pose7 = np.asarray(h5_file["full_body_mocap/Ts_world_root"][frame_slice], dtype=np.float32)
        body_quats = np.asarray(h5_file["full_body_mocap/body_quats"][frame_slice], dtype=np.float32)
        left_hand_quats = np.asarray(h5_file["full_body_mocap/left_hand_quats"][frame_slice], dtype=np.float32)
        right_hand_quats = np.asarray(h5_file["full_body_mocap/right_hand_quats"][frame_slice], dtype=np.float32)
        betas = np.asarray(h5_file["full_body_mocap/betas"][frame_slice], dtype=np.float32)
        frame_nums = np.asarray(h5_file["full_body_mocap/frame_nums"][frame_slice], dtype=np.int32)

        video_length_sec = None
        if "video/length_sec" in h5_file:
            video_length_sec = float(np.asarray(h5_file["video/length_sec"][()]).flat[0])
        fps = compute_fps(num_frames, video_length_sec)

    if drop_invalid_frames:
        valid_mask_parts = [
            _array_is_finite(root_pose7),
            _array_is_finite(body_quats),
            _array_is_finite(left_hand_quats),
            _array_is_finite(right_hand_quats),
            _array_is_finite(betas),
        ]
        valid_mask = np.logical_and.reduce([part for part in valid_mask_parts if part is not None])
        root_pose7 = apply_frame_valid_mask(root_pose7, valid_mask)
        body_quats = apply_frame_valid_mask(body_quats, valid_mask)
        left_hand_quats = apply_frame_valid_mask(left_hand_quats, valid_mask)
        right_hand_quats = apply_frame_valid_mask(right_hand_quats, valid_mask)
        betas = apply_frame_valid_mask(betas, valid_mask)
        frame_nums = apply_frame_valid_mask(frame_nums, valid_mask)

    return build_smplh_motion_clip(
        root_pose7=root_pose7,
        body_quats=body_quats,
        left_hand_quats=left_hand_quats,
        right_hand_quats=right_hand_quats,
        betas=betas,
        frame_nums=frame_nums,
        fps=fps,
    )


def export_motion_clip_npz(clip: SMPLMotionClip, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": np.asarray(clip.model_type),
        "global_orient": np.asarray(clip.global_orient, dtype=np.float32),
        "body_pose": np.asarray(clip.body_pose, dtype=np.float32),
        "transl": np.asarray(clip.transl, dtype=np.float32),
        "fps": np.asarray(clip.fps, dtype=np.float32),
        "frame_nums": np.asarray(clip.frame_nums, dtype=np.int32),
    }
    if clip.betas is not None:
        payload["betas"] = np.asarray(clip.betas, dtype=np.float32)
    if clip.left_hand_pose is not None:
        payload["left_hand_pose"] = np.asarray(clip.left_hand_pose, dtype=np.float32)
    if clip.right_hand_pose is not None:
        payload["right_hand_pose"] = np.asarray(clip.right_hand_pose, dtype=np.float32)
    np.savez(output_path, **payload)
    return output_path
