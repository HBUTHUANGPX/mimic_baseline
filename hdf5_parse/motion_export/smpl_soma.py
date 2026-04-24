from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from ..utils.smpl_motion_tools import (
    DEFAULT_HDF5_PATH,
    compute_fps,
    quat_wxyz_to_rotvec,
    resolve_body_model_path,
    split_pose7_qwxyz_xyz,
)
from motion_reconstruction.human_pose import (
    apply_visualization_frame_xyzw,
    compute_global_joint_transforms_xyzw,
    convert_root_to_pre_visualization_frame_xyzw,
    quat_conjugate_batch_xyzw,
    quat_mul_batch_xyzw,
    quat_rotate_batch_xyzw,
)


DEFAULT_SOMA_X_ROOT = Path("/home/hpx/HPX_LOCO_2/SOMA-X")


@dataclass
class BodyFrameSelection:
    root_pose7: np.ndarray
    body_quats: np.ndarray
    betas: np.ndarray
    frame_nums: np.ndarray
    frame_timestamps: np.ndarray
    fps: float

    @property
    def num_frames(self) -> int:
        return int(self.frame_nums.shape[0])


@dataclass
class SMPLBodyMotion:
    global_orient: np.ndarray
    body_pose: np.ndarray
    transl: np.ndarray
    betas: np.ndarray
    frame_nums: np.ndarray
    frame_timestamps: np.ndarray
    fps: float

    @property
    def num_frames(self) -> int:
        return int(self.frame_nums.shape[0])


def _array_is_finite(array: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(array), axis=tuple(range(1, array.ndim)))


def build_body_valid_mask(*, root_pose7: np.ndarray, body_quats: np.ndarray, betas: np.ndarray) -> np.ndarray:
    return np.logical_and.reduce(
        [
            _array_is_finite(np.asarray(root_pose7)),
            _array_is_finite(np.asarray(body_quats)),
            _array_is_finite(np.asarray(betas)),
        ]
    )


def build_frame_timestamp_lookup(
    *, video_frame_numbers: np.ndarray, video_timestamps: np.ndarray
) -> dict[int, int]:
    frame_numbers = np.asarray(video_frame_numbers, dtype=np.int32).reshape(-1)
    timestamps = np.asarray(video_timestamps, dtype=np.int64).reshape(-1)
    if frame_numbers.shape[0] != timestamps.shape[0]:
        raise ValueError("video_frame_numbers and video_timestamps must have the same length.")
    return {int(frame_num): int(timestamp) for frame_num, timestamp in zip(frame_numbers, timestamps)}


def load_body_frame_selection(
    hdf5_path: str | Path = DEFAULT_HDF5_PATH,
    *,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
) -> BodyFrameSelection:
    hdf5_path = Path(hdf5_path)
    with h5py.File(hdf5_path, "r") as h5_file:
        total_frames = int(h5_file["full_body_mocap/frame_nums"].shape[0])
        stop = total_frames if end_frame in (-1, None) else min(int(end_frame), total_frames)
        frame_slice = slice(int(start_frame), stop, int(stride))

        root_pose7 = np.asarray(h5_file["full_body_mocap/Ts_world_root"][frame_slice], dtype=np.float32)
        body_quats = np.asarray(h5_file["full_body_mocap/body_quats"][frame_slice], dtype=np.float32)
        betas = np.asarray(h5_file["full_body_mocap/betas"][frame_slice], dtype=np.float32)
        frame_nums = np.asarray(h5_file["full_body_mocap/frame_nums"][frame_slice], dtype=np.int32)
        frame_timestamp_lookup = build_frame_timestamp_lookup(
            video_frame_numbers=np.asarray(h5_file["video/frame_number"][:], dtype=np.int32),
            video_timestamps=np.asarray(h5_file["video/device_timestamp"][:], dtype=np.int64),
        )

        video_length_sec = None
        if "video/length_sec" in h5_file:
            video_length_sec = float(np.asarray(h5_file["video/length_sec"][()]).flat[0])
        fps = compute_fps(total_frames, video_length_sec)

    frame_timestamps = np.asarray([frame_timestamp_lookup[int(frame_num)] for frame_num in frame_nums], dtype=np.int64)
    valid_mask = build_body_valid_mask(root_pose7=root_pose7, body_quats=body_quats, betas=betas)

    return BodyFrameSelection(
        root_pose7=root_pose7[valid_mask],
        body_quats=body_quats[valid_mask],
        betas=betas[valid_mask],
        frame_nums=frame_nums[valid_mask],
        frame_timestamps=frame_timestamps[valid_mask],
        fps=float(fps),
    )


def selection_to_smpl_body_motion(selection: BodyFrameSelection) -> SMPLBodyMotion:
    root_quat_wxyz, transl = split_pose7_qwxyz_xyz(selection.root_pose7)
    global_orient = quat_wxyz_to_rotvec(root_quat_wxyz)
    body_pose = quat_wxyz_to_rotvec(selection.body_quats).reshape(selection.num_frames, -1)
    if body_pose.shape[-1] == 63:
        body_pose = np.concatenate(
            [body_pose, np.zeros((selection.num_frames, 6), dtype=np.float32)],
            axis=-1,
        )
    if body_pose.shape[-1] != 69:
        raise ValueError(f"Expected SMPL body pose with 69 dims after conversion, got {body_pose.shape}.")
    return SMPLBodyMotion(
        global_orient=np.asarray(global_orient, dtype=np.float32),
        body_pose=np.asarray(body_pose, dtype=np.float32),
        transl=np.asarray(transl, dtype=np.float32),
        betas=np.asarray(selection.betas, dtype=np.float32),
        frame_nums=np.asarray(selection.frame_nums, dtype=np.int32),
        frame_timestamps=np.asarray(selection.frame_timestamps, dtype=np.int64),
        fps=float(selection.fps),
    )


def normalize_root_parent_index(parent_indices: np.ndarray) -> np.ndarray:
    normalized = np.asarray(parent_indices, dtype=np.int32).copy()
    if normalized.size > 0 and normalized[0] == 0:
        normalized[0] = -1
    return normalized


def _ensure_repo_on_sys_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _infer_model_num_betas(model_path: Path, fallback: int) -> int:
    model_path = Path(model_path)
    if model_path.suffix == ".npz":
        with np.load(model_path, allow_pickle=True) as model_data:
            shapedirs = model_data["shapedirs"]
        return int(max(1, min(int(shapedirs.shape[-1]), int(fallback))))

    if model_path.suffix == ".pkl":
        with model_path.open("rb") as handle:
            model_data = pickle.load(handle, encoding="latin1")
        shapedirs = model_data["shapedirs"]
        return int(max(1, min(int(shapedirs.shape[-1]), int(fallback))))

    return int(max(1, fallback))


def _match_beta_dimension(betas: np.ndarray, target_dim: int) -> np.ndarray:
    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim == 1:
        betas = betas[None, :]
    if betas.shape[-1] == target_dim:
        return betas.astype(np.float32, copy=False)
    if betas.shape[-1] > target_dim:
        return betas[..., :target_dim].astype(np.float32, copy=False)

    padded = np.zeros(betas.shape[:-1] + (target_dim,), dtype=np.float32)
    padded[..., : betas.shape[-1]] = betas
    return padded


def _build_smpl_model(model_path: Path, *, num_betas: int, device: str):
    _ensure_repo_on_sys_path(DEFAULT_SOMA_X_ROOT)
    import smplx
    from smplx.body_models import Struct

    ext = model_path.suffix.lstrip(".")
    if ext == "npz":
        model_data = np.load(model_path, allow_pickle=True)
        data_struct = Struct(**{key: model_data[key] for key in model_data.files})
        return smplx.SMPL(
            str(model_path),
            data_struct=data_struct,
            gender="neutral",
            num_betas=num_betas,
            batch_size=1,
        ).to(device)

    return smplx.create(
        model_path=str(model_path),
        model_type="smpl",
        gender="neutral",
        ext=ext,
        num_betas=num_betas,
        batch_size=1,
    ).to(device)


def _import_soma_x_runtime():
    _ensure_repo_on_sys_path(DEFAULT_SOMA_X_ROOT)
    from soma._compat import ensure_legacy_dependency_apis

    ensure_legacy_dependency_apis()

    import torch
    from soma.geometry.rig_utils import joint_world_to_local, remove_joint_orient_local
    from soma.geometry.transforms import matrix_to_rotvec
    from soma.pose_inversion import PoseInversion
    from soma.soma import SOMALayer

    return torch, SOMALayer, PoseInversion, joint_world_to_local, remove_joint_orient_local, matrix_to_rotvec


def _rotation_matrices_to_quat_xyzw(rot_mats: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    rotations = Rotation.from_matrix(np.asarray(rot_mats, dtype=np.float64).reshape(-1, 3, 3))
    return rotations.as_quat().reshape(np.asarray(rot_mats).shape[:-2] + (4,)).astype(np.float32)


def _pose7_from_transforms(transforms: np.ndarray) -> np.ndarray:
    positions = np.asarray(transforms[..., :3, 3], dtype=np.float32)
    quats = _rotation_matrices_to_quat_xyzw(transforms[..., :3, :3])
    return np.concatenate([positions, quats], axis=-1).astype(np.float32)


def quat_mul_batch(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    return quat_mul_batch_xyzw(q1, q2)


def quat_conjugate_batch(quat: np.ndarray) -> np.ndarray:
    return quat_conjugate_batch_xyzw(quat)


def quat_rotate_batch(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return quat_rotate_batch_xyzw(quat, vec)


def compute_global_joint_transforms(
    local_transforms: np.ndarray, parent_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return compute_global_joint_transforms_xyzw(local_transforms, parent_indices)


def apply_visualization_frame(
    positions: np.ndarray,
    rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return apply_visualization_frame_xyzw(positions, rotations)


def convert_root_to_pre_visualization_frame(local_transforms: np.ndarray) -> np.ndarray:
    return convert_root_to_pre_visualization_frame_xyzw(local_transforms)


def ensure_local_transforms_pre_visualization_frame(
    *,
    local_transforms: np.ndarray,
    parent_indices: np.ndarray,
    joint_names: list[str],
) -> np.ndarray:
    local_transforms = np.asarray(local_transforms, dtype=np.float32)
    if "Hips" not in joint_names or "Head" not in joint_names:
        return local_transforms

    hips_idx = joint_names.index("Hips")
    head_idx = joint_names.index("Head")
    global_positions, _ = compute_global_joint_transforms(local_transforms, parent_indices)
    spine_vector = np.asarray(global_positions[:, head_idx] - global_positions[:, hips_idx], dtype=np.float32)
    mean_abs = np.mean(np.abs(spine_vector), axis=0)
    if mean_abs[2] > mean_abs[1]:
        return convert_root_to_pre_visualization_frame(local_transforms)
    return local_transforms


def run_soma_inversion(
    motion: SMPLBodyMotion,
    *,
    device: str = "cuda",
    batch_size: int | None = None,
    body_iters: int = 2,
    finger_iters: int = 0,
    full_iters: int = 1,
    autograd_iters: int = 0,
    autograd_lr: float = 5e-3,
    soma_x_root: str | Path = DEFAULT_SOMA_X_ROOT,
    smpl_model_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    soma_root = Path(soma_x_root)
    if not str(device).startswith("cuda"):
        raise ValueError("This exporter only supports CUDA execution.")

    torch, SOMALayer, PoseInversion, joint_world_to_local, remove_joint_orient_local, matrix_to_rotvec = _import_soma_x_runtime()
    _ensure_repo_on_sys_path(soma_root)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this exporter, but no CUDA device is available.")

    smpl_model_path = resolve_body_model_path("smpl", smpl_model_path)
    num_betas = _infer_model_num_betas(Path(smpl_model_path), fallback=int(np.asarray(motion.betas).shape[-1]))
    smpl_model = _build_smpl_model(Path(smpl_model_path), num_betas=num_betas, device=device)

    motion_betas = _match_beta_dimension(motion.betas, num_betas)
    if motion_betas.shape[0] == 1 and motion.num_frames > 1:
        motion_betas = np.broadcast_to(motion_betas, (motion.num_frames, motion_betas.shape[-1])).copy()

    body_pose = torch.as_tensor(motion.body_pose, dtype=torch.float32, device=device)
    global_orient = torch.as_tensor(motion.global_orient, dtype=torch.float32, device=device)
    transl = torch.as_tensor(motion.transl, dtype=torch.float32, device=device)
    betas = torch.as_tensor(motion_betas, dtype=torch.float32, device=device)

    soma = SOMALayer(
        soma_root / "assets",
        identity_model_type="smpl",
        device=device,
        mode="warp",
        identity_model_kwargs={
            "model_path": str(smpl_model_path),
            "num_betas": num_betas,
        },
    )
    inv = PoseInversion(soma, low_lod=True)
    inv.prepare_identity(betas[:1])

    with torch.no_grad():
        warmup_out = smpl_model(
            body_pose=body_pose[:1],
            global_orient=global_orient[:1],
            betas=betas[:1],
            transl=transl[:1],
        )
    inv.fit(
        warmup_out.vertices,
        body_iters=body_iters,
        finger_iters=finger_iters,
        full_iters=full_iters,
        autograd_iters=autograd_iters,
        autograd_lr=autograd_lr,
    )

    active_batch_size = batch_size or motion.num_frames
    all_rotations = []
    all_root_transl = []
    all_errors = []
    for start in range(0, motion.num_frames, active_batch_size):
        end = min(start + active_batch_size, motion.num_frames)
        with torch.no_grad():
            smpl_out = smpl_model(
                body_pose=body_pose[start:end],
                global_orient=global_orient[start:end],
                betas=betas[start:end],
                transl=transl[start:end],
            )
        result = inv.fit(
            smpl_out.vertices,
            body_iters=body_iters,
            finger_iters=finger_iters,
            full_iters=full_iters,
            autograd_iters=autograd_iters,
            autograd_lr=autograd_lr,
        )
        all_rotations.append(result["rotations"].detach().cpu())
        all_root_transl.append(result["root_translation"].detach().cpu())
        all_errors.append(result["per_vertex_error"].detach().cpu())

    rotations = torch.cat(all_rotations, dim=0).to(device)
    root_transl = torch.cat(all_root_transl, dim=0).to(device)
    per_vertex_error = torch.cat(all_errors, dim=0).cpu().numpy().astype(np.float32)

    active_soma = inv.soma
    bind_transforms = active_soma._cached_bind_transforms_world
    rest_shape = active_soma._cached_rest_shape
    if bind_transforms.shape[0] == 1 and rotations.shape[0] > 1:
        bind_transforms = bind_transforms.expand(rotations.shape[0], -1, -1, -1)
    if rest_shape.shape[0] == 1 and rotations.shape[0] > 1:
        rest_shape = rest_shape.expand(rotations.shape[0], -1, -1)
    active_soma.batched_skinning.rebind(bind_transforms, rest_shape)
    with torch.no_grad():
        _, world_transforms = active_soma.batched_skinning.pose(
            rotations,
            root_transl,
            absolute_pose=True,
            return_transforms=True,
        )
    local_transforms = joint_world_to_local(world_transforms, active_soma.joint_parent_ids)
    relative_rotations = remove_joint_orient_local(
        rotations,
        active_soma._t_pose_orient,
        active_soma._t_pose_orient_parent_T,
    )
    soma_poses = matrix_to_rotvec(relative_rotations.reshape(-1, 3, 3)).reshape(
        relative_rotations.shape[0],
        relative_rotations.shape[1],
        3,
    )

    return {
        "joint_names": list(active_soma.rig_data["joint_names"]),
        "parent_indices": active_soma.joint_parent_ids.detach().cpu().numpy().astype(np.int32),
        "reference_local_transforms": _pose7_from_transforms(active_soma.t_pose_local.detach().cpu().numpy()),
        "local_transforms": _pose7_from_transforms(local_transforms.detach().cpu().numpy()),
        "world_transforms": _pose7_from_transforms(world_transforms.detach().cpu().numpy()),
        "soma_poses": soma_poses.detach().cpu().numpy().astype(np.float32),
        "soma_transl": root_transl.detach().cpu().numpy().astype(np.float32),
        "soma_joint_orient": active_soma._t_pose_orient.detach().cpu().numpy().astype(np.float32),
        "per_vertex_error": per_vertex_error,
    }
