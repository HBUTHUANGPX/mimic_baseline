from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from smpl_motion_tools import (
    DEFAULT_HDF5_PATH,
    compute_fps,
    quat_wxyz_to_rotvec,
    resolve_body_model_path,
    split_pose7_qwxyz_xyz,
)


UNKNOWN_TEXT = "UNKNOWN"
DEFAULT_OUTPUT_PATH = Path("hdf5_parse/out/annotation_soma.npz")
DEFAULT_DUAL_FSQ_PATH = Path("motion_reconstruction/configs/dual_fsq.yaml")
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


def load_caption_json(hdf5_path: str | Path = DEFAULT_HDF5_PATH) -> dict[str, Any]:
    with h5py.File(Path(hdf5_path), "r") as h5_file:
        raw = h5_file["caption"][()]
    if isinstance(raw, bytes):
        raw = raw.decode()
    return json.loads(raw)


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


def _normalize_text_value(value: Any) -> str:
    text = str(value).strip()
    return text if text else UNKNOWN_TEXT


def _make_text_array(values: list[str]) -> np.ndarray:
    if not values:
        values = [UNKNOWN_TEXT]
    max_len = max(len(value) for value in values)
    return np.asarray(values, dtype=f"<U{max(1, max_len)}")


def _build_index_sequence(texts: list[str], values: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    text_to_index = {UNKNOWN_TEXT: 0}
    deduped = [UNKNOWN_TEXT]
    indices = np.zeros(values.shape[0], dtype=np.int32)
    for frame_idx, raw_text in enumerate(values):
        text = _normalize_text_value(raw_text)
        if text not in text_to_index:
            text_to_index[text] = len(deduped)
            deduped.append(text)
        indices[frame_idx] = text_to_index[text]
    return _make_text_array(deduped), indices


def align_caption_texts_to_frames(*, caption: dict[str, Any], frame_timestamps: np.ndarray) -> dict[str, np.ndarray]:
    frame_timestamps = np.asarray(frame_timestamps, dtype=np.int64).reshape(-1)

    main_task_values = np.full(frame_timestamps.shape[0], _normalize_text_value(caption["config"]["Main Task"]), dtype=object)
    sub_task_values = np.full(frame_timestamps.shape[0], UNKNOWN_TEXT, dtype=object)
    action_values = np.full(frame_timestamps.shape[0], UNKNOWN_TEXT, dtype=object)
    interaction_values = np.full(frame_timestamps.shape[0], UNKNOWN_TEXT, dtype=object)

    for segment in caption.get("segments", []):
        segment_start = int(segment["start_frame"])
        segment_end = int(segment["end_frame"])
        segment_mask = (frame_timestamps >= segment_start) & (frame_timestamps <= segment_end)
        sub_task_values[segment_mask] = _normalize_text_value(segment.get("Sub Task", UNKNOWN_TEXT))

        for action in segment.get("Current Action", []):
            action_start = int(action["start_frame"])
            action_end = int(action["end_frame"])
            action_mask = (frame_timestamps >= action_start) & (frame_timestamps <= action_end)
            action_values[action_mask] = _normalize_text_value(action.get("label", UNKNOWN_TEXT))

        interaction_items = sorted(
            ((int(timestamp), _normalize_text_value(text)) for timestamp, text in segment.get("interaction", {}).items()),
            key=lambda item: item[0],
        )
        for item_idx, (interaction_start, interaction_text) in enumerate(interaction_items):
            interaction_end = interaction_items[item_idx + 1][0] - 1 if item_idx + 1 < len(interaction_items) else segment_end
            interaction_mask = (frame_timestamps >= interaction_start) & (frame_timestamps <= interaction_end)
            interaction_values[interaction_mask] = interaction_text

    main_task_texts, main_task_indices = _build_index_sequence([UNKNOWN_TEXT], main_task_values)
    sub_task_texts, sub_task_indices = _build_index_sequence([UNKNOWN_TEXT], sub_task_values)
    action_texts, action_indices = _build_index_sequence([UNKNOWN_TEXT], action_values)
    interaction_texts, interaction_indices = _build_index_sequence([UNKNOWN_TEXT], interaction_values)

    return {
        "main_task_texts": main_task_texts,
        "sub_task_texts": sub_task_texts,
        "current_action_texts": action_texts,
        "interaction_texts": interaction_texts,
        "main_task_text_indices": main_task_indices,
        "sub_task_text_indices": sub_task_indices,
        "current_action_text_indices": action_indices,
        "interaction_text_indices": interaction_indices,
    }


def _rotation_matrices_to_quat_xyzw(rot_mats: np.ndarray) -> np.ndarray:
    rotations = Rotation.from_matrix(np.asarray(rot_mats, dtype=np.float64).reshape(-1, 3, 3))
    return rotations.as_quat().reshape(np.asarray(rot_mats).shape[:-2] + (4,)).astype(np.float32)


def _pose7_from_transforms(transforms: np.ndarray) -> np.ndarray:
    positions = np.asarray(transforms[..., :3, 3], dtype=np.float32)
    quats = _rotation_matrices_to_quat_xyzw(transforms[..., :3, :3])
    return np.concatenate([positions, quats], axis=-1).astype(np.float32)


def quat_mul_batch(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.moveaxis(q1, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def quat_conjugate_batch(quat: np.ndarray) -> np.ndarray:
    result = np.array(quat, dtype=np.float32, copy=True)
    result[..., :3] *= -1.0
    return result


def quat_rotate_batch(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q_xyz = quat[..., :3]
    qw = quat[..., 3:4]
    uv = np.cross(q_xyz, vec)
    uuv = np.cross(q_xyz, uv)
    return (vec + 2.0 * (qw * uv + uuv)).astype(np.float32, copy=False)


def compute_global_joint_transforms(
    local_transforms: np.ndarray, parent_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    num_frames, num_joints = local_transforms.shape[:2]
    global_positions = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    global_rotations = np.zeros((num_frames, num_joints, 4), dtype=np.float32)

    local_positions = local_transforms[..., :3]
    local_rotations = local_transforms[..., 3:7]

    for joint_idx in range(num_joints):
        parent_idx = int(parent_indices[joint_idx])
        if parent_idx < 0:
            global_positions[:, joint_idx] = local_positions[:, joint_idx]
            global_rotations[:, joint_idx] = local_rotations[:, joint_idx]
            continue

        parent_rot = global_rotations[:, parent_idx]
        parent_pos = global_positions[:, parent_idx]
        global_positions[:, joint_idx] = parent_pos + quat_rotate_batch(parent_rot, local_positions[:, joint_idx])
        global_rotations[:, joint_idx] = quat_mul_batch(parent_rot, local_rotations[:, joint_idx])

    return global_positions, global_rotations


def apply_visualization_frame(
    positions: np.ndarray, rotations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    y_up_to_z_up = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32)
    expanded = np.broadcast_to(y_up_to_z_up, rotations.shape)
    corrected_positions = quat_rotate_batch(expanded, positions)
    corrected_rotations = quat_mul_batch(
        quat_mul_batch(expanded, rotations),
        quat_conjugate_batch(expanded),
    )
    return corrected_positions.astype(np.float32, copy=False), corrected_rotations.astype(np.float32, copy=False)


def drop_soma_virtual_root(
    *,
    joint_names: list[str],
    parent_indices: np.ndarray,
    reference_local_transforms: np.ndarray,
    local_transforms: np.ndarray,
    global_transforms: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray] | tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not joint_names or joint_names[0] != "Root":
        raise ValueError("Expected SOMA joint list with virtual Root at index 0.")

    dropped_names = list(joint_names[1:])
    parent_indices = np.asarray(parent_indices, dtype=np.int32)
    dropped_parents = np.where(parent_indices[1:] == 0, -1, parent_indices[1:] - 1).astype(np.int32)
    dropped_reference = np.asarray(reference_local_transforms[1:], dtype=np.float32)
    dropped_local = np.asarray(local_transforms[:, 1:], dtype=np.float32)
    if global_transforms is None:
        return dropped_names, dropped_parents, dropped_reference, dropped_local
    return dropped_names, dropped_parents, dropped_reference, dropped_local, np.asarray(global_transforms[:, 1:], dtype=np.float32)


def mask_joint_data(
    *,
    joint_names: list[str],
    human_local_transforms: np.ndarray,
    human_global_pos: np.ndarray,
    human_global_quat: np.ndarray,
    selected_joint_names: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local = np.asarray(human_local_transforms, dtype=np.float32).copy()
    global_pos = np.asarray(human_global_pos, dtype=np.float32).copy()
    global_quat = np.asarray(human_global_quat, dtype=np.float32).copy()
    selected_mask = np.asarray([joint_name in selected_joint_names for joint_name in joint_names], dtype=bool)
    local[:, ~selected_mask] = 0.0
    global_pos[:, ~selected_mask] = 0.0
    global_quat[:, ~selected_mask] = 0.0
    return local, global_pos, global_quat


def build_human_export_payload(
    *,
    fps: float,
    joint_names: list[str],
    parent_indices: np.ndarray,
    reference_local_transforms: np.ndarray,
    human_local_transforms: np.ndarray,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    human_local_transforms = np.asarray(human_local_transforms, dtype=np.float32)
    global_positions, global_rotations = compute_global_joint_transforms(
        human_local_transforms,
        np.asarray(parent_indices, dtype=np.int32),
    )
    human_global_pos, human_global_quat = apply_visualization_frame(global_positions, global_rotations)

    payload = {
        "fps": np.asarray(int(round(float(fps))), dtype=np.int32),
        "num_frames": np.asarray(human_local_transforms.shape[0], dtype=np.int32),
        "scalar_first": np.asarray(False),
        "human_joint_names": _make_text_array(list(joint_names)),
        "human_parent_indices": np.asarray(parent_indices, dtype=np.int32),
        "human_up_axis": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        "human_forward_axis": np.asarray([0.0, -1.0, 0.0], dtype=np.float32),
        "human_reference_local_transforms": np.asarray(reference_local_transforms, dtype=np.float32),
        "human_local_transforms": human_local_transforms,
        "human_global_pos": human_global_pos,
        "human_global_quat": human_global_quat,
    }
    for key, value in (extra_payload or {}).items():
        if isinstance(value, list) and value and isinstance(value[0], str):
            payload[key] = _make_text_array(list(value))
        elif isinstance(value, np.ndarray) and value.dtype == object:
            if value.ndim == 1 and all(isinstance(item, str) for item in value.tolist()):
                payload[key] = _make_text_array(value.tolist())
            else:
                payload[key] = value
        else:
            payload[key] = np.asarray(value)
    return payload


def load_selected_joint_names(config_path: str | Path = DEFAULT_DUAL_FSQ_PATH) -> set[str]:
    import yaml

    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    feature_cfg = config["features"]
    return {str(feature_cfg["human_anchor_body"])} | {str(name) for name in feature_cfg["human_body_names"]}


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
    import torch
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
    soma_poses = matrix_to_rotvec(relative_rotations.reshape(-1, 3, 3)).reshape(relative_rotations.shape[0], relative_rotations.shape[1], 3)

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


def export_hdf5_to_soma_payload(
    hdf5_path: str | Path = DEFAULT_HDF5_PATH,
    *,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    device: str = "cuda",
    batch_size: int | None = None,
    dual_fsq_path: str | Path = DEFAULT_DUAL_FSQ_PATH,
    soma_x_root: str | Path = DEFAULT_SOMA_X_ROOT,
    smpl_model_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    selection = load_body_frame_selection(
        hdf5_path,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
    )
    motion = selection_to_smpl_body_motion(selection)
    caption = load_caption_json(hdf5_path)
    text_payload = align_caption_texts_to_frames(caption=caption, frame_timestamps=selection.frame_timestamps)
    selected_joint_names = load_selected_joint_names(dual_fsq_path)

    soma_output = run_soma_inversion(
        motion,
        device=device,
        batch_size=batch_size,
        soma_x_root=soma_x_root,
        smpl_model_path=smpl_model_path,
    )

    (
        joint_names,
        parent_indices,
        reference_local_transforms,
        human_local_transforms,
        _,
    ) = drop_soma_virtual_root(
        joint_names=soma_output["joint_names"],
        parent_indices=soma_output["parent_indices"],
        reference_local_transforms=soma_output["reference_local_transforms"],
        local_transforms=soma_output["local_transforms"],
        global_transforms=soma_output["world_transforms"][None] if soma_output["world_transforms"].ndim == 2 else soma_output["world_transforms"],
    )

    missing_joint_names = sorted(selected_joint_names.difference(joint_names))
    if missing_joint_names:
        raise ValueError(f"Selected joints missing from SOMA skeleton: {missing_joint_names}")

    global_pos, global_quat = compute_global_joint_transforms(human_local_transforms, parent_indices)
    global_pos, global_quat = apply_visualization_frame(global_pos, global_quat)
    human_local_transforms, global_pos, global_quat = mask_joint_data(
        joint_names=joint_names,
        human_local_transforms=human_local_transforms,
        human_global_pos=global_pos,
        human_global_quat=global_quat,
        selected_joint_names=selected_joint_names,
    )

    payload = build_human_export_payload(
        fps=motion.fps,
        joint_names=joint_names,
        parent_indices=parent_indices,
        reference_local_transforms=reference_local_transforms,
        human_local_transforms=human_local_transforms,
        extra_payload={
            "human_global_pos": global_pos,
            "human_global_quat": global_quat,
            "timeline_frame_indices": selection.frame_nums.astype(np.int32),
            "smpl_global_orient": motion.global_orient.astype(np.float32),
            "smpl_body_pose": motion.body_pose.astype(np.float32),
            "smpl_transl": motion.transl.astype(np.float32),
            "smpl_betas": motion.betas.astype(np.float32),
            "soma_poses": np.asarray(soma_output["soma_poses"][:, 1:], dtype=np.float32),
            "soma_transl": np.asarray(soma_output["soma_transl"], dtype=np.float32),
            "soma_joint_orient": np.asarray(soma_output["soma_joint_orient"][1:], dtype=np.float32),
            "per_vertex_error": np.asarray(soma_output["per_vertex_error"], dtype=np.float32),
            **text_payload,
        },
    )
    payload["human_global_pos"] = global_pos
    payload["human_global_quat"] = global_quat
    return payload


def save_hdf5_soma_payload(payload: dict[str, np.ndarray], output_path: str | Path = DEFAULT_OUTPUT_PATH) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return output_path
