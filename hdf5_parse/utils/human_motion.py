from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class HumanMotionNPZ:
    local_transforms: np.ndarray
    parent_indices: np.ndarray
    joint_names: list[str]
    fps: float
    scalar_first: bool
    global_positions: np.ndarray
    global_rotations: np.ndarray


def quat_mul_batch(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.moveaxis(np.asarray(q1, dtype=np.float32), -1, 0)
    x2, y2, z2, w2 = np.moveaxis(np.asarray(q2, dtype=np.float32), -1, 0)
    return np.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def quat_rotate_batch(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    vec = np.asarray(vec, dtype=np.float32)
    q_xyz = quat[..., :3]
    qw = quat[..., 3:4]
    uv = np.cross(q_xyz, vec)
    uuv = np.cross(q_xyz, uv)
    return (vec + 2.0 * (qw * uv + uuv)).astype(np.float32, copy=False)


def quat_conjugate_batch(quat: np.ndarray) -> np.ndarray:
    result = np.array(quat, dtype=np.float32, copy=True)
    result[..., :3] *= -1.0
    return result


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quat, dtype=np.float32)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def compute_global_joint_transforms(
    local_transforms: np.ndarray,
    parent_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    local_transforms = np.asarray(local_transforms, dtype=np.float32)
    parent_indices = np.asarray(parent_indices, dtype=np.int32)
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
        global_positions[:, joint_idx] = parent_pos + quat_rotate_batch(
            parent_rot,
            local_positions[:, joint_idx],
        )
        global_rotations[:, joint_idx] = quat_mul_batch(
            parent_rot,
            local_rotations[:, joint_idx],
        )

    return global_positions, global_rotations


def apply_visualization_frame(
    positions: np.ndarray,
    rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_up_to_z_up = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32)
    expanded = np.broadcast_to(y_up_to_z_up, np.asarray(rotations).shape)
    corrected_positions = quat_rotate_batch(expanded, positions)
    corrected_rotations = quat_mul_batch(
        quat_mul_batch(expanded, rotations),
        quat_conjugate_batch(expanded),
    )
    return corrected_positions.astype(np.float32, copy=False), corrected_rotations.astype(np.float32, copy=False)


def compute_visualized_global_transforms(
    local_transforms: np.ndarray,
    parent_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions, rotations = compute_global_joint_transforms(local_transforms, parent_indices)
    return apply_visualization_frame(positions, rotations)


def load_human_motion_npz(npz_path: str | Path) -> HumanMotionNPZ:
    with np.load(npz_path, allow_pickle=False) as payload:
        scalar_first = bool(payload["scalar_first"].item()) if "scalar_first" in payload.files else False
        if scalar_first:
            raise ValueError("human motion npz 的 human quaternion 期望是 XYZW，当前工具不支持 scalar_first=True。")
        local_transforms = np.asarray(payload["human_local_transforms"], dtype=np.float32)
        parent_indices = np.asarray(payload["human_parent_indices"], dtype=np.int32)
        joint_names = [str(name) for name in payload["human_joint_names"].tolist()]
        fps = float(np.asarray(payload["fps"]).item())

    global_positions, global_rotations = compute_visualized_global_transforms(local_transforms, parent_indices)
    return HumanMotionNPZ(
        local_transforms=local_transforms,
        parent_indices=parent_indices,
        joint_names=joint_names,
        fps=fps,
        scalar_first=scalar_first,
        global_positions=global_positions,
        global_rotations=global_rotations,
    )
