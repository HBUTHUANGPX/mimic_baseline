"""Shared human-skeleton pose helpers aligned with soma-retargeter players."""

from __future__ import annotations

import numpy as np


VISUALIZATION_FRAME_QUAT_XYZW = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32)


def quat_mul_batch_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
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


def quat_conjugate_batch_xyzw(quat: np.ndarray) -> np.ndarray:
    result = np.array(quat, dtype=np.float32, copy=True)
    result[..., :3] *= -1.0
    return result


def quat_rotate_batch_xyzw(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    vec = np.asarray(vec, dtype=np.float32)
    q_xyz = quat[..., :3]
    qw = quat[..., 3:4]
    uv = np.cross(q_xyz, vec)
    uuv = np.cross(q_xyz, uv)
    return (vec + 2.0 * (qw * uv + uuv)).astype(np.float32, copy=False)


def compute_global_joint_transforms_xyzw(
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
        global_positions[:, joint_idx] = parent_pos + quat_rotate_batch_xyzw(
            parent_rot,
            local_positions[:, joint_idx],
        )
        global_rotations[:, joint_idx] = quat_mul_batch_xyzw(
            parent_rot,
            local_rotations[:, joint_idx],
        )

    return global_positions, global_rotations


def apply_visualization_frame_xyzw(
    positions: np.ndarray,
    rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotations = np.asarray(rotations, dtype=np.float32)
    expanded = np.broadcast_to(VISUALIZATION_FRAME_QUAT_XYZW, rotations.shape)
    corrected_positions = quat_rotate_batch_xyzw(expanded, positions)
    corrected_rotations = quat_mul_batch_xyzw(
        quat_mul_batch_xyzw(expanded, rotations),
        quat_conjugate_batch_xyzw(expanded),
    )
    return corrected_positions.astype(np.float32, copy=False), corrected_rotations.astype(np.float32, copy=False)


def compute_visualized_global_joint_transforms_xyzw(
    local_transforms: np.ndarray,
    parent_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions, rotations = compute_global_joint_transforms_xyzw(local_transforms, parent_indices)
    return apply_visualization_frame_xyzw(positions, rotations)


def convert_root_to_pre_visualization_frame_xyzw(local_transforms: np.ndarray) -> np.ndarray:
    local_transforms = np.asarray(local_transforms, dtype=np.float32).copy()
    inverse_frame = quat_conjugate_batch_xyzw(VISUALIZATION_FRAME_QUAT_XYZW)
    local_transforms[:, 0, :3] = quat_rotate_batch_xyzw(inverse_frame, local_transforms[:, 0, :3])
    local_transforms[:, 0, 3:7] = quat_mul_batch_xyzw(
        np.broadcast_to(inverse_frame, local_transforms[:, 0, 3:7].shape),
        local_transforms[:, 0, 3:7],
    )
    return local_transforms
