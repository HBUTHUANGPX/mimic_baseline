"""机器人和人体可视化需要的状态转换。"""

from __future__ import annotations

import numpy as np


HUMAN_SKELETON_CHAINS = (
    ("Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd"),
    ("Hips", "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd"),
    ("Hips", "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd"),
    ("Chest", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"),
    ("Chest", "RightShoulder", "RightArm", "RightForeArm", "RightHand"),
)


def rot6d_to_quat_wxyz_numpy(rot6d: np.ndarray) -> np.ndarray:
    """将 6D rotation 转成 MuJoCo 使用的 wxyz quaternion。"""
    matrix = rot6d_to_matrix_numpy(rot6d)
    return matrix_to_quat_wxyz_numpy(matrix)


def rot6d_to_matrix_numpy(rot6d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """按 6D rotation 定义还原旋转矩阵。"""
    values = np.asarray(rot6d, dtype=np.float32)
    if values.shape[-1] != 6:
        raise ValueError(f"rot6d final dim must be 6, got {values.shape}.")
    first = values[..., 0:3]
    second = values[..., 3:6]
    basis0 = _normalize(first, eps)
    second_orthogonal = second - np.sum(basis0 * second, axis=-1, keepdims=True) * basis0
    basis1 = _normalize(second_orthogonal, eps)
    basis2 = np.cross(basis0, basis1)
    return np.stack((basis0, basis1, basis2), axis=-1)


def matrix_to_quat_wxyz_numpy(matrix: np.ndarray) -> np.ndarray:
    """将旋转矩阵转成 wxyz quaternion。"""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix final dims must be 3x3, got {matrix.shape}.")

    flat = matrix.reshape(-1, 3, 3)
    quats = np.empty((flat.shape[0], 4), dtype=np.float32)
    for index, rot in enumerate(flat):
        trace = float(np.trace(rot))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            quats[index, 0] = 0.25 * scale
            quats[index, 1] = (rot[2, 1] - rot[1, 2]) / scale
            quats[index, 2] = (rot[0, 2] - rot[2, 0]) / scale
            quats[index, 3] = (rot[1, 0] - rot[0, 1]) / scale
        elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            scale = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            quats[index, 0] = (rot[2, 1] - rot[1, 2]) / scale
            quats[index, 1] = 0.25 * scale
            quats[index, 2] = (rot[0, 1] + rot[1, 0]) / scale
            quats[index, 3] = (rot[0, 2] + rot[2, 0]) / scale
        elif rot[1, 1] > rot[2, 2]:
            scale = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            quats[index, 0] = (rot[0, 2] - rot[2, 0]) / scale
            quats[index, 1] = (rot[0, 1] + rot[1, 0]) / scale
            quats[index, 2] = 0.25 * scale
            quats[index, 3] = (rot[1, 2] + rot[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            quats[index, 0] = (rot[1, 0] - rot[0, 1]) / scale
            quats[index, 1] = (rot[0, 2] + rot[2, 0]) / scale
            quats[index, 2] = (rot[1, 2] + rot[2, 1]) / scale
            quats[index, 3] = 0.25 * scale

    quats = _normalize(quats)
    return quats.reshape(matrix.shape[:-2] + (4,))


def robot_feature_to_qpos(feature: np.ndarray, *, anchor_pos_w: np.ndarray, expected_nq: int) -> np.ndarray:
    """将 robot feature 转成 MuJoCo qpos。"""
    feature = np.asarray(feature, dtype=np.float32)
    anchor_pos_w = np.asarray(anchor_pos_w, dtype=np.float32)
    if feature.shape[-1] < 6:
        raise ValueError(f"robot feature dim must be at least 6, got {feature.shape}.")
    if anchor_pos_w.shape[-1] != 3:
        raise ValueError(f"anchor_pos_w final dim must be 3, got {anchor_pos_w.shape}.")

    joint_pos = feature[..., 6:]
    joint_count = joint_pos.shape[-1]
    if expected_nq == joint_count:
        return joint_pos.copy()
    if expected_nq == joint_count + 7:
        quat = rot6d_to_quat_wxyz_numpy(feature[..., :6])
        return np.concatenate((anchor_pos_w, quat, joint_pos), axis=-1)
    raise ValueError(
        "MuJoCo model nq 与 robot feature 不匹配: "
        f"expected_nq={expected_nq}, joint_count={joint_count}。"
    )


def human_skeleton_edges(names: list[str]) -> list[tuple[int, int]]:
    """根据已存在的名字生成人体骨架连线。"""
    name_to_index = {name: index for index, name in enumerate(names)}
    edges: list[tuple[int, int]] = []
    for chain in HUMAN_SKELETON_CHAINS:
        for parent_name, child_name in zip(chain[:-1], chain[1:]):
            if parent_name in name_to_index and child_name in name_to_index:
                edges.append((name_to_index[parent_name], name_to_index[child_name]))
    return edges


def _normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, eps)
