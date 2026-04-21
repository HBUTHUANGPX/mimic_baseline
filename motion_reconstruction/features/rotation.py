"""quaternion 与 rotation 表示工具。

本文件统一使用 wxyz quaternion。FeatureBuilder 在进入这里之前已经完成
scalar_first/xyzw 到 wxyz 的转换。
"""

from __future__ import annotations

import torch


def normalize_quat_wxyz(quat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_to_matrix_wxyz(quat: torch.Tensor) -> torch.Tensor:
    quat = normalize_quat_wxyz(quat)
    w, x, y, z = quat.unbind(dim=-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    row0 = torch.stack((ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)), dim=-1)
    row1 = torch.stack((2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)), dim=-1)
    row2 = torch.stack((2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def quat_to_rot6d_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """将 wxyz quaternion 转成 6D rotation 表示。

    6D 表示取旋转矩阵前两列，适合作为连续网络输入。
    """
    matrix = quat_to_matrix_wxyz(quat)
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def quat_conjugate_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)


def quat_rotate_wxyz(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quat = normalize_quat_wxyz(quat)
    q_vec = quat[..., 1:]
    q_w = quat[..., :1]
    uv = torch.cross(q_vec, vector, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return vector + 2.0 * (q_w * uv + uuv)


def quat_inverse_rotate_wxyz(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """将 world-frame vector 旋转到 `quat` 的 local frame。

    human joint position 特征用它把 world 位移转到 anchor body frame。
    """
    return quat_rotate_wxyz(quat_conjugate_wxyz(quat), vector)
