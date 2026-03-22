"""Quaternion and rotation helpers used by the deployment runtime."""

import numpy as np


def matrix_from_quat(quaternions: np.ndarray) -> np.ndarray:
    """Converts quaternions into rotation matrices.

    Args:
        quaternions: Quaternion array in ``(w, x, y, z)`` order with shape
            ``(..., 4)``.

    Returns:
        Rotation matrices with shape ``(..., 3, 3)``.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L41-L70
    """
    r, i, j, k = np.moveaxis(quaternions, -1, 0)
    two_s = 2.0 / np.sum(quaternions * quaternions, axis=-1)
    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalizes an array to unit length along the last dimension.

    Args:
        x: Input array of shape ``(..., dims)``.
        eps: Lower bound applied to the norm to avoid division by zero.

    Returns:
        Normalized array with the same shape as ``x``.
    """
    norms = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    norms = np.clip(norms, eps, None)
    return x / norms


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Computes the conjugate of a quaternion.

    Args:
        q: Quaternion array in ``(w, x, y, z)`` order with shape ``(..., 4)``.

    Returns:
        Conjugated quaternion array with the same shape as ``q``.
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return np.concatenate((q[:, 0:1], -q[:, 1:]), axis=-1).reshape(shape)


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Computes the inverse of a quaternion.

    Args:
        q: Quaternion array in ``(w, x, y, z)`` order with shape ``(..., 4)``.

    Returns:
        Inverted quaternion array with the same shape as ``q``.
    """
    return normalize(quat_conjugate(q))


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiplies two quaternion arrays elementwise.

    Args:
        q1: First quaternion array in ``(w, x, y, z)`` order.
        q2: Second quaternion array in ``(w, x, y, z)`` order.

    Returns:
        Hamilton product of ``q1`` and ``q2`` with the same shape.

    Raises:
        ValueError: If the two input arrays do not have identical shapes.
    """
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # Flatten to a batch so the closed-form Hamilton product can be applied to
    # arbitrary leading dimensions.
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.stack([w, x, y, z], axis=-1).reshape(shape)
