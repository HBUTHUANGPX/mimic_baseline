"""Shared MuJoCo overlay helpers for raw Xsens link frame visualization."""

from __future__ import annotations

import numpy as np


def quaternion_wxyz_to_matrix(w, x, y, z):
    """Converts a WXYZ quaternion into a 3x3 rotation matrix."""
    quat = np.array([w, x, y, z], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)

    w, x, y, z = quat / norm
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def init_arrow_geom(mujoco_module, geom, rgba, label, show_labels):
    mujoco_module.mjv_initGeom(
        geom,
        type=mujoco_module.mjtGeom.mjGEOM_ARROW,
        size=np.zeros(3, dtype=np.float64),
        pos=np.zeros(3, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).reshape(-1),
        rgba=np.array(rgba, dtype=np.float32),
    )
    geom.label = label if show_labels else ""


def draw_link_frames(
    mujoco_module,
    viewer,
    human_frame,
    axis_length,
    shaft_width,
    show_labels,
    clear_existing=True,
):
    """Draws all parsed xsens human-frame entries into ``viewer.user_scn``."""
    if clear_existing:
        viewer.user_scn.ngeom = 0
    if human_frame is None:
        return

    axis_colors = (
        np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
    )

    for link_name, (origin, quat_wxyz) in human_frame.items():
        origin = np.asarray(origin, dtype=np.float64)
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
        rotation = quaternion_wxyz_to_matrix(
            quat_wxyz[0],
            quat_wxyz[1],
            quat_wxyz[2],
            quat_wxyz[3],
        )

        for axis_index in range(3):
            if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                return

            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            init_arrow_geom(
                mujoco_module,
                geom,
                axis_colors[axis_index],
                link_name,
                show_labels,
            )
            endpoint = origin + axis_length * rotation[:, axis_index]
            mujoco_module.mjv_connector(
                geom,
                type=mujoco_module.mjtGeom.mjGEOM_ARROW,
                width=shaft_width,
                from_=origin,
                to=endpoint,
            )
            viewer.user_scn.ngeom += 1
