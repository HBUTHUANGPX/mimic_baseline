import types

import numpy as np

from awesome_deploy.utils.xsens_link_frame_viz import (
    draw_link_frames,
    quaternion_wxyz_to_matrix,
)


def test_quaternion_wxyz_to_matrix_identity():
    rotation = quaternion_wxyz_to_matrix(1.0, 0.0, 0.0, 0.0)

    assert np.allclose(rotation, np.eye(3, dtype=np.float64))


def test_quaternion_wxyz_to_matrix_matches_known_xyzw_reference():
    quat_xyzw = np.asarray([0.1, 0.2, 0.3, 0.9], dtype=np.float64)
    quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw)
    x, y, z, w = quat_xyzw
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    expected = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )

    actual = quaternion_wxyz_to_matrix(w, x, y, z)

    assert np.allclose(actual, expected)
    assert np.allclose(actual.T @ actual, np.eye(3, dtype=np.float64))


def test_draw_link_frames_populates_three_axes_per_human_frame_entry():
    calls = []

    class FakeMujoco:
        class mjtGeom:
            mjGEOM_ARROW = 1

        @staticmethod
        def mjv_initGeom(geom, type, size, pos, mat, rgba):
            geom.type = type
            geom.size = size
            geom.pos = pos
            geom.mat = mat
            geom.rgba = rgba

        @staticmethod
        def mjv_connector(geom, type, width, from_, to):
            calls.append(
                {
                    "geom": geom,
                    "type": type,
                    "width": width,
                    "from": np.asarray(from_, dtype=np.float64),
                    "to": np.asarray(to, dtype=np.float64),
                }
            )

    class Geom:
        def __init__(self):
            self.label = ""

    geoms = [Geom() for _ in range(6)]
    viewer = types.SimpleNamespace(
        user_scn=types.SimpleNamespace(ngeom=0, maxgeom=len(geoms), geoms=geoms)
    )
    human_frame = {
        "pelvis": (
            np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        ),
        "torso": (
            np.asarray([4.0, 5.0, 6.0], dtype=np.float64),
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        ),
    }

    draw_link_frames(
        mujoco_module=FakeMujoco,
        viewer=viewer,
        human_frame=human_frame,
        axis_length=0.1,
        shaft_width=0.01,
        show_labels=False,
    )

    assert viewer.user_scn.ngeom == 6
    assert len(calls) == 6
    assert np.allclose(calls[0]["from"], np.asarray([1.0, 2.0, 3.0], dtype=np.float64))
    assert np.allclose(calls[0]["to"], np.asarray([1.1, 2.0, 3.0], dtype=np.float64))
    assert all(geom.label == "" for geom in geoms)
