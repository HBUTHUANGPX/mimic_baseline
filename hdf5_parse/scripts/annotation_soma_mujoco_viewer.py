from __future__ import annotations

"""Play human motion npz skeletons in MuJoCo.

This viewer intentionally follows the same human parsing pipeline as
`soma-retargeter/app/play_npz_mujoco.py`:

human_local_transforms
-> compute_global_joint_transforms
-> apply_visualization_frame
-> draw_animation_frame
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _bootstrap import ensure_repo_root_on_sys_path

ensure_repo_root_on_sys_path(__file__)

from hdf5_parse.utils.human_motion import (
    HumanMotionNPZ,
    apply_visualization_frame,
    compute_global_joint_transforms,
    compute_visualized_global_transforms,
    load_human_motion_npz,
    quat_to_mat,
)


def draw_sphere(scene, position: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0.0, 0.0], dtype=np.float64),
        pos=np.asarray(position, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).reshape(-1),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def draw_line(
    scene,
    start: np.ndarray,
    end: np.ndarray,
    width: float,
    rgba: np.ndarray,
) -> None:
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=np.zeros(3, dtype=np.float64),
        pos=np.zeros(3, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).reshape(-1),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        width=width,
        from_=np.asarray(start, dtype=np.float64),
        to=np.asarray(end, dtype=np.float64),
    )
    scene.ngeom += 1


def draw_axes(
    scene,
    position: np.ndarray,
    rotation: np.ndarray,
    axis_length: float,
    axis_width: float,
) -> None:
    rot_mat = quat_to_mat(rotation)
    colors = (
        np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
    )
    for axis_idx in range(3):
        if scene.ngeom >= scene.maxgeom:
            return
        draw_line(
            scene,
            position,
            position + rot_mat[:, axis_idx] * axis_length,
            axis_width,
            colors[axis_idx],
        )


def draw_animation_frame(
    viewer,
    positions: np.ndarray,
    rotations: np.ndarray,
    parent_indices: np.ndarray,
    *,
    show_axes: bool,
    joint_radius: float,
    bone_width: float,
) -> None:
    scene = viewer.user_scn
    scene.ngeom = 0
    joint_rgba = np.array([1.0, 0.8, 0.1, 0.9], dtype=np.float32)
    bone_rgba = np.array([0.3, 0.9, 1.0, 0.7], dtype=np.float32)

    for joint_idx, position in enumerate(np.asarray(positions, dtype=np.float32)):
        if scene.ngeom >= scene.maxgeom:
            break
        draw_sphere(scene, position, joint_radius, joint_rgba)
        parent_idx = int(parent_indices[joint_idx])
        if parent_idx >= 0 and scene.ngeom < scene.maxgeom:
            draw_line(scene, positions[parent_idx], position, bone_width, bone_rgba)
        if show_axes and scene.ngeom + 3 < scene.maxgeom:
            draw_axes(
                scene,
                position,
                rotations[joint_idx],
                axis_length=0.06,
                axis_width=0.003,
            )


def set_default_camera(viewer) -> None:
    viewer.cam.distance = 4.0
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -15.0
    viewer.cam.fixedcamid = -1
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE


def build_empty_scene_xml() -> str:
    return """
    <mujoco model="human_skeleton_viewer">
      <option gravity="0 0 -9.81" timestep="0.01"/>
      <visual>
        <headlight diffuse="0.8 0.8 0.8" ambient="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
        <rgba haze="0.15 0.25 0.35 1"/>
      </visual>
      <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.25 0.30 0.35" rgb2="0.18 0.20 0.24" width="512" height="512"/>
        <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
      </asset>
      <worldbody>
        <geom name="floor" type="plane" size="10 10 0.1" rgba="0.9 0.9 0.9 1" material="grid"/>
        <light name="key" pos="0 0 5" dir="0 0 -1" directional="true"/>
      </worldbody>
    </mujoco>
    """


def play_human_motion(
    npz_path: Path,
    *,
    loop: bool,
    show_axes: bool,
    fps_override: float | None,
) -> None:
    motion = load_human_motion_npz(npz_path)
    global_positions = motion.global_positions
    global_rotations = motion.global_rotations
    model = mujoco.MjModel.from_xml_string(build_empty_scene_xml())
    data = mujoco.MjData(model)
    fps = float(fps_override) if fps_override is not None else motion.fps
    dt = 1.0 / max(fps, 1e-6)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        set_default_camera(viewer)
        frame_idx = 0
        while viewer.is_running():
            data.qpos[:] = 0.0
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            draw_animation_frame(
                viewer,
                global_positions[frame_idx],
                global_rotations[frame_idx],
                motion.parent_indices,
                show_axes=show_axes,
                joint_radius=0.025,
                bone_width=0.008,
            )
            viewer.sync()
            time.sleep(dt)

            frame_idx += 1
            if frame_idx >= motion.local_transforms.shape[0]:
                if not loop:
                    break
                frame_idx = 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用 soma-retargeter 同语义的人体解析流程，可视化 human motion npz。"
    )
    parser.add_argument(
        "--npz",
        type=Path,
        required=True,
        help="human motion npz 路径，通常来自 SOMA BVH 经过 bvh_to_csv_converter.py 的输出。",
    )
    parser.add_argument("--loop", action="store_true", help="循环播放。")
    parser.add_argument(
        "--hide-axes",
        action="store_true",
        help="关闭每个 joint 的局部坐标轴绘制。默认会显示坐标轴。",
    )
    parser.add_argument("--fps", type=float, default=None, help="覆盖文件中的播放帧率。")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    play_human_motion(
        args.npz,
        loop=args.loop,
        show_axes=not args.hide_axes,
        fps_override=args.fps,
    )


if __name__ == "__main__":
    main()
