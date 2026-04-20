from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
from scipy.spatial.transform import Rotation


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import smplx
from smplx.body_models import Struct

from smpl_motion_tools import (
    DEFAULT_HDF5_PATH,
    SMPLMotionClip,
    convert_smplh_motion_clip_to_smpl,
    load_smplh_motion_clip,
    resolve_body_model_path,
)


JOINT_RADIUS = 0.015
BONE_WIDTH = 0.007
VERTEX_RADIUS = 0.006


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize SMPL-H or converted SMPL motion reconstructed from annotation.hdf5."
    )
    parser.add_argument("--hdf5-path", type=Path, default=DEFAULT_HDF5_PATH)
    parser.add_argument("--model-type", choices=("smplh", "smpl"), default="smplh")
    parser.add_argument("--smplh-model-path", type=Path, default=None)
    parser.add_argument("--smpl-model-path", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--root-frame", action="store_true")
    parser.add_argument("--mesh-points", type=int, default=400)
    return parser.parse_args(argv)


def sample_vertex_indices(num_vertices: int, max_points: int) -> np.ndarray:
    if num_vertices <= 0 or max_points <= 0:
        return np.zeros((0,), dtype=np.int32)
    count = min(int(num_vertices), int(max_points))
    return np.linspace(0, num_vertices - 1, num=count, dtype=np.int32)


def load_motion_clip_for_viewer(
    hdf5_path: str | Path,
    *,
    model_type: str,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
) -> SMPLMotionClip:
    clip = load_smplh_motion_clip(
        hdf5_path,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
        drop_invalid_frames=True,
    )
    if str(model_type).lower() == "smpl":
        return convert_smplh_motion_clip_to_smpl(clip)
    return clip


def resolve_active_model_path(args: argparse.Namespace) -> Path:
    explicit_path = args.smplh_model_path if args.model_type == "smplh" else args.smpl_model_path
    return resolve_body_model_path(args.model_type, explicit_path)


def instantiate_body_model(args: argparse.Namespace, clip: SMPLMotionClip):
    model_path = resolve_active_model_path(args)
    ext = model_path.suffix.lstrip(".")
    num_betas = 10 if clip.betas is None else max(1, min(clip.betas.shape[-1], 16))

    if args.model_type == "smpl" and ext == "npz":
        model_data = np.load(model_path, allow_pickle=True)
        data_struct = Struct(**{key: model_data[key] for key in model_data.files})
        return smplx.SMPL(
            str(model_path),
            data_struct=data_struct,
            gender="neutral",
            num_betas=num_betas,
            batch_size=1,
        )

    create_kwargs: dict[str, object] = {
        "model_path": str(model_path),
        "model_type": args.model_type,
        "gender": "neutral",
        "ext": ext,
        "num_betas": num_betas,
        "batch_size": 1,
    }
    if args.model_type == "smplh":
        create_kwargs["use_pca"] = False
        create_kwargs["flat_hand_mean"] = True
    return smplx.create(**create_kwargs)


def make_viewer_model() -> mujoco.MjModel:
    xml = """
    <mujoco model="smpl_body_viewer">
      <option timestep="0.01" gravity="0 0 0"/>
      <visual>
        <global offwidth="1600" offheight="900"/>
      </visual>
      <worldbody>
        <light pos="0 0 4" dir="0 0 -1" diffuse="1 1 1"/>
        <light pos="2 -2 3" dir="-1 1 -1" diffuse="0.7 0.7 0.7"/>
        <geom name="floor" type="plane" size="6 6 0.1" rgba="0.16 0.18 0.20 1"/>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def draw_sphere(scene, position: np.ndarray, radius: float, rgba: np.ndarray) -> bool:
    if not np.isfinite(position).all():
        return True
    if scene.ngeom >= scene.maxgeom:
        return False
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
    return True


def draw_line(scene, start: np.ndarray, end: np.ndarray, width: float, rgba: np.ndarray) -> bool:
    if not np.isfinite(start).all() or not np.isfinite(end).all():
        return True
    if scene.ngeom >= scene.maxgeom:
        return False
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
    return True


def rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_quat()
    return quat_xyzw[[3, 0, 1, 2]].astype(np.float64)


def quat_wxyz_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_wxyz, dtype=np.float64)[[1, 2, 3, 0]]
    return Rotation.from_quat(quat_xyzw).as_matrix()


def draw_axes(
    scene,
    position: np.ndarray,
    quat_wxyz: np.ndarray,
    *,
    axis_length: float = 0.18,
    axis_width: float = BONE_WIDTH,
) -> None:
    if not np.isfinite(position).all() or not np.isfinite(quat_wxyz).all():
        return
    rot = quat_wxyz_to_mat(quat_wxyz)
    colors = (
        np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32),
        np.array([0.2, 1.0, 0.2, 1.0], dtype=np.float32),
        np.array([0.2, 0.4, 1.0, 1.0], dtype=np.float32),
    )
    for axis_idx in range(3):
        direction = rot[:, axis_idx]
        if not draw_line(scene, position, position + direction * axis_length, axis_width, colors[axis_idx]):
            return


def to_torch(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array, dtype=np.float32)).reshape(1, -1)


def prepare_betas(clip: SMPLMotionClip, frame_idx: int, num_betas: int) -> torch.Tensor | None:
    if clip.betas is None or num_betas <= 0:
        return None
    betas = np.zeros((1, num_betas), dtype=np.float32)
    source = np.asarray(clip.betas[frame_idx], dtype=np.float32)
    width = min(num_betas, source.shape[-1])
    betas[0, :width] = source[:width]
    return torch.as_tensor(betas)


def evaluate_body_frame(body_model, clip: SMPLMotionClip, frame_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kwargs: dict[str, torch.Tensor] = {
        "global_orient": to_torch(clip.global_orient[frame_idx]),
        "body_pose": to_torch(clip.body_pose[frame_idx]),
        "transl": to_torch(clip.transl[frame_idx]),
        "return_verts": True,
    }
    betas = prepare_betas(clip, frame_idx, getattr(body_model, "num_betas", 0))
    if betas is not None:
        kwargs["betas"] = betas
    if clip.model_type == "smplh":
        kwargs["left_hand_pose"] = to_torch(clip.left_hand_pose[frame_idx])
        kwargs["right_hand_pose"] = to_torch(clip.right_hand_pose[frame_idx])

    with torch.no_grad():
        output = body_model(**kwargs)

    vertices = output.vertices.detach().cpu().numpy()[0]
    joints = output.joints.detach().cpu().numpy()[0]
    parents = body_model.parents.detach().cpu().numpy().astype(np.int32)
    return vertices.astype(np.float32), joints.astype(np.float32), parents


def populate_scene(
    scene,
    *,
    vertices: np.ndarray,
    joints: np.ndarray,
    parents: np.ndarray,
    root_quat_wxyz: np.ndarray | None,
    mesh_points: int,
    show_root_frame: bool,
) -> None:
    scene.ngeom = 0

    joint_rgba = np.array([0.96, 0.92, 0.45, 0.95], dtype=np.float32)
    bone_rgba = np.array([0.18, 0.92, 1.0, 0.78], dtype=np.float32)
    mesh_rgba = np.array([1.0, 0.47, 0.12, 0.65], dtype=np.float32)

    usable_joint_count = min(int(joints.shape[0]), int(parents.shape[0]))
    for joint_idx in range(1, usable_joint_count):
        parent_idx = int(parents[joint_idx])
        if parent_idx < 0 or parent_idx >= usable_joint_count:
            continue
        if not draw_line(scene, joints[parent_idx], joints[joint_idx], BONE_WIDTH, bone_rgba):
            return

    for joint in joints[:usable_joint_count]:
        if not draw_sphere(scene, joint, JOINT_RADIUS, joint_rgba):
            return

    for vertex_idx in sample_vertex_indices(vertices.shape[0], mesh_points):
        if not draw_sphere(scene, vertices[vertex_idx], VERTEX_RADIUS, mesh_rgba):
            return

    if show_root_frame and root_quat_wxyz is not None and usable_joint_count > 0:
        draw_axes(scene, joints[0], root_quat_wxyz, axis_length=0.16, axis_width=BONE_WIDTH * 0.8)


def set_default_camera(viewer, lookat: np.ndarray) -> None:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.fixedcamid = -1
    viewer.cam.lookat[:] = np.asarray(lookat, dtype=np.float64)
    viewer.cam.distance = 2.8
    viewer.cam.azimuth = 145.0
    viewer.cam.elevation = -20.0


def print_summary(args: argparse.Namespace, clip: SMPLMotionClip, model_path: Path) -> None:
    print(f"HDF5: {Path(args.hdf5_path).resolve()}")
    print(f"Model type: {clip.model_type}")
    print(f"Body model: {model_path}")
    print(f"Frames: {clip.num_frames}  fps: {clip.fps:.3f}")
    print(f"Global orient: {clip.global_orient.shape}")
    print(f"Body pose: {clip.body_pose.shape}")
    print(f"Translation: {clip.transl.shape}")
    if clip.left_hand_pose is not None:
        print(f"Left hand pose: {clip.left_hand_pose.shape}")
    if clip.right_hand_pose is not None:
        print(f"Right hand pose: {clip.right_hand_pose.shape}")
    if clip.betas is not None:
        print(f"Betas: {clip.betas.shape}")


def run_viewer(args: argparse.Namespace) -> None:
    clip = load_motion_clip_for_viewer(
        args.hdf5_path,
        model_type=args.model_type,
        start_frame=args.start,
        end_frame=args.end,
        stride=args.stride,
    )
    if clip.num_frames == 0:
        raise RuntimeError("No valid frames remain after filtering non-finite values.")

    body_model = instantiate_body_model(args, clip)
    model_path = resolve_active_model_path(args)
    print_summary(args, clip, model_path)

    mj_model = make_viewer_model()
    mj_data = mujoco.MjData(mj_model)
    frame_idx = 0
    frame_dt = 1.0 / max(float(clip.fps), 1e-6)

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        vertices, joints, parents = evaluate_body_frame(body_model, clip, frame_idx)
        root_quat_wxyz = rotvec_to_quat_wxyz(clip.global_orient[frame_idx]) if args.root_frame else None
        populate_scene(
            viewer.user_scn,
            vertices=vertices,
            joints=joints,
            parents=parents,
            root_quat_wxyz=root_quat_wxyz,
            mesh_points=args.mesh_points,
            show_root_frame=args.root_frame,
        )
        set_default_camera(viewer, joints[0])
        viewer.sync()

        while viewer.is_running():
            start_time = time.perf_counter()

            vertices, joints, parents = evaluate_body_frame(body_model, clip, frame_idx)
            root_quat_wxyz = rotvec_to_quat_wxyz(clip.global_orient[frame_idx]) if args.root_frame else None
            populate_scene(
                viewer.user_scn,
                vertices=vertices,
                joints=joints,
                parents=parents,
                root_quat_wxyz=root_quat_wxyz,
                mesh_points=args.mesh_points,
                show_root_frame=args.root_frame,
            )
            viewer.sync()

            frame_idx += 1
            if frame_idx >= clip.num_frames:
                if args.loop:
                    frame_idx = 0
                else:
                    break

            elapsed = time.perf_counter() - start_time
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_viewer(args)


if __name__ == "__main__":
    main()
