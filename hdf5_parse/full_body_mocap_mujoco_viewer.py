from __future__ import annotations

"""Visualize Xperience-10M full-body mocap keypoints in MuJoCo."""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import mujoco
import mujoco.viewer
import numpy as np

# Source: Ropedia/HOMIE-toolkit utils/constants_utils.py
SMPL_H_BODY_PARENT_INDICES = np.array(
    [
        -1,
        -1,
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        8,
        8,
        11,
        12,
        13,
        15,
        16,
        17,
        18,
        19,
        21,
        22,
        19,
        24,
        25,
        19,
        27,
        28,
        19,
        30,
        31,
        19,
        33,
        34,
        20,
        36,
        37,
        20,
        39,
        40,
        20,
        42,
        43,
        20,
        45,
        46,
        20,
        48,
        49,
    ],
    dtype=np.int32,
)

DEFAULT_HDF5_PATH = Path(__file__).resolve().parent / "hdf5" / "annotation.hdf5"
FULL_BODY_KEYPOINT_COUNT = 52
BODY_VISUAL_KEYPOINT_INDICES = np.arange(20, dtype=np.int32)
LEFT_HAND_KEYPOINT_INDICES = np.array([20, *range(22, 37)], dtype=np.int32)
RIGHT_HAND_KEYPOINT_INDICES = np.array([21, *range(37, 52)], dtype=np.int32)
HAND_FINGER_KEYPOINT_INDICES = np.arange(22, 52, dtype=np.int32)
SMPLH_HAND_PARENT_INDICES = np.array(
    [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14],
    dtype=np.int32,
)
BODY_JOINT_RADIUS = 0.02
HAND_JOINT_RADIUS = 0.012
BONE_WIDTH = 0.006


@dataclass
class MotionClip:
    keypoints: np.ndarray
    left_hand_joints: np.ndarray | None
    right_hand_joints: np.ndarray | None
    root_quat_wxyz: np.ndarray
    root_translation: np.ndarray
    cpf_quat_wxyz: np.ndarray | None
    cpf_translation: np.ndarray | None
    body_quats: np.ndarray | None
    left_hand_quats: np.ndarray | None
    right_hand_quats: np.ndarray | None
    contacts: np.ndarray | None
    frame_nums: np.ndarray
    fps: float
    slam_points: np.ndarray | None
    caption: str | None
    display_offset: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.keypoints.shape[0])


def expect_keypoint_shape(keypoints: np.ndarray) -> np.ndarray:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.shape[-2:] != (FULL_BODY_KEYPOINT_COUNT, 3):
        raise ValueError(
            f"Expected keypoints with shape (..., {FULL_BODY_KEYPOINT_COUNT}, 3), got {keypoints.shape}."
        )
    return keypoints


def split_pose7_qwxyz_xyz(pose7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose7 = np.asarray(pose7, dtype=np.float32)
    if pose7.shape[-1] != 7:
        raise ValueError(f"Expected a 7D pose vector, got shape {pose7.shape}.")
    return pose7[..., :4], pose7[..., 4:]


def quat_wxyz_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
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


def build_segments(
    joints: np.ndarray,
    parent_indices: np.ndarray,
    *,
    plus_one: bool,
) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float32)
    segments: list[np.ndarray] = []
    if plus_one:
        for child_offset, parent_idx in enumerate(parent_indices):
            child_idx = child_offset + 1
            if child_idx >= len(joints):
                break
            parent_joint_idx = parent_idx + 1 if parent_idx >= 0 else 0
            if parent_joint_idx >= len(joints):
                continue
            segments.append(np.stack([joints[parent_joint_idx], joints[child_idx]], axis=0))
    else:
        for child_idx, parent_idx in enumerate(parent_indices):
            if child_idx == 0 or parent_idx < 0:
                continue
            if parent_idx >= len(joints) or child_idx >= len(joints):
                continue
            segments.append(np.stack([joints[parent_idx], joints[child_idx]], axis=0))
    if not segments:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.stack(segments, axis=0)


def build_segment_index_pairs(
    parent_indices: np.ndarray,
    *,
    plus_one: bool,
    num_joints: int,
) -> np.ndarray:
    pairs: list[tuple[int, int]] = []
    if plus_one:
        for child_offset, parent_idx in enumerate(parent_indices):
            child_idx = child_offset + 1
            if child_idx >= num_joints:
                break
            parent_joint_idx = parent_idx + 1 if parent_idx >= 0 else 0
            if parent_joint_idx >= num_joints:
                continue
            pairs.append((parent_joint_idx, child_idx))
    else:
        for child_idx, parent_idx in enumerate(parent_indices):
            if child_idx == 0 or parent_idx < 0:
                continue
            if parent_idx >= num_joints or child_idx >= num_joints:
                continue
            pairs.append((parent_idx, child_idx))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(pairs, dtype=np.int32)


def build_body_bone_segments(joints: np.ndarray) -> np.ndarray:
    return build_segments(joints, SMPL_H_BODY_PARENT_INDICES, plus_one=True)


def build_body_visual_bone_segments(joints: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float32)
    index_pairs = build_segment_index_pairs(
        SMPL_H_BODY_PARENT_INDICES,
        plus_one=True,
        num_joints=len(joints),
    )
    keep_mask = ~np.isin(index_pairs[:, 1], HAND_FINGER_KEYPOINT_INDICES)
    index_pairs = index_pairs[keep_mask]
    if len(index_pairs) == 0:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return joints[index_pairs]


def build_visual_hand_bone_segments(joints: np.ndarray) -> np.ndarray:
    return build_segments(joints, SMPLH_HAND_PARENT_INDICES, plus_one=False)


def compute_fps(num_frames: int, video_length_sec: float | None) -> float:
    if video_length_sec is None or video_length_sec <= 0:
        return 20.0
    return float(num_frames) / float(video_length_sec)


def decode_optional_string(dataset: h5py.Dataset | None) -> str | None:
    if dataset is None:
        return None
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip("\x00")
    return str(value)


def summarize_caption(caption: str | None, *, max_chars: int = 120) -> str | None:
    if not caption:
        return None
    try:
        payload = json.loads(caption)
    except json.JSONDecodeError:
        return caption if len(caption) <= max_chars else f"{caption[:max_chars - 3]}..."

    if isinstance(payload, dict):
        config = payload.get("config")
        if isinstance(config, dict):
            main_task = config.get("Main Task")
            if isinstance(main_task, str) and main_task:
                return main_task

    compact = json.dumps(payload, ensure_ascii=False)
    return compact if len(compact) <= max_chars else f"{compact[:max_chars - 3]}..."


def extract_visual_hand_keypoints(keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keypoints = expect_keypoint_shape(keypoints)
    left_hand = keypoints[..., LEFT_HAND_KEYPOINT_INDICES, :]
    right_hand = keypoints[..., RIGHT_HAND_KEYPOINT_INDICES, :]
    return left_hand, right_hand


def extract_body_visual_keypoints(keypoints: np.ndarray) -> np.ndarray:
    keypoints = expect_keypoint_shape(keypoints)
    return keypoints[..., BODY_VISUAL_KEYPOINT_INDICES, :]


def frame_is_finite(array: np.ndarray | None) -> np.ndarray | None:
    if array is None:
        return None
    array = np.asarray(array)
    if array.ndim == 0:
        return None
    return np.isfinite(array).all(axis=tuple(range(1, array.ndim)))


def filter_valid_frames(*arrays: np.ndarray | None) -> tuple[np.ndarray, ...]:
    valid_mask = None
    for array in arrays:
        frame_mask = frame_is_finite(array)
        if frame_mask is None:
            continue
        valid_mask = frame_mask if valid_mask is None else (valid_mask & frame_mask)

    if valid_mask is None:
        return tuple(array for array in arrays if array is not None)
    if not np.any(valid_mask):
        raise ValueError("No valid frames remain after filtering non-finite values.")

    filtered: list[np.ndarray] = []
    for array in arrays:
        if array is None:
            continue
        filtered.append(np.asarray(array)[valid_mask])
    return tuple(filtered)


def compute_display_offset(keypoints: np.ndarray, root_translation: np.ndarray) -> np.ndarray:
    origin_xy = np.asarray(root_translation[0, :2], dtype=np.float32)
    floor_z = float(np.min(keypoints[..., 2]))
    return np.array([origin_xy[0], origin_xy[1], floor_z - 0.02], dtype=np.float32)


def offset_points(points: np.ndarray, display_offset: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float32) - np.asarray(display_offset, dtype=np.float32)


def load_optional_array(
    h5_file: h5py.File,
    path: str,
    frame_slice: slice | None = None,
    *,
    dtype: np.dtype = np.float32,
) -> np.ndarray | None:
    if path not in h5_file:
        return None
    dataset = h5_file[path]
    values = dataset[...] if frame_slice is None else dataset[frame_slice]
    return np.asarray(values, dtype=dtype)


def sample_point_cloud(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if max_points <= 0 or len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, num=max_points, dtype=np.int64)
    return points[indices]


def apply_frame_valid_mask(arrays: dict[str, np.ndarray | None]) -> dict[str, np.ndarray | None]:
    filtered = filter_valid_frames(*arrays.values())
    filtered_iter = iter(filtered)
    result: dict[str, np.ndarray | None] = {}
    for name, array in arrays.items():
        if array is None:
            result[name] = None
            continue
        result[name] = next(filtered_iter)
    return result


def load_motion_clip(
    hdf5_path: str | Path,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    stride: int = 1,
    max_slam_points: int = 0,
    drop_invalid_frames: bool = True,
) -> MotionClip:
    hdf5_path = Path(hdf5_path)
    if stride <= 0:
        raise ValueError("stride must be positive.")

    with h5py.File(hdf5_path, "r") as h5_file:
        keypoints_ds = h5_file["full_body_mocap/keypoints"]
        num_frames = int(keypoints_ds.shape[0])
        stop = num_frames if end_frame is None or end_frame < 0 else min(end_frame, num_frames)
        frame_slice = slice(start_frame, stop, stride)

        keypoints = np.asarray(keypoints_ds[frame_slice], dtype=np.float32)
        root_quat_wxyz, root_translation = split_pose7_qwxyz_xyz(
            np.asarray(h5_file["full_body_mocap/Ts_world_root"][frame_slice], dtype=np.float32)
        )

        cpf_quat_wxyz = None
        cpf_translation = None
        if "full_body_mocap/Ts_world_cpf" in h5_file:
            cpf_quat_wxyz, cpf_translation = split_pose7_qwxyz_xyz(
                np.asarray(h5_file["full_body_mocap/Ts_world_cpf"][frame_slice], dtype=np.float32)
            )

        left_hand_joints, right_hand_joints = extract_visual_hand_keypoints(keypoints)
        body_quats = load_optional_array(h5_file, "full_body_mocap/body_quats", frame_slice)
        left_hand_quats = load_optional_array(h5_file, "full_body_mocap/left_hand_quats", frame_slice)
        right_hand_quats = load_optional_array(h5_file, "full_body_mocap/right_hand_quats", frame_slice)
        contacts = load_optional_array(h5_file, "full_body_mocap/contacts", frame_slice)

        frame_nums = np.arange(start_frame, stop, stride, dtype=np.int64)
        if "full_body_mocap/frame_nums" in h5_file:
            frame_nums = np.asarray(h5_file["full_body_mocap/frame_nums"][frame_slice], dtype=np.int64)

        video_length_sec = None
        if "video/length_sec" in h5_file:
            video_length_sec = float(np.asarray(h5_file["video/length_sec"][()]).item())

        slam_points = None
        if max_slam_points > 0 and "slam/point_cloud" in h5_file:
            points = np.asarray(h5_file["slam/point_cloud"][...], dtype=np.float32)
            slam_points = sample_point_cloud(points, max_slam_points)

        caption = decode_optional_string(h5_file.get("caption"))

    if drop_invalid_frames:
        filtered_arrays = apply_frame_valid_mask(
            {
                "keypoints": keypoints,
                "left_hand_joints": left_hand_joints,
                "right_hand_joints": right_hand_joints,
                "root_quat_wxyz": root_quat_wxyz,
                "root_translation": root_translation,
                "cpf_quat_wxyz": cpf_quat_wxyz,
                "cpf_translation": cpf_translation,
                "body_quats": body_quats,
                "left_hand_quats": left_hand_quats,
                "right_hand_quats": right_hand_quats,
                "contacts": contacts,
                "frame_nums": frame_nums,
            }
        )
        keypoints = filtered_arrays["keypoints"]
        left_hand_joints = filtered_arrays["left_hand_joints"]
        right_hand_joints = filtered_arrays["right_hand_joints"]
        root_quat_wxyz = filtered_arrays["root_quat_wxyz"]
        root_translation = filtered_arrays["root_translation"]
        cpf_quat_wxyz = filtered_arrays["cpf_quat_wxyz"]
        cpf_translation = filtered_arrays["cpf_translation"]
        body_quats = filtered_arrays["body_quats"]
        left_hand_quats = filtered_arrays["left_hand_quats"]
        right_hand_quats = filtered_arrays["right_hand_quats"]
        contacts = filtered_arrays["contacts"]
        frame_nums = filtered_arrays["frame_nums"]

    fps = compute_fps(num_frames=num_frames, video_length_sec=video_length_sec)
    display_offset = compute_display_offset(keypoints, root_translation)
    return MotionClip(
        keypoints=keypoints,
        left_hand_joints=left_hand_joints,
        right_hand_joints=right_hand_joints,
        root_quat_wxyz=root_quat_wxyz,
        root_translation=root_translation,
        cpf_quat_wxyz=cpf_quat_wxyz,
        cpf_translation=cpf_translation,
        body_quats=body_quats,
        left_hand_quats=left_hand_quats,
        right_hand_quats=right_hand_quats,
        contacts=contacts,
        frame_nums=frame_nums,
        fps=fps,
        slam_points=slam_points,
        caption=caption,
        display_offset=display_offset,
    )


def make_viewer_model() -> mujoco.MjModel:
    xml = """
    <mujoco model="xperience_full_body_viewer">
      <option timestep="0.01" gravity="0 0 0"/>
      <visual>
        <global offwidth="1600" offheight="900"/>
      </visual>
      <worldbody>
        <light pos="0 0 4" dir="0 0 -1" diffuse="1 1 1"/>
        <light pos="2 -2 3" dir="-1 1 -1" diffuse="0.7 0.7 0.7"/>
        <geom name="floor" type="plane" size="5 5 0.1" rgba="0.16 0.18 0.20 1"/>
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


def draw_line(
    scene,
    start: np.ndarray,
    end: np.ndarray,
    width: float,
    rgba: np.ndarray,
) -> bool:
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


def draw_axes(
    scene,
    position: np.ndarray,
    quat_wxyz: np.ndarray,
    *,
    axis_length: float,
    axis_width: float,
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


def populate_scene(
    scene,
    clip: MotionClip,
    frame_idx: int,
    *,
    show_hands: bool,
    show_root_frame: bool,
    show_slam_points: bool,
) -> None:
    scene.ngeom = 0

    body_full_joints = offset_points(clip.keypoints[frame_idx], clip.display_offset)
    body_joints = extract_body_visual_keypoints(body_full_joints)
    body_segments = build_body_visual_bone_segments(body_full_joints)
    body_joint_rgba = np.array([0.98, 0.80, 0.18, 0.95], dtype=np.float32)
    body_bone_rgba = np.array([0.25, 0.92, 1.00, 0.72], dtype=np.float32)

    for segment in body_segments:
        if not draw_line(scene, segment[0], segment[1], BONE_WIDTH, body_bone_rgba):
            return
    for joint in body_joints:
        if not draw_sphere(scene, joint, BODY_JOINT_RADIUS, body_joint_rgba):
            return

    if show_hands and clip.left_hand_joints is not None and clip.right_hand_joints is not None:
        hand_joint_rgba = np.array([1.0, 0.45, 0.1, 0.9], dtype=np.float32)
        hand_bone_rgba = np.array([1.0, 0.65, 0.25, 0.7], dtype=np.float32)
        for hand_joints in (clip.left_hand_joints[frame_idx], clip.right_hand_joints[frame_idx]):
            hand_joints = offset_points(hand_joints, clip.display_offset)
            for segment in build_visual_hand_bone_segments(hand_joints):
                if not draw_line(scene, segment[0], segment[1], BONE_WIDTH * 0.75, hand_bone_rgba):
                    return
            for joint in hand_joints:
                if not draw_sphere(scene, joint, HAND_JOINT_RADIUS, hand_joint_rgba):
                    return

    if show_root_frame:
        draw_axes(
            scene,
            offset_points(clip.root_translation[frame_idx], clip.display_offset),
            clip.root_quat_wxyz[frame_idx],
            axis_length=0.18,
            axis_width=BONE_WIDTH * 0.8,
        )

    if show_slam_points and clip.slam_points is not None:
        slam_rgba = np.array([0.65, 0.72, 0.80, 0.45], dtype=np.float32)
        for point in offset_points(clip.slam_points, clip.display_offset):
            if not draw_sphere(scene, point, BODY_JOINT_RADIUS * 0.35, slam_rgba):
                return


def set_default_camera(viewer, lookat: np.ndarray) -> None:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.fixedcamid = -1
    viewer.cam.lookat[:] = np.asarray(lookat, dtype=np.float64)
    viewer.cam.distance = 2.4
    viewer.cam.azimuth = 140.0
    viewer.cam.elevation = -18.0


def print_summary(clip: MotionClip, hdf5_path: Path) -> None:
    print(f"HDF5: {hdf5_path}")
    print(f"Frames: {clip.num_frames}  fps: {clip.fps:.3f}")
    print(f"Body keypoints: {clip.keypoints.shape}")
    if clip.left_hand_joints is not None:
        print(f"Left hand keypoints: {clip.left_hand_joints.shape}")
    if clip.right_hand_joints is not None:
        print(f"Right hand keypoints: {clip.right_hand_joints.shape}")
    if clip.body_quats is not None:
        print(f"Body quats: {clip.body_quats.shape}")
    if clip.contacts is not None:
        print(f"Contacts: {clip.contacts.shape}")
    caption_summary = summarize_caption(clip.caption)
    if caption_summary:
        print(f"Caption: {caption_summary}")


def play_clip(
    clip: MotionClip,
    *,
    loop: bool,
    show_hands: bool,
    show_root_frame: bool,
    show_slam_points: bool,
) -> None:
    model = make_viewer_model()
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame_idx = 0
        set_default_camera(viewer, offset_points(clip.root_translation[0], clip.display_offset))

        while viewer.is_running():
            mujoco.mj_forward(model, data)
            # set_default_camera(viewer, clip.root_translation[frame_idx])
            populate_scene(
                viewer.user_scn,
                clip,
                frame_idx,
                show_hands=show_hands,
                show_root_frame=show_root_frame,
                show_slam_points=show_slam_points,
            )
            viewer.sync()
            time.sleep(1.0 / max(clip.fps, 1e-6))

            frame_idx += 1
            if frame_idx >= clip.num_frames:
                if not loop:
                    break
                frame_idx = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Xperience-10M full_body_mocap keypoints as a skeleton in MuJoCo.",
        epilog=(
            "Example: python hdf5_parse/full_body_mocap_mujoco_viewer.py "
            "--hands --root-frame --slam-points 300"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = parser.add_argument_group("input")
    io_group.add_argument(
        "--hdf5-path",
        type=Path,
        default=DEFAULT_HDF5_PATH,
        help="Path to annotation.hdf5.",
    )

    playback_group = parser.add_argument_group("playback")
    playback_group.add_argument("--start", type=int, default=0, help="Inclusive start frame.")
    playback_group.add_argument("--end", type=int, default=-1, help="Exclusive end frame, -1 means full clip.")
    playback_group.add_argument("--stride", type=int, default=1, help="Sample every Nth frame.")
    playback_group.add_argument("--loop", action="store_true", help="Loop playback.")

    overlay_group = parser.add_argument_group("overlays")
    overlay_group.add_argument(
        "--hands",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw hand branches extracted from full_body_mocap/keypoints.",
    )
    overlay_group.add_argument(
        "--root-frame",
        action="store_true",
        help="Draw the root coordinate frame axes.",
    )
    overlay_group.add_argument(
        "--slam-points",
        type=int,
        default=0,
        metavar="N",
        help="Draw up to N decimated SLAM points. Use 0 to disable.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    clip = load_motion_clip(
        args.hdf5_path,
        start_frame=args.start,
        end_frame=args.end,
        stride=args.stride,
        max_slam_points=max(args.slam_points, 0),
    )
    print_summary(clip, args.hdf5_path)
    play_clip(
        clip,
        loop=args.loop,
        show_hands=args.hands,
        show_root_frame=args.root_frame,
        show_slam_points=args.slam_points > 0,
    )


if __name__ == "__main__":
    main()
