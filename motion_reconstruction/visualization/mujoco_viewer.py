"""使用 MuJoCo 播放原始数据和重构数据。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from motion_reconstruction.evaluation import ReconstructionResult
from motion_reconstruction.evaluation.robot_state import (
    human_skeleton_edges,
    robot_feature_to_qpos,
    rot6d_to_quat_wxyz_numpy,
)


EMPTY_SCENE_XML = """
<mujoco model="motion_reconstruction_viewer">
  <visual>
    <global offwidth="1920" offheight="1080"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".18 .18 .18" rgb2=".23 .23 .23"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".1"/>
  </asset>
  <worldbody>
    <light name="main" pos="0 -3 5" dir="0 1 -1"/>
    <geom name="floor" type="plane" size="20 20 .1" material="grid"/>
  </worldbody>
</mujoco>
""".strip()


@dataclass
class RobotKinematics:
    """用 MuJoCo XML 做机器人正运动学。"""

    model: object
    data: object
    body_edges: list[tuple[int, int]]
    anchor_body_id: int | None
    root_body_id: int | None
    free_joint_id: int | None
    free_qposadr: int | None
    joint_qposadrs: list[int]

    def body_positions(self, qpos: np.ndarray) -> np.ndarray:
        mujoco = _import_mujoco()
        if qpos.shape[-1] != self.model.nq:
            raise ValueError(f"qpos 维度不匹配: got {qpos.shape[-1]}, model.nq={self.model.nq}。")
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return np.asarray(self.data.xpos, dtype=np.float32).copy()

    def qpos_from_anchor_feature(self, feature: np.ndarray, anchor_pos_w: np.ndarray) -> np.ndarray:
        feature = np.asarray(feature, dtype=np.float32)
        joint_pos = feature[6:]
        if self.model.nq == joint_pos.shape[-1]:
            return robot_feature_to_qpos(feature, anchor_pos_w=anchor_pos_w, expected_nq=self.model.nq)
        if self.model.nq != joint_pos.shape[-1] + 7:
            return robot_feature_to_qpos(feature, anchor_pos_w=anchor_pos_w, expected_nq=self.model.nq)
        if self.anchor_body_id is None or self.root_body_id is None or self.free_qposadr is None:
            return robot_feature_to_qpos(feature, anchor_pos_w=anchor_pos_w, expected_nq=self.model.nq)
        if len(self.joint_qposadrs) != joint_pos.shape[-1]:
            raise ValueError(
                "MuJoCo 非 free joint 数量与 robot feature 不匹配: "
                f"feature_joint_count={joint_pos.shape[-1]}, mujoco_joint_count={len(self.joint_qposadrs)}。"
            )

        qpos = np.zeros(self.model.nq, dtype=np.float32)
        qpos[self.free_qposadr : self.free_qposadr + 3] = 0.0
        qpos[self.free_qposadr + 3 : self.free_qposadr + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        for value, qposadr in zip(joint_pos, self.joint_qposadrs):
            qpos[qposadr] = value

        mujoco = _import_mujoco()
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        root_pos = np.asarray(self.data.xpos[self.root_body_id], dtype=np.float32).copy()
        root_quat = np.asarray(self.data.xquat[self.root_body_id], dtype=np.float32).copy()
        anchor_pos = np.asarray(self.data.xpos[self.anchor_body_id], dtype=np.float32).copy()
        anchor_quat = np.asarray(self.data.xquat[self.anchor_body_id], dtype=np.float32).copy()

        root_inv = _quat_inverse(root_quat)
        local_anchor_pos = _quat_rotate(root_inv, anchor_pos - root_pos)
        local_anchor_quat = _quat_multiply(root_inv, anchor_quat)

        desired_anchor_quat = rot6d_to_quat_wxyz_numpy(feature[:6])
        desired_root_quat = _quat_multiply(desired_anchor_quat, _quat_inverse(local_anchor_quat))
        desired_root_pos = np.asarray(anchor_pos_w, dtype=np.float32) - _quat_rotate(
            desired_root_quat,
            local_anchor_pos,
        )
        qpos[self.free_qposadr : self.free_qposadr + 3] = desired_root_pos
        qpos[self.free_qposadr + 3 : self.free_qposadr + 7] = desired_root_quat
        return qpos

    def center_from(self, positions: np.ndarray) -> np.ndarray:
        if self.anchor_body_id is not None:
            return positions[self.anchor_body_id]
        return positions[1:].mean(axis=0)


def play_reconstruction(
    *,
    result: ReconstructionResult,
    xml_path: str | Path,
    pair: str = "both",
    loop: bool = False,
    fps: int | None = None,
    keep_world: bool = False,
) -> None:
    """打开 MuJoCo viewer 播放重构对比。"""
    validate_reconstruction_for_pair(result=result, pair=pair)
    mujoco = _import_mujoco()
    viewer_module = _import_mujoco_viewer()
    robot = _load_robot_kinematics(xml_path, result.robot_anchor_body)
    scene_model = mujoco.MjModel.from_xml_string(EMPTY_SCENE_XML)
    scene_data = mujoco.MjData(scene_model)
    playback_fps = int(fps or result.fps or 30)
    frame_dt = 1.0 / max(playback_fps, 1)

    with viewer_module.launch_passive(scene_model, scene_data) as viewer:
        _set_camera(viewer)
        while viewer.is_running():
            for frame_index in range(result.center_indices.shape[0]):
                if not viewer.is_running():
                    break
                start_time = time.perf_counter()
                viewer.user_scn.ngeom = 0
                _draw_frame(
                    scene=viewer.user_scn,
                    robot=robot,
                    result=result,
                    frame_index=frame_index,
                    pair=pair,
                    keep_world=keep_world,
                )
                viewer.sync()
                sleep_time = frame_dt - (time.perf_counter() - start_time)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
            if not loop:
                break


def validate_reconstruction_for_pair(*, result: ReconstructionResult, pair: str) -> None:
    if pair not in {"robot", "human", "both"}:
        raise ValueError("pair 必须是 robot、human 或 both。")
    if pair == "robot" and (result.original_robot_feature is None or result.recon_from_robot_feature is None):
        raise ValueError("当前结果不包含 robot 原始/重构分支，无法使用 pair=robot。")
    if pair == "human" and result.recon_from_human_feature is None:
        raise ValueError("当前结果不包含 human->decoder 重构分支，无法使用 pair=human。")
    if pair == "both":
        if result.original_robot_feature is None or result.recon_from_robot_feature is None:
            raise ValueError("当前结果不包含 robot 原始/重构分支，无法使用 pair=both。")
        if result.recon_from_human_feature is None:
            raise ValueError("当前结果不包含 human->decoder 重构分支，无法使用 pair=both。")


def _draw_frame(
    *,
    scene,
    robot: RobotKinematics,
    result: ReconstructionResult,
    frame_index: int,
    pair: str,
    keep_world: bool,
) -> None:
    if pair in {"robot", "both"}:
        y_offset = 0.85 if pair == "both" else 0.0
        original_positions, original_center = _robot_positions(
            robot,
            result.original_robot_feature[frame_index],
            result.robot_anchor_pos_w[frame_index],
        )
        recon_positions, _ = _robot_positions(
            robot,
            result.recon_from_robot_feature[frame_index],
            result.robot_anchor_pos_w[frame_index],
        )
        _draw_robot_pair(
            scene,
            robot,
            original_positions,
            recon_positions,
            center=original_center,
            y_offset=y_offset,
            keep_world=keep_world,
        )

    if pair in {"human", "both"}:
        y_offset = -0.85 if pair == "both" else 0.0
        human_positions, human_body_names = _select_human_positions(result, frame_index)
        human_center = _human_center(human_positions, human_body_names, result.human_anchor_body)
        recon_positions, recon_center = _robot_positions(
            robot,
            result.recon_from_human_feature[frame_index],
            result.robot_anchor_pos_w[frame_index],
        )
        _draw_human_robot_pair(
            scene,
            robot,
            human_positions,
            human_body_names,
            recon_positions,
            human_center=human_center,
            robot_center=recon_center,
            y_offset=y_offset,
            keep_world=keep_world,
        )


def _draw_robot_pair(
    scene,
    robot: RobotKinematics,
    original_positions: np.ndarray,
    recon_positions: np.ndarray,
    *,
    center: np.ndarray,
    y_offset: float,
    keep_world: bool,
) -> None:
    original_offset = np.array([-0.9, y_offset, 0.0], dtype=np.float32)
    recon_offset = np.array([0.9, y_offset, 0.0], dtype=np.float32)
    base_center = np.zeros(3, dtype=np.float32) if keep_world else center
    _draw_skeleton(
        scene,
        original_positions - base_center + original_offset,
        robot.body_edges,
        (0.15, 0.55, 1.0, 1.0),
        skip_first_point=True,
    )
    _draw_skeleton(
        scene,
        recon_positions - base_center + recon_offset,
        robot.body_edges,
        (1.0, 0.35, 0.15, 1.0),
        skip_first_point=True,
    )


def _draw_human_robot_pair(
    scene,
    robot: RobotKinematics,
    human_positions: np.ndarray,
    human_body_names: list[str],
    recon_positions: np.ndarray,
    *,
    human_center: np.ndarray,
    robot_center: np.ndarray,
    y_offset: float,
    keep_world: bool,
) -> None:
    human_offset = np.array([-0.9, y_offset, 0.0], dtype=np.float32)
    robot_offset = np.array([0.9, y_offset, 0.0], dtype=np.float32)
    human_base = np.zeros(3, dtype=np.float32) if keep_world else human_center
    robot_base = np.zeros(3, dtype=np.float32) if keep_world else robot_center
    _draw_skeleton(
        scene,
        human_positions - human_base + human_offset,
        human_skeleton_edges(human_body_names),
        (0.15, 0.95, 0.65, 1.0),
        point_radius=0.035,
        line_radius=0.012,
    )
    _draw_skeleton(
        scene,
        recon_positions - robot_base + robot_offset,
        robot.body_edges,
        (1.0, 0.55, 0.1, 1.0),
        skip_first_point=True,
    )


def _robot_positions(
    robot: RobotKinematics,
    feature: np.ndarray,
    anchor_pos_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    qpos = robot.qpos_from_anchor_feature(feature, anchor_pos_w)
    positions = robot.body_positions(qpos)
    return positions, robot.center_from(positions)


def _select_human_positions(result: ReconstructionResult, frame_index: int) -> tuple[np.ndarray, list[str]]:
    positions = result.human_body_pos_w[frame_index]
    source_names = list(result.human_body_names)
    display_names = result.display_human_body_names or source_names
    if display_names == source_names:
        return positions, source_names

    name_to_index = {name: index for index, name in enumerate(source_names)}
    missing_names = [name for name in display_names if name not in name_to_index]
    if missing_names:
        raise ValueError(f"可视化人体 body 不存在: {missing_names}。可用名字: {source_names}")

    selected_indices = np.asarray([name_to_index[name] for name in display_names], dtype=np.int64)
    return positions[selected_indices], list(display_names)


def _draw_skeleton(
    scene,
    positions: np.ndarray,
    edges: list[tuple[int, int]],
    rgba: tuple[float, float, float, float],
    *,
    point_radius: float = 0.04,
    line_radius: float = 0.018,
    skip_first_point: bool = False,
) -> None:
    for parent, child in edges:
        _draw_line(scene, positions[parent], positions[child], rgba, line_radius)
    point_positions = positions[1:] if skip_first_point and positions.shape[0] > 1 else positions
    for pos in point_positions:
        _draw_sphere(scene, pos, rgba, point_radius)


def _draw_sphere(scene, pos: np.ndarray, rgba: tuple[float, float, float, float], radius: float) -> None:
    if not _can_draw(scene, pos):
        return
    mujoco = _import_mujoco()
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _draw_line(
    scene,
    start: np.ndarray,
    end: np.ndarray,
    rgba: tuple[float, float, float, float],
    radius: float,
) -> None:
    if not _can_draw(scene, start) or not _can_draw(scene, end):
        return
    mujoco = _import_mujoco()
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        float(radius),
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    scene.ngeom += 1


def _can_draw(scene, pos: np.ndarray) -> bool:
    return scene.ngeom < scene.maxgeom and bool(np.all(np.isfinite(pos)))


def _human_center(positions: np.ndarray, names: list[str], anchor_name: str) -> np.ndarray:
    if anchor_name in names:
        return positions[names.index(anchor_name)]
    return positions.mean(axis=0)


def _load_robot_kinematics(xml_path: str | Path, anchor_body: str) -> RobotKinematics:
    mujoco = _import_mujoco()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    edges: list[tuple[int, int]] = []
    for body_id in range(1, model.nbody):
        parent_id = int(model.body_parentid[body_id])
        if parent_id > 0:
            edges.append((parent_id, body_id))
    anchor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, anchor_body)
    free_joint_ids = [
        joint_id
        for joint_id in range(model.njnt)
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if len(free_joint_ids) > 1:
        raise ValueError("当前可视化只支持最多一个 free joint 的机器人 XML。")
    free_joint_id = free_joint_ids[0] if free_joint_ids else None
    root_body_id = int(model.jnt_bodyid[free_joint_id]) if free_joint_id is not None else _model_root_body_id(model)
    free_qposadr = int(model.jnt_qposadr[free_joint_id]) if free_joint_id is not None else None
    joint_qposadrs = _single_dof_joint_qposadrs(model, mujoco, free_joint_id)
    return RobotKinematics(
        model=model,
        data=data,
        body_edges=edges,
        anchor_body_id=anchor_id if anchor_id >= 0 else None,
        root_body_id=root_body_id,
        free_joint_id=free_joint_id,
        free_qposadr=free_qposadr,
        joint_qposadrs=joint_qposadrs,
    )


def _model_root_body_id(model) -> int | None:
    for body_id in range(1, model.nbody):
        if int(model.body_parentid[body_id]) == 0:
            return body_id
    return None


def _single_dof_joint_qposadrs(model, mujoco, free_joint_id: int | None) -> list[int]:
    qposadrs: list[int] = []
    for joint_id in range(model.njnt):
        if joint_id == free_joint_id:
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            raise ValueError("当前可视化只支持 hinge/slide robot joints，不支持 ball joint。")
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        qposadrs.append(int(model.jnt_qposadr[joint_id]))
    return qposadrs


def _quat_inverse(quat: np.ndarray) -> np.ndarray:
    normalized = _quat_normalize(quat)
    return np.array(
        [normalized[0], -normalized[1], -normalized[2], -normalized[3]],
        dtype=np.float32,
    )


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = _quat_normalize(left)
    right = _quat_normalize(right)
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return _quat_normalize(
        np.array(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ],
            dtype=np.float32,
        )
    )


def _quat_rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quat = _quat_normalize(quat)
    vector = np.asarray(vector, dtype=np.float32)
    qvec = quat[1:]
    uv = np.cross(qvec, vector)
    uuv = np.cross(qvec, uv)
    return vector + 2.0 * (quat[0] * uv + uuv)


def _quat_normalize(quat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    return quat / max(float(np.linalg.norm(quat)), eps)


def _set_camera(viewer) -> None:
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -18
    viewer.cam.distance = 4.5
    viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.9])


def _import_mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise ImportError("缺少 mujoco 依赖，请先运行: python -m pip install mujoco") from exc
    return mujoco


def _import_mujoco_viewer():
    try:
        import mujoco.viewer
    except ImportError as exc:
        raise ImportError("缺少 mujoco viewer 依赖，请先运行: python -m pip install mujoco") from exc
    return mujoco.viewer
