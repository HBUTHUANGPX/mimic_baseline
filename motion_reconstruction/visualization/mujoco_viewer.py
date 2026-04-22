"""使用 MuJoCo 播放原始数据和重构数据。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from motion_reconstruction.evaluation import ReconstructionResult
from motion_reconstruction.evaluation.robot_state import human_skeleton_edges, robot_feature_to_qpos


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

    def body_positions(self, qpos: np.ndarray) -> np.ndarray:
        mujoco = _import_mujoco()
        if qpos.shape[-1] != self.model.nq:
            raise ValueError(f"qpos 维度不匹配: got {qpos.shape[-1]}, model.nq={self.model.nq}。")
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return np.asarray(self.data.xpos, dtype=np.float32).copy()

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
    if pair not in {"robot", "human", "both"}:
        raise ValueError("pair 必须是 robot、human 或 both。")
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
        human_positions = result.human_body_pos_w[frame_index]
        human_center = _human_center(human_positions, result.human_body_names, result.human_anchor_body)
        recon_positions, recon_center = _robot_positions(
            robot,
            result.recon_from_human_feature[frame_index],
            result.robot_anchor_pos_w[frame_index],
        )
        _draw_human_robot_pair(
            scene,
            robot,
            result,
            human_positions,
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
    result: ReconstructionResult,
    human_positions: np.ndarray,
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
        human_skeleton_edges(result.human_body_names),
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
    qpos = robot_feature_to_qpos(feature, anchor_pos_w=anchor_pos_w, expected_nq=robot.model.nq)
    positions = robot.body_positions(qpos)
    return positions, robot.center_from(positions)


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
    return RobotKinematics(
        model=model,
        data=data,
        body_edges=edges,
        anchor_body_id=anchor_id if anchor_id >= 0 else None,
    )


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
