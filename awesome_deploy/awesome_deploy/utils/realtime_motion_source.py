"""Realtime motion-source support for Xsens ZMQ + GMR retargeting."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import zmq

from awesome_deploy import AWESOME_DIR


PROTO_MODULE_NAME = "link_states_pb2"
PROTO_CACHE_DIR = Path(tempfile.gettempdir()) / "xsens_proto_runtime"
PROTO_FILE = Path(AWESOME_DIR).resolve().parents[1] / "xsens_mvn_refactor_build" / "proto" / "link_states.proto"
MIMIC_BASELINE_ROOT = Path(AWESOME_DIR).resolve().parents[1]
GMR_ROOT = MIMIC_BASELINE_ROOT / "GMR"


@dataclass
class RealtimeMotionFrame:
    """One robot-reference frame in MotionLoader-compatible layout."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray


class _RealtimeArrayView:
    """Array-like view mapping global motion indices to the live ring buffer."""

    def __init__(self, source: "RealtimeMotionSource", key: str, tail_shape: tuple[int, ...]):
        self._source = source
        self._key = key
        self._tail_shape = tail_shape

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._source.time_step_total,) + self._tail_shape

    def __len__(self) -> int:
        return self._source.time_step_total

    def __getitem__(self, index):
        return self._source._take(self._key, index)


class RealtimeMotionSource:
    """Ring-buffer-backed motion source updated once per policy step."""

    is_realtime = True
    joint_order_space = "mujoco"
    body_order_space = "policy"

    def __init__(
        self,
        frame_provider: Callable[[], RealtimeMotionFrame | None],
        buffer_size: int,
        bootstrap_timeout_sec: float = 5.0,
    ) -> None:
        self._frame_provider = frame_provider
        self._buffer_size = max(1, int(buffer_size))
        self._buffers: dict[str, deque[np.ndarray]] = {}
        self._latest_index = 0
        self._start_index = 0
        initial_frame = self._bootstrap_initial_frame(bootstrap_timeout_sec)
        self._append_initial_frame(initial_frame)

    @property
    def fps(self) -> np.ndarray:
        return np.asarray([0.0], dtype=np.float32)

    @property
    def time_step_total(self) -> int:
        return self._latest_index + 1

    def advance(self) -> None:
        """Advances the source by one policy step, repeating the last frame if idle."""
        frame = self._frame_provider()
        if frame is None:
            frame = self._snapshot_latest_frame()
        self._append_frame(frame)

    def get_latest_xsens_human_frame(self):
        """Returns the latest parsed Xsens human-frame dict when available."""
        getter = getattr(self._frame_provider, "get_latest_human_frame", None)
        if getter is None:
            return None
        return getter()

    def _bootstrap_initial_frame(self, timeout_sec: float) -> RealtimeMotionFrame:
        deadline = time.monotonic() + max(timeout_sec, 0.0)
        while True:
            frame = self._frame_provider()
            if frame is not None:
                return frame
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for the first realtime motion frame."
                )
            time.sleep(0.01)

    def _snapshot_latest_frame(self) -> RealtimeMotionFrame:
        return RealtimeMotionFrame(
            joint_pos=np.copy(self.joint_pos[self._latest_index]),
            joint_vel=np.copy(self.joint_vel[self._latest_index]),
            body_pos_w=np.copy(self.body_pos_w[self._latest_index]),
            body_quat_w=np.copy(self.body_quat_w[self._latest_index]),
            body_lin_vel_w=np.copy(self.body_lin_vel_w[self._latest_index]),
            body_ang_vel_w=np.copy(self.body_ang_vel_w[self._latest_index]),
        )

    def _append_initial_frame(self, frame: RealtimeMotionFrame) -> None:
        arrays = self._frame_to_arrays(frame)
        for key, value in arrays.items():
            self._buffers[key] = deque([value], maxlen=self._buffer_size)
        self.joint_pos = _RealtimeArrayView(self, "joint_pos", arrays["joint_pos"].shape)
        self.joint_vel = _RealtimeArrayView(self, "joint_vel", arrays["joint_vel"].shape)
        self.body_pos_w = _RealtimeArrayView(self, "body_pos_w", arrays["body_pos_w"].shape)
        self.body_quat_w = _RealtimeArrayView(self, "body_quat_w", arrays["body_quat_w"].shape)
        self.body_lin_vel_w = _RealtimeArrayView(self, "body_lin_vel_w", arrays["body_lin_vel_w"].shape)
        self.body_ang_vel_w = _RealtimeArrayView(self, "body_ang_vel_w", arrays["body_ang_vel_w"].shape)

    def _append_frame(self, frame: RealtimeMotionFrame) -> None:
        arrays = self._frame_to_arrays(frame)
        if len(next(iter(self._buffers.values()))) == self._buffer_size:
            self._start_index += 1
        for key, value in arrays.items():
            self._buffers[key].append(value)
        self._latest_index += 1

    def _frame_to_arrays(self, frame: RealtimeMotionFrame) -> dict[str, np.ndarray]:
        return {
            "joint_pos": np.asarray(frame.joint_pos, dtype=np.float32),
            "joint_vel": np.asarray(frame.joint_vel, dtype=np.float32),
            "body_pos_w": np.asarray(frame.body_pos_w, dtype=np.float32),
            "body_quat_w": np.asarray(frame.body_quat_w, dtype=np.float32),
            "body_lin_vel_w": np.asarray(frame.body_lin_vel_w, dtype=np.float32),
            "body_ang_vel_w": np.asarray(frame.body_ang_vel_w, dtype=np.float32),
        }

    def _take(self, key: str, index):
        if isinstance(index, tuple):
            if not index:
                return self._take(key, slice(None))
            first_index = index[0]
            base = self._take(key, first_index)
            if len(index) == 1:
                return base
            trailing_index = index[1:]
            base_array = np.asarray(base, dtype=np.float32)
            if isinstance(first_index, (int, np.integer)):
                return base_array[trailing_index]
            return base_array[(slice(None),) + trailing_index]
        if isinstance(index, slice):
            start, stop, step = index.indices(self.time_step_total)
            indices = np.arange(start, stop, step, dtype=np.int64)
            return self._take(key, indices)
        if isinstance(index, (list, np.ndarray)):
            indices = np.asarray(index, dtype=np.int64)
            stacked = [self._take_one(key, int(item)) for item in indices.reshape(-1)]
            return np.stack(stacked, axis=0).reshape(indices.shape + self._buffers[key][-1].shape)
        return self._take_one(key, int(index))

    def _take_one(self, key: str, index: int) -> np.ndarray:
        clamped = min(max(index, self._start_index), self._latest_index)
        offset = clamped - self._start_index
        return np.asarray(self._buffers[key][offset], dtype=np.float32)


def build_xsens_online_frame(message) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Converts protobuf LinkStateArray into the dict consumed by GMR."""
    frame: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for state in message.states:
        pos = np.asarray(
            [
                state.pose.position.x,
                state.pose.position.y,
                state.pose.position.z,
            ],
            dtype=np.float64,
        )
        quat_xyzw = np.asarray(
            [
                state.pose.orientation.x,
                state.pose.orientation.y,
                state.pose.orientation.z,
                state.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        norm = np.linalg.norm(quat_xyzw)
        if norm < 1e-12:
            quat_xyzw = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        else:
            quat_xyzw = quat_xyzw / norm
        frame[state.name] = (
            pos,
            np.asarray(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
                dtype=np.float64,
            ),
        )
    if "left_foot" in frame:
        frame["LeftFootMod"] = (
            np.asarray(frame["left_foot"][0], dtype=np.float64),
            np.asarray(frame["left_foot"][1], dtype=np.float64),
        )
    if "right_foot" in frame:
        frame["RightFootMod"] = (
            np.asarray(frame["right_foot"][0], dtype=np.float64),
            np.asarray(frame["right_foot"][1], dtype=np.float64),
        )
    return frame


def import_link_states_proto(module_name: str = PROTO_MODULE_NAME):
    """Imports generated protobuf code, compiling it locally on demand."""
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    try:
        return importlib.import_module(module_name)
    except ImportError:
        pass

    PROTO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    generated_path = PROTO_CACHE_DIR / f"{module_name}.py"
    if not generated_path.exists():
        if not PROTO_FILE.exists():
            raise FileNotFoundError(f"Proto file not found: {PROTO_FILE}")
        subprocess.run(
            [
                "protoc",
                f"--proto_path={PROTO_FILE.parent}",
                f"--python_out={PROTO_CACHE_DIR}",
                str(PROTO_FILE),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    if str(PROTO_CACHE_DIR) not in sys.path:
        sys.path.insert(0, str(PROTO_CACHE_DIR))
    return importlib.import_module(module_name)


def import_gmr_modules():
    """Imports local GMR modules even if the package is not installed globally."""
    if str(GMR_ROOT) not in sys.path:
        sys.path.insert(0, str(GMR_ROOT))
    from general_motion_retargeting import GeneralMotionRetargeting, ROBOT_XML_DICT
    import mujoco

    return GeneralMotionRetargeting, ROBOT_XML_DICT, mujoco


class XsensZmqSubscriber:
    """Receives the latest LinkStateArray protobuf from a ZMQ PUB endpoint."""

    def __init__(
        self,
        uri: str,
        topic: str,
        timeout_ms: int = 0,
        proto_module_name: str = PROTO_MODULE_NAME,
    ) -> None:
        self._proto_module = import_link_states_proto(proto_module_name)
        self._topic = topic
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.connect(uri)

    def poll_latest(self):
        """Returns the most recent protobuf message available, or ``None``."""
        latest_payload = None
        while True:
            try:
                parts = self._socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            if len(parts) != 2:
                continue
            topic_bytes, payload = parts
            if topic_bytes.decode("utf-8") != self._topic:
                continue
            latest_payload = payload
        if latest_payload is None:
            return None
        message = self._proto_module.LinkStateArray()
        message.ParseFromString(latest_payload)
        return message

    def close(self) -> None:
        self._socket.close(0)
        self._context.term()


class XsensRealtimeFrameProvider:
    """Builds MotionLoader-compatible frames from ZMQ Xsens link states."""

    def __init__(
        self,
        uri: str,
        topic: str,
        gmr_robot: str,
        gmr_human_height: float,
        body_names: list[str],
        sample_dt: float,
    ) -> None:
        self._subscriber = XsensZmqSubscriber(uri=uri, topic=topic)
        GeneralMotionRetargeting, robot_xml_dict, mujoco = import_gmr_modules()
        self._mujoco = mujoco
        self._retargeter = GeneralMotionRetargeting(
            src_human="xsens_bvh_online",
            tgt_robot=gmr_robot,
            actual_human_height=gmr_human_height,
        )
        self._model = mujoco.MjModel.from_xml_path(str(robot_xml_dict[gmr_robot]))
        self._data = mujoco.MjData(self._model)
        self._body_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in body_names
        ]
        self._sample_dt = float(sample_dt)
        self._prev_joint_pos: np.ndarray | None = None
        self._latest_human_frame = None

    def __call__(self) -> RealtimeMotionFrame | None:
        message = self._subscriber.poll_latest()
        if message is None:
            return None
        human_frame = build_xsens_online_frame(message)
        self._latest_human_frame = human_frame
        qpos = np.asarray(self._retargeter.retarget(human_frame), dtype=np.float64)
        self._data.qpos[: qpos.shape[0]] = qpos
        self._mujoco.mj_forward(self._model, self._data)

        joint_pos = np.asarray(qpos[7:], dtype=np.float32)
        if self._prev_joint_pos is None or self._sample_dt <= 0.0:
            joint_vel = np.zeros_like(joint_pos, dtype=np.float32)
        else:
            joint_vel = ((joint_pos - self._prev_joint_pos) / self._sample_dt).astype(np.float32)
        self._prev_joint_pos = joint_pos.copy()

        body_pos_w = np.asarray(self._data.xpos[self._body_ids, :], dtype=np.float32)
        body_quat_w = np.asarray(self._data.xquat[self._body_ids, :], dtype=np.float32)
        body_lin_vel_w = np.zeros((len(self._body_ids), 3), dtype=np.float32)
        body_ang_vel_w = np.zeros((len(self._body_ids), 3), dtype=np.float32)
        velocity = np.zeros(6, dtype=np.float64)
        for row, body_id in enumerate(self._body_ids):
            self._mujoco.mj_objectVelocity(
                self._model,
                self._data,
                self._mujoco.mjtObj.mjOBJ_BODY,
                body_id,
                velocity,
                0,
            )
            body_ang_vel_w[row, :] = velocity[:3]
            body_lin_vel_w[row, :] = velocity[3:]

        return RealtimeMotionFrame(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
        )

    def get_latest_human_frame(self):
        return self._latest_human_frame


def build_realtime_motion_source(cfg, body_names: list[str]) -> RealtimeMotionSource:
    """Constructs the concrete realtime motion source from deploy config."""
    provider = XsensRealtimeFrameProvider(
        uri=cfg.motion_source_uri,
        topic=cfg.motion_source_topic,
        gmr_robot=cfg.gmr_robot,
        gmr_human_height=cfg.gmr_human_height,
        body_names=body_names,
        sample_dt=cfg.policy_dt,
    )
    return RealtimeMotionSource(
        frame_provider=provider,
        buffer_size=cfg.motion_source_buffer_size,
        bootstrap_timeout_sec=5.0,
    )
