#!/usr/bin/env python3
import argparse
import importlib
import os
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import zmq


DEFAULT_XML_PATH = Path(
    "/home/hpx/HPX_LOCO_2/mimic_baseline/general_motion_tracker_whole_body_teleoperation/"
    "general_motion_tracker_whole_body_teleoperation/assets/Q1/mjcf/Q1_wo_hand.xml"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Subscribe xsens ZMQ link_states and draw all link frames in MuJoCo."
    )
    parser.add_argument(
        "--connect",
        default="tcp://127.0.0.1:5555",
        help="ZMQ publisher address, default: tcp://127.0.0.1:5555",
    )
    parser.add_argument(
        "--topic",
        default="xsens.link_states.v1",
        help="Subscription topic, default: xsens.link_states.v1",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=1000,
        help="ZMQ receive timeout in milliseconds, default: 100",
    )
    parser.add_argument(
        "--proto-module",
        default="link_states_pb2",
        help="Generated Python protobuf module name, default: link_states_pb2",
    )
    parser.add_argument(
        "--xml-path",
        default=str(DEFAULT_XML_PATH),
        help="MuJoCo XML path used as background scene",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.08,
        help="Axis arrow length for each link frame",
    )
    parser.add_argument(
        "--shaft-width",
        type=float,
        default=0.006,
        help="Arrow shaft width for each link frame",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Show link names on rendered frame axes",
    )
    return parser.parse_args()


def quaternion_xyzw_to_matrix(x, y, z, w):
    quat = np.array([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)

    x, y, z, w = quat / norm
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
            [2.0 * (xy + wz), 1.0 - 2.0 * (yy + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (yy + zz)],
        ],
        dtype=np.float64,
    )


def import_proto_module(module_name):
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        print(f"failed_to_import_proto_module={module_name} error={exc}", file=sys.stderr)
        raise


def create_viewer(xml_path):
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    viewer = mujoco.viewer.launch_passive(model, data)
    return model, data, viewer


def init_arrow_geom(geom, rgba, label, show_labels):
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.zeros(3, dtype=np.float64),
        pos=np.zeros(3, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).reshape(-1),
        rgba=np.array(rgba, dtype=np.float32),
    )
    geom.label = label if show_labels else ""


def draw_link_frames(viewer, message, axis_length, shaft_width, show_labels):
    viewer.user_scn.ngeom = 0
    axis_colors = (
        np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
    )

    for state in message.states:
        origin = np.array(
            [
                state.pose.position.x,
                state.pose.position.y,
                state.pose.position.z,
            ],
            dtype=np.float64,
        )
        rotation = quaternion_xyzw_to_matrix(
            state.pose.orientation.x,
            state.pose.orientation.y,
            state.pose.orientation.z,
            state.pose.orientation.w,
        )

        # print(state.name)
        for axis_index in range(3):
            if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                return

            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            init_arrow_geom(geom, axis_colors[axis_index], state.name, show_labels)
            endpoint = origin + axis_length * rotation[:, axis_index]
            mujoco.mjv_connector(
                geom,
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                width=shaft_width,
                from_=origin,
                to=endpoint,
            )
            viewer.user_scn.ngeom += 1


def main():
    args = parse_args()
    xml_path = Path(args.xml_path).expanduser().resolve()
    if not xml_path.exists():
        print(f"xml_path_not_found={xml_path}", file=sys.stderr)
        return 1

    proto_module = import_proto_module(args.proto_module)

    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    subscriber.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    subscriber.connect(args.connect)

    model, data, viewer = create_viewer(xml_path)
    # del model
    # del data

    print(f"subscriber_connect={args.connect}")
    print(f"subscriber_topic={args.topic}")
    print(f"viewer_xml_path={xml_path}")
    import time
    try:
        while viewer.is_running():
            try:
                topic = subscriber.recv_string()
                if topic != args.topic:
                    print("topic != args.topic")
                    continue
                # print("=============================")
                # print("start",time.time())
                payload = subscriber.recv()
                # print("payload",time.time())
                # print("recv")
                message = proto_module.LinkStateArray()
                # print("message",time.time())
                message.ParseFromString(payload)
                # print("ParseFromString",time.time())
                draw_link_frames(
                    viewer,
                    message,
                    args.axis_length,
                    args.shaft_width,
                    args.show_labels,
                )
                # print("draw_link_frames",time.time())
                viewer.sync()
                # print("sync",time.time())
            except zmq.error.Again:
                # print("zmq.error.Again")
                viewer.sync()
                continue

            

            
    finally:
        subscriber.close(0)
        context.term()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
