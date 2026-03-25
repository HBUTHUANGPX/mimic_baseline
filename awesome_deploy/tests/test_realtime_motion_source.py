"""Tests for the realtime motion source ring-buffer semantics."""

import time
from types import SimpleNamespace

import numpy as np
import zmq

from awesome_deploy.utils.realtime_motion_source import (
    RealtimeMotionFrame,
    RealtimeMotionSource,
    build_xsens_online_frame,
    XsensRealtimeFrameProvider,
    import_link_states_proto,
    XsensZmqSubscriber,
)


def _frame(seed: float) -> RealtimeMotionFrame:
    return RealtimeMotionFrame(
        joint_pos=np.asarray([seed, seed + 1.0], dtype=np.float32),
        joint_vel=np.asarray([seed + 2.0, seed + 3.0], dtype=np.float32),
        body_pos_w=np.asarray([[seed, 0.0, 0.0]], dtype=np.float32),
        body_quat_w=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        body_lin_vel_w=np.asarray([[0.0, seed, 0.0]], dtype=np.float32),
        body_ang_vel_w=np.asarray([[0.0, 0.0, seed]], dtype=np.float32),
    )


def test_realtime_motion_source_repeats_last_frame_when_provider_stalls():
    """Realtime source should reuse the previous frame when no update arrives."""
    sequence = iter([_frame(0.0), _frame(1.0), None])

    source = RealtimeMotionSource(
        frame_provider=lambda: next(sequence),
        buffer_size=3,
        bootstrap_timeout_sec=0.0,
    )

    source.advance()
    source.advance()

    assert source.time_step_total == 3
    assert np.allclose(source.joint_pos[0], np.asarray([0.0, 1.0], dtype=np.float32))
    assert np.allclose(source.joint_pos[1], np.asarray([1.0, 2.0], dtype=np.float32))
    assert np.allclose(source.joint_pos[2], np.asarray([1.0, 2.0], dtype=np.float32))


def test_realtime_motion_source_clamps_requests_older_than_ring_buffer():
    """Requests older than the retained window should clamp to the oldest sample."""
    sequence = iter([_frame(0.0), _frame(1.0), _frame(2.0), _frame(3.0)])

    source = RealtimeMotionSource(
        frame_provider=lambda: next(sequence),
        buffer_size=2,
        bootstrap_timeout_sec=0.0,
    )

    source.advance()
    source.advance()
    source.advance()

    assert source.time_step_total == 4
    assert np.allclose(source.joint_pos[0], source.joint_pos[2])
    assert np.allclose(source.joint_pos[1], source.joint_pos[2])
    assert np.allclose(source.joint_pos[3], np.asarray([3.0, 4.0], dtype=np.float32))


def test_realtime_motion_source_supports_numpy_style_tuple_indexing():
    """Realtime array views should behave like ndarrays for tuple indexing."""
    sequence = iter([_frame(0.0), _frame(1.0)])

    source = RealtimeMotionSource(
        frame_provider=lambda: next(sequence),
        buffer_size=3,
        bootstrap_timeout_sec=0.0,
    )

    source.advance()

    quat_value = source.body_quat_w[1, 0, :]
    x_pos_column = source.body_pos_w[:, 0, 0]

    assert np.allclose(quat_value, np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(x_pos_column, np.asarray([0.0, 1.0], dtype=np.float32))


def test_build_xsens_online_frame_keeps_hands_and_builds_modified_feet():
    """Xsens conversion should preserve hand links and synthesize foot aliases."""
    def state(name, pos, quat):
        return SimpleNamespace(
            name=name,
            pose=SimpleNamespace(
                position=SimpleNamespace(x=pos[0], y=pos[1], z=pos[2]),
                orientation=SimpleNamespace(
                    x=quat[0], y=quat[1], z=quat[2], w=quat[3]
                ),
            ),
        )

    message = SimpleNamespace(
        states=[
            state("left_foot", [1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]),
            state("right_foot", [4.0, 5.0, 6.0], [0.0, 0.0, 0.0, 1.0]),
            state("left_hand", [7.0, 8.0, 9.0], [0.0, 0.0, 0.0, 1.0]),
            state("right_hand", [10.0, 11.0, 12.0], [0.0, 0.0, 0.0, 1.0]),
        ]
    )

    frame = build_xsens_online_frame(message)

    assert "left_hand" in frame
    assert "right_hand" in frame
    assert "LeftFootMod" in frame
    assert "RightFootMod" in frame
    assert np.allclose(frame["LeftFootMod"][0], np.asarray([1.0, 2.0, 3.0]))
    assert np.allclose(frame["RightFootMod"][0], np.asarray([4.0, 5.0, 6.0]))
    assert np.allclose(
        frame["left_hand"][1], np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )


def test_xsens_zmq_subscriber_receives_latest_message_over_loopback():
    """Subscriber should decode the newest protobuf frame published over ZMQ."""
    proto = import_link_states_proto()
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    topic = "xsens.link_states.v1"
    subscriber = XsensZmqSubscriber(
        uri=f"tcp://127.0.0.1:{port}",
        topic=topic,
        proto_module_name=proto.__name__,
    )

    time.sleep(0.1)

    first = proto.LinkStateArray()
    first.header.frame_id = "frame_1"
    first_state = first.states.add()
    first_state.name = "pelvis"
    first_state.pose.position.x = 1.0

    second = proto.LinkStateArray()
    second.header.frame_id = "frame_2"
    second_state = second.states.add()
    second_state.name = "pelvis"
    second_state.pose.position.x = 2.0

    publisher.send_multipart([topic.encode("utf-8"), first.SerializeToString()])
    publisher.send_multipart([topic.encode("utf-8"), second.SerializeToString()])
    time.sleep(0.1)

    message = subscriber.poll_latest()

    assert message is not None
    assert message.header.frame_id == "frame_2"
    assert message.states[0].pose.position.x == 2.0

    subscriber.close()
    publisher.close(0)
    context.term()


def test_xsens_realtime_frame_provider_returns_motionloader_compatible_frame(monkeypatch):
    """Realtime provider should convert one ZMQ protobuf frame into robot reference arrays."""
    proto = import_link_states_proto()
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    topic = "xsens.link_states.v1"

    class FakeRetargeter:
        def __init__(self, src_human, tgt_robot, actual_human_height):
            self.src_human = src_human
            self.tgt_robot = tgt_robot
            self.actual_human_height = actual_human_height

        def retarget(self, human_frame):
            assert "LeftFootMod" in human_frame
            assert "RightFootMod" in human_frame
            return np.asarray([0.0] * 7 + [0.1, 0.2, 0.3], dtype=np.float64)

    class FakeModel:
        def __init__(self):
            self.body_name_to_id = {"pelvis_link": 0, "torso_link": 1}

        @classmethod
        def from_xml_path(cls, path):
            return cls()

    class FakeData:
        def __init__(self, model):
            self.qpos = np.zeros(10, dtype=np.float64)
            self.xpos = np.zeros((2, 3), dtype=np.float64)
            self.xquat = np.zeros((2, 4), dtype=np.float64)

    class FakeMujoco:
        class mjtObj:
            mjOBJ_BODY = 1

        MjModel = FakeModel
        MjData = FakeData

        @staticmethod
        def mj_name2id(model, obj_type, name):
            return model.body_name_to_id[name]

        @staticmethod
        def mj_forward(model, data):
            data.xpos[0, :] = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
            data.xpos[1, :] = np.asarray([4.0, 5.0, 6.0], dtype=np.float64)
            data.xquat[0, :] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            data.xquat[1, :] = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

        @staticmethod
        def mj_objectVelocity(model, data, obj_type, obj_id, res, flg_local):
            if obj_id == 0:
                res[:] = np.asarray([0.1, 0.2, 0.3, 1.0, 2.0, 3.0], dtype=np.float64)
            else:
                res[:] = np.asarray([0.4, 0.5, 0.6, 4.0, 5.0, 6.0], dtype=np.float64)

    monkeypatch.setattr(
        "awesome_deploy.utils.realtime_motion_source.import_gmr_modules",
        lambda: (FakeRetargeter, {"Q1": "/tmp/fake_q1.xml"}, FakeMujoco),
    )

    provider = XsensRealtimeFrameProvider(
        uri=f"tcp://127.0.0.1:{port}",
        topic=topic,
        gmr_robot="Q1",
        gmr_human_height=1.66,
        body_names=["pelvis_link", "torso_link"],
        sample_dt=0.02,
    )

    time.sleep(0.1)
    message = proto.LinkStateArray()
    message.header.frame_id = "frame_1"
    for name, x in (("left_foot", 1.0), ("right_foot", 2.0), ("pelvis", 3.0)):
        state = message.states.add()
        state.name = name
        state.pose.position.x = x
        state.pose.position.y = x + 1.0
        state.pose.position.z = x + 2.0
        state.pose.orientation.w = 1.0

    publisher.send_multipart([topic.encode("utf-8"), message.SerializeToString()])
    time.sleep(0.1)

    frame = provider()

    assert frame is not None
    assert np.allclose(frame.joint_pos, np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
    assert np.allclose(frame.joint_vel, np.zeros(3, dtype=np.float32))
    assert np.allclose(
        frame.body_pos_w,
        np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
    )
    assert np.allclose(
        frame.body_lin_vel_w,
        np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
    )
    assert np.allclose(
        frame.body_ang_vel_w,
        np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
    )

    provider._subscriber.close()
    publisher.close(0)
    context.term()
