"""Tests for motion source configuration and factory selection."""

from pathlib import Path

import numpy as np
import pytest

from awesome_deploy.utils.cfg import Q1RobotCfg
from awesome_deploy.utils.motion_loader import MotionLoader
from awesome_deploy.utils.motion_source_factory import build_motion_source


def test_q1_robot_cfg_defaults_to_offline_motion_source():
    """Q1 config should keep offline motion as the default reference source."""
    cfg = Q1RobotCfg()

    assert cfg.motion_source == "offline"
    assert cfg.motion_source_uri == cfg.motion_file
    assert cfg.motion_source_topic == ""
    assert cfg.motion_source_buffer_size == 1
    assert cfg.gmr_robot == "Q1"
    assert cfg.gmr_human_height == pytest.approx(1.66)


def test_q1_robot_cfg_accepts_environment_overrides(monkeypatch):
    """Realtime-related config should be switchable without editing Python code."""
    monkeypatch.setenv("AWESOME_DEPLOY_MOTION_SOURCE", "realtime")
    monkeypatch.setenv("AWESOME_DEPLOY_MOTION_SOURCE_URI", "tcp://192.168.1.2:5555")
    monkeypatch.setenv("AWESOME_DEPLOY_MOTION_SOURCE_TOPIC", "xsens.link_states.v1")
    monkeypatch.setenv("AWESOME_DEPLOY_MOTION_SOURCE_BUFFER_SIZE", "11")
    monkeypatch.setenv("AWESOME_DEPLOY_GMR_ROBOT", "Q1")
    monkeypatch.setenv("AWESOME_DEPLOY_GMR_HUMAN_HEIGHT", "1.72")

    cfg = Q1RobotCfg()

    assert cfg.motion_source == "realtime"
    assert cfg.motion_source_uri == "tcp://192.168.1.2:5555"
    assert cfg.motion_source_topic == "xsens.link_states.v1"
    assert cfg.motion_source_buffer_size == 11
    assert cfg.gmr_robot == "Q1"
    assert cfg.gmr_human_height == pytest.approx(1.72)


def test_q1_robot_cfg_uses_sensible_realtime_defaults(monkeypatch):
    """Switching only the source mode should still produce a runnable realtime config."""
    monkeypatch.setenv("AWESOME_DEPLOY_MOTION_SOURCE", "realtime")

    cfg = Q1RobotCfg()

    assert cfg.motion_source == "realtime"
    assert cfg.motion_source_uri == "tcp://127.0.0.1:5555"
    assert cfg.motion_source_topic == "xsens.link_states.v1"
    assert cfg.motion_source_buffer_size >= 11


def test_build_motion_source_returns_motion_loader_for_offline_mode(tmp_path: Path):
    """Offline mode should keep returning the legacy MotionLoader-compatible type."""
    motion_path = tmp_path / "motion.npz"
    np.savez(
        motion_path,
        fps=np.asarray([50.0], dtype=np.float32),
        joint_pos=np.zeros((2, 3), dtype=np.float32),
        joint_vel=np.zeros((2, 3), dtype=np.float32),
        body_pos_w=np.zeros((2, 4, 3), dtype=np.float32),
        body_quat_w=np.zeros((2, 4, 4), dtype=np.float32),
        body_lin_vel_w=np.zeros((2, 4, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((2, 4, 3), dtype=np.float32),
    )

    cfg = type(
        "Cfg",
        (),
        {
            "motion_source": "offline",
            "motion_source_uri": str(motion_path),
        },
    )()

    motion = build_motion_source(cfg, body_indexes=[1, 3], device="cpu")

    assert isinstance(motion, MotionLoader)
    assert motion.time_step_total == 2
    assert motion.body_pos_w.shape == (2, 2, 3)


def test_build_motion_source_rejects_unknown_motion_source():
    """Unknown motion source types should fail with a clear error."""
    cfg = type(
        "Cfg",
        (),
        {
            "motion_source": "invalid",
            "motion_source_uri": "/tmp/nowhere.npz",
        },
    )()

    with pytest.raises(ValueError, match="Unsupported motion_source"):
        build_motion_source(cfg, body_indexes=[0], device="cpu")


def test_build_motion_source_uses_realtime_builder(monkeypatch):
    """Realtime mode should delegate construction to the realtime builder."""
    expected = object()
    captured = {}

    def fake_builder(cfg, body_names):
        captured["cfg"] = cfg
        captured["body_names"] = body_names
        return expected

    monkeypatch.setattr(
        "awesome_deploy.utils.motion_source_factory.build_realtime_motion_source",
        fake_builder,
    )

    cfg = type(
        "Cfg",
        (),
        {
            "motion_source": "realtime",
            "motion_source_uri": "tcp://127.0.0.1:5555",
        },
    )()

    motion = build_motion_source(
        cfg,
        body_indexes=[1, 3],
        device="cpu",
        body_names=["pelvis_link", "torso_link"],
    )

    assert motion is expected
    assert captured["cfg"] is cfg
    assert captured["body_names"] == ["pelvis_link", "torso_link"]
