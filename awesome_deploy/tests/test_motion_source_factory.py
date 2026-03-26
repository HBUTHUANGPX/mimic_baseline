"""Tests for motion source configuration and factory selection."""

from pathlib import Path

import numpy as np
import pytest

from awesome_deploy.utils import cfg as cfg_module
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


def test_q1_robot_cfg_ignores_environment_overrides(monkeypatch):
    """Runtime config should no longer be mutated by environment variables."""
    monkeypatch.setenv("AWESOME_DEPLOY_MOTION_SOURCE", "realtime")
    monkeypatch.setenv("AWESOME_DEPLOY_MOTION_SOURCE_URI", "tcp://192.168.1.2:5555")
    monkeypatch.setenv("AWESOME_DEPLOY_MOTION_SOURCE_TOPIC", "xsens.link_states.v1")
    monkeypatch.setenv("AWESOME_DEPLOY_MOTION_SOURCE_BUFFER_SIZE", "11")
    monkeypatch.setenv("AWESOME_DEPLOY_GMR_ROBOT", "Q1")
    monkeypatch.setenv("AWESOME_DEPLOY_GMR_HUMAN_HEIGHT", "1.72")

    cfg = Q1RobotCfg()

    assert cfg.motion_source == "offline"
    assert cfg.motion_source_uri == cfg.motion_file
    assert cfg.motion_source_topic == ""
    assert cfg.motion_source_buffer_size == 1
    assert cfg.gmr_robot == "Q1"
    assert cfg.gmr_human_height == pytest.approx(1.66)


def test_q1_robot_cfg_uses_sensible_realtime_defaults_from_cli(monkeypatch):
    """Realtime defaults should be populated from CLI selection alone."""
    monkeypatch.setenv("AWESOME_DEPLOY_MOTION_SOURCE", "realtime")

    cfg = Q1RobotCfg()
    cfg.apply_runtime_overrides(
        cfg_module.parse_runtime_overrides(
            [
                "deploy_g1_mujoco.py",
                "--motion-source",
                "realtime",
            ]
        )
    )

    assert cfg.motion_source == "realtime"
    assert cfg.motion_source_uri == "tcp://127.0.0.1:5555"
    assert cfg.motion_source_topic == "xsens.link_states.v1"
    assert cfg.motion_source_buffer_size >= 11


def test_q1_robot_cfg_accepts_cli_runtime_overrides():
    """CLI arguments should populate runtime config without env vars."""
    cfg = Q1RobotCfg()

    cfg.apply_runtime_overrides(
        cfg_module.parse_runtime_overrides(
            [
                "deploy_g1_mujoco.py",
                "--motion-source",
                "realtime",
                "--motion-source-uri",
                "tcp://192.168.1.10:6000",
                "--motion-source-topic",
                "xsens.link_states.v2",
                "--motion-source-buffer-size",
                "24",
                "--gmr-robot",
                "Q1",
                "--gmr-human-height",
                "1.75",
                "--motion-play",
                "--draw-xsens-frames",
                "--draw-xsens-labels",
                "--xsens-frame-axis-length",
                "0.11",
                "--xsens-frame-shaft-width",
                "0.01",
            ]
        )
    )

    assert cfg.motion_source == "realtime"
    assert cfg.motion_source_uri == "tcp://192.168.1.10:6000"
    assert cfg.motion_source_topic == "xsens.link_states.v2"
    assert cfg.motion_source_buffer_size == 24
    assert cfg.gmr_robot == "Q1"
    assert cfg.gmr_human_height == pytest.approx(1.75)
    assert cfg.motion_play is True
    assert cfg.realtime_draw_xsens_frames is True
    assert cfg.realtime_draw_xsens_labels is True
    assert cfg.realtime_xsens_frame_axis_length == pytest.approx(0.11)
    assert cfg.realtime_xsens_frame_shaft_width == pytest.approx(0.01)


def test_cli_runtime_overrides_ignore_environment(monkeypatch):
    """CLI flags should be the only runtime override mechanism."""
    monkeypatch.setenv("AWESOME_DEPLOY_MOTION_SOURCE", "offline")
    monkeypatch.setenv("AWESOME_DEPLOY_MOTION_SOURCE_URI", "/tmp/from_env.npz")
    monkeypatch.setenv("AWESOME_DEPLOY_GMR_HUMAN_HEIGHT", "1.60")
    monkeypatch.setenv("AWESOME_DEPLOY_REALTIME_DRAW_XSENS_FRAMES", "0")

    cfg = Q1RobotCfg()
    cfg.apply_runtime_overrides(
        cfg_module.parse_runtime_overrides(
            [
                "deploy_g1_mujoco.py",
                "--motion-source",
                "realtime",
                "--motion-source-uri",
                "tcp://127.0.0.1:5555",
                "--gmr-human-height",
                "1.82",
                "--draw-xsens-frames",
            ]
        )
    )

    assert cfg.motion_source == "realtime"
    assert cfg.motion_source_uri == "tcp://127.0.0.1:5555"
    assert cfg.gmr_human_height == pytest.approx(1.82)
    assert cfg.realtime_draw_xsens_frames is True


def test_q1_robot_cfg_accepts_offline_cli_motion_file_override(tmp_path: Path):
    """Offline mode should accept a custom motion npz path from CLI."""
    motion_path = tmp_path / "custom_motion.npz"
    motion_path.write_bytes(b"placeholder")

    cfg = Q1RobotCfg()
    cfg.apply_runtime_overrides(
        cfg_module.parse_runtime_overrides(
            [
                "deploy_g1_mujoco.py",
                "--motion-source",
                "offline",
                "--motion-source-uri",
                str(motion_path),
            ]
        )
    )

    assert cfg.motion_source == "offline"
    assert cfg.motion_source_uri == str(motion_path)
    assert cfg.motion_source_topic == ""
    assert cfg.motion_source_buffer_size == 1


def test_q1_robot_cfg_accepts_cli_disable_flags():
    """CLI should support explicit false toggles for boolean runtime flags."""
    cfg = Q1RobotCfg()
    cfg.apply_runtime_overrides(
        cfg_module.parse_runtime_overrides(
            [
                "deploy_g1_mujoco.py",
                "--motion-source",
                "realtime",
                "--motion-play",
                "--draw-xsens-frames",
                "--draw-xsens-labels",
                "--no-motion-play",
                "--no-draw-xsens-frames",
                "--no-draw-xsens-labels",
            ]
        )
    )

    assert cfg.motion_source == "realtime"
    assert cfg.motion_play is False
    assert cfg.realtime_draw_xsens_frames is False
    assert cfg.realtime_draw_xsens_labels is False


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
