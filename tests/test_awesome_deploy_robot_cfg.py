import importlib
import sys

import pytest


def test_build_robot_cfg_returns_g1_instance():
    from awesome_deploy.robots import G1RobotCfg, build_robot_cfg

    cfg = build_robot_cfg("g1")

    assert isinstance(cfg, G1RobotCfg)
    assert cfg.robot_name == "g1"
    assert cfg.mjcf_path.endswith("g1_29dof_rev_1_0.xml")
    assert cfg.urdf_path.endswith("g1_29dof_mode_15.urdf")


def test_build_robot_cfg_rejects_unknown_robot():
    from awesome_deploy.robots import build_robot_cfg

    with pytest.raises(ValueError, match="Unknown robot_name"):
        build_robot_cfg("unknown")


def test_resolve_robot_name_prefers_cli_over_env(monkeypatch):
    from awesome_deploy.robots import resolve_robot_name

    monkeypatch.setenv("AWESOME_DEPLOY_ROBOT_NAME", "env_robot")

    resolved = resolve_robot_name(
        argv=["deploy_mujoco.py", "robot_name=cli_robot"],
        env_var="AWESOME_DEPLOY_ROBOT_NAME",
        default="g1",
    )

    assert resolved == "cli_robot"


def test_utils_cfg_uses_runtime_robot_resolution(monkeypatch):
    monkeypatch.setenv("AWESOME_DEPLOY_ROBOT_NAME", "g1")
    monkeypatch.setattr(sys, "argv", ["deploy_mujoco.py", "robot_name=g1"])

    sys.modules.pop("awesome_deploy.utils.cfg", None)
    cfg_module = importlib.import_module("awesome_deploy.utils.cfg")

    assert cfg_module.cfg.robot_name == "g1"
