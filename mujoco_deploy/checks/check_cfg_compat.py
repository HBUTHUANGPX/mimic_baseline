import contextlib
import importlib
import io
import os
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def _import_cfg_fresh():
    sys.modules.pop("deploy.utils.cfg", None)
    captured_stdout = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout):
        module = importlib.import_module("deploy.utils.cfg")
    return module, captured_stdout.getvalue()


def check_cfg_sections_keep_legacy_access_without_import_stdout():
    cfg_module, stdout = _import_cfg_fresh()

    assert stdout == ""

    expected_sections = (
        "HumanCfg",
        "EnvCfg",
        "G1PolicyCfg",
        "G1PathCfg",
        "G1RobotControlCfg",
        "G1RobotModelCfg",
        "G1Cfg",
    )
    for section_name in expected_sections:
        assert hasattr(cfg_module, section_name)

    cfg = cfg_module.cfg
    assert cfg.robot_name == "g1"
    assert cfg_module.available_robots() == ("g1", "mdrx")

    assert cfg_module.current_path == os.getcwd()
    assert cfg.policy_raw_path == os.path.join(
        cfg_module.current_path, cfg.group["policy"]
    )
    assert cfg.policy_path == os.path.join(cfg.policy_raw_path, "policy.onnx")
    assert cfg.asset_path == os.path.join(
        cfg_module.current_path, "deploy/assets/unitree_g1"
    )
    assert cfg.mjcf_path == os.path.join(cfg.asset_path, "g1_29dof_rev_1_0.xml")
    assert cfg.urdf_path == os.path.join(cfg.asset_path, "g1_29dof_mode_15.urdf")
    assert cfg.motion_file == [
        os.path.join(cfg.policy_raw_path, "motion", name + ".npz")
        for name in cfg.motion_names
    ]

    assert cfg.desire_human_joint_names[0] == "Hips"
    assert cfg.human_anchor_name in cfg.desire_human_joint_names
    assert cfg.motion_reference_body in cfg.motion_body_names
    assert len(cfg.isaac_sim_joint_name) == 29
    assert "left_hip_pitch_joint" in cfg.isaac_sim_joint_name
    assert len(cfg.leg_P_gains + cfg.pelvis_P_gains + cfg.arm_P_gains) == 29
    assert len(cfg.leg_D_gains + cfg.pelvis_D_gains + cfg.arm_D_gains) == 29
    assert len(cfg.leg_tq_max + cfg.pelvis_tq_max + cfg.arm_tq_max) == 29
    assert len(cfg.leg_default_pos + cfg.pelvis_default_pos + cfg.arm_default_pos) == 29

    cfg_module.select_robot("mdrx")
    assert cfg.robot_name == "mdrx"
    assert cfg.policy_raw_path == os.path.join(
        cfg_module.current_path, "deploy/policy/mdrx/2026-06-23_17-09-42_test"
    )
    assert cfg.policy_path == os.path.join(cfg.policy_raw_path, "policy.onnx")
    assert cfg.asset_path == os.path.join(
        cfg_module.current_path, "deploy/assets/rx_27dof"
    )
    assert cfg.mjcf_path == os.path.join(cfg.asset_path, "rx_27dof.xml")
    assert cfg.urdf_path == os.path.join(
        cfg.asset_path, "rx_custom_collision_27dof.urdf"
    )
    assert cfg.motion_reference_body == "waist_pitch_link"
    assert len(cfg.isaac_sim_joint_name) == 27
    assert "l_hip_pitch_joint" in cfg.isaac_sim_joint_name
    assert len(cfg.leg_P_gains + cfg.pelvis_P_gains + cfg.arm_P_gains) == 27
    assert len(cfg.leg_D_gains + cfg.pelvis_D_gains + cfg.arm_D_gains) == 27
    assert len(cfg.leg_tq_max + cfg.pelvis_tq_max + cfg.arm_tq_max) == 27
    assert len(cfg.leg_default_pos + cfg.pelvis_default_pos + cfg.arm_default_pos) == 27

    cfg_module.select_robot("G1")
    assert cfg.robot_name == "g1"

    try:
        cfg_module.select_robot("unknown")
    except ValueError as exc:
        assert "unknown robot" in str(exc)
        assert "g1, mdrx" in str(exc)
    else:
        raise AssertionError("select_robot should reject unknown robots")


if __name__ == "__main__":
    check_cfg_sections_keep_legacy_access_without_import_stdout()
