import importlib.util
import math
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "mujoco_deploy/scripts/plot_joint_torque_speed.py"
LOG_PATH = REPO_ROOT / "mujoco_deploy/tmp/motion.npz"


EXPECTED_GROUPS = {
    "l_leg": [
        "l_hip_pitch_joint",
        "l_hip_roll_joint",
        "l_hip_yaw_joint",
        "l_knee_joint",
        "l_ankle_pitch_joint",
        "l_ankle_roll_joint",
    ],
    "r_leg": [
        "r_hip_pitch_joint",
        "r_hip_roll_joint",
        "r_hip_yaw_joint",
        "r_knee_joint",
        "r_ankle_pitch_joint",
        "r_ankle_roll_joint",
    ],
    "l_arm": [
        "l_shoulder_pitch_joint",
        "l_shoulder_roll_joint",
        "l_shoulder_yaw_joint",
        "l_elbow_joint",
        "l_wrist_yaw_joint",
        "l_wrist_roll_joint",
    ],
    "r_arm": [
        "r_shoulder_pitch_joint",
        "r_shoulder_roll_joint",
        "r_shoulder_yaw_joint",
        "r_elbow_joint",
        "r_wrist_yaw_joint",
        "r_wrist_roll_joint",
    ],
    "waist": [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ],
}


def _load_module():
    spec = importlib.util.spec_from_file_location("plot_joint_torque_speed", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    module = _load_module()
    assert module.JOINT_GROUPS == EXPECTED_GROUPS

    with np.load(LOG_PATH, allow_pickle=True) as log:
        data = module.load_joint_log(log)

    assert data.joint_names == sum(EXPECTED_GROUPS.values(), [])
    assert data.velocities.shape[1] == 27
    assert data.torques.shape == data.velocities.shape

    layout = module.build_joint_layout(data.joint_names)
    assert len(layout) == 5
    assert [row.group_name for row in layout] == ["l_leg", "r_leg", "l_arm", "r_arm", "waist"]
    assert [len(row.joint_indices) for row in layout] == [6, 6, 6, 6, 3]

    limits = module.read_mdrx_actuator_limits()
    fast_torque = 62.4
    fast_speed = (4000 * 2.0 * math.pi / 60.0) / 24.0
    slow_torque = 19.5
    slow_speed = (6000 * 2.0 * math.pi / 60.0) / 19.36

    assert limits["l_hip_pitch_joint"]["torque"] == fast_torque
    assert limits["l_hip_pitch_joint"]["speed"] == fast_speed
    assert limits["l_hip_yaw_joint"]["torque"] == slow_torque
    assert limits["l_hip_yaw_joint"]["speed"] == slow_speed
    assert limits["l_ankle_pitch_joint"]["torque"] == slow_torque * 2.0
    assert limits["waist_yaw_joint"]["torque"] == fast_torque
    assert limits["waist_roll_joint"]["torque"] == slow_torque * 2.0
    assert limits["l_shoulder_pitch_joint"]["torque"] == slow_torque
    assert set(limits) == set(data.joint_names)


if __name__ == "__main__":
    main()
