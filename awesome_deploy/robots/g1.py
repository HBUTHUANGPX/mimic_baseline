from awesome_deploy.robots.base import BaseRobotCfg
from awesome_deploy.utils.motor_conf import (
    DAMPING_4010,
    DAMPING_5020,
    DAMPING_7520_14,
    DAMPING_7520_22,
    STIFFNESS_4010,
    STIFFNESS_5020,
    STIFFNESS_7520_14,
    STIFFNESS_7520_22,
)


class G1RobotCfg(BaseRobotCfg):
    robot_name = "g1"
    group = {
        "policy": "deploy/policy/g1/2026-02-26_22-16-14_G1_slowly_walk",
        "motion": "03_fast_forward_walk_120Hz",
    }
    asset_dirname = "unitree_g1"
    mjcf_filename = "g1_29dof_rev_1_0.xml"
    urdf_filename = "g1_29dof_mode_15.urdf"

    leg_P_gains = [
        STIFFNESS_7520_14,
        STIFFNESS_7520_22,
        STIFFNESS_7520_14,
        STIFFNESS_7520_22,
        2.0 * STIFFNESS_5020,
        2.0 * STIFFNESS_5020,
    ] * 2
    leg_D_gains = [
        DAMPING_7520_14,
        DAMPING_7520_22,
        DAMPING_7520_14,
        DAMPING_7520_22,
        2.0 * DAMPING_5020,
        2.0 * DAMPING_5020,
    ] * 2
    leg_tq_max = [88.0, 139.0, 88.0, 139.0, 50.0, 50.0] * 2

    pelvis_P_gains = [STIFFNESS_7520_14, 2.0 * STIFFNESS_5020, 2.0 * STIFFNESS_5020]
    pelvis_D_gains = [DAMPING_7520_14, 2.0 * DAMPING_5020, 2.0 * DAMPING_5020]
    pelvis_tq_max = [88, 50, 50]

    arm_P_gains = [
        STIFFNESS_5020,
        STIFFNESS_5020,
        STIFFNESS_5020,
        STIFFNESS_5020,
        STIFFNESS_5020,
        STIFFNESS_4010,
        STIFFNESS_4010,
    ] * 2
    arm_D_gains = [
        DAMPING_5020,
        DAMPING_5020,
        DAMPING_5020,
        DAMPING_5020,
        DAMPING_5020,
        DAMPING_4010,
        DAMPING_4010,
    ] * 2
    arm_tq_max = [25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0] * 2

    leg_default_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * 2
    pelvis_default_pos = [0.0] * 3
    arm_default_pos = [0.0] * 14

    motion_body_names = [
        "pelvis",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ]
    motion_reference_body = "torso_link"
