from awesome_deploy.utils.urdf_graph import UrdfGraph
from awesome_deploy.utils.motor_conf import *
from awesome_deploy import AWESOME_DIR

import os
import sys

print(AWESOME_DIR)
current_path = AWESOME_DIR


class BaseRobotCfg:
    group: dict
    urdf_path: str

    def __init__(self):

        self.simulator_dt = 0.002
        self.policy_dt = 0.02

        self.policy_path = current_path + "/" + self.group["policy"] + "/policy.onnx"

        self.motion_file = (
            current_path
            + "/"
            + self.group["policy"]
            + "/"
            + self.group["motion"]
            + ".npz"
        )
        ################
        # action param #
        ################
        self.action_clip = 10.0
        self.action_scale = 0.25

        ####################
        # motion play mode #
        ####################
        """
        if motion_play is true, robots in mujoco will set 
        qpos and qvel through the retargeting dataset 
        """
        self.motion_play = False  # False, True

        ###########################################
        # Data conversion of isaac sim and mujoco #
        ###########################################
        self.urdf_graph = UrdfGraph(self.urdf_path)
        self.isaac_sim_joint_name = self.urdf_graph.bfs_joint_order()

        self.isaac_sim_link_name = (
            self.urdf_graph.bfs_link_order()
        )  # env.unwrapped.scene["robot"].body_names


class G1RobotCfg(BaseRobotCfg):
    group = {
        "policy": "policy/g1/2026-02-26_22-16-14_G1_slowly_walk",
        "motion": "03_fast_forward_walk_120Hz",
        # "motion": "01_slowly_forward_walk_120Hz"
    }

    asset_path = current_path + "/assets/unitree_g1"
    mjcf_path = asset_path + "/g1_29dof_rev_1_0.xml"
    urdf_path = asset_path + "/g1_29dof_mode_15.urdf"

    ###################################################
    # stiffness damping and joint maximum torqueparam #
    ###################################################
    motor_cfg = {
        "leg": {
            "stiffness": [
                STIFFNESS_7520_14,
                STIFFNESS_7520_22,
                STIFFNESS_7520_14,
                STIFFNESS_7520_22,
                2.0 * STIFFNESS_5020,
                2.0 * STIFFNESS_5020,
            ]
            * 2,
            "damping": [
                DAMPING_7520_14,
                DAMPING_7520_22,
                DAMPING_7520_14,
                DAMPING_7520_22,
                2.0 * DAMPING_5020,
                2.0 * DAMPING_5020,
            ]
            * 2,
            "torque_max": [88.0, 139.0, 88.0, 139.0, 50.0, 50.0] * (2),
            "default_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * (2),
        },
        "pelvis": {
            "stiffness": [
                STIFFNESS_7520_14,
                2.0 * STIFFNESS_5020,
                2.0 * STIFFNESS_5020,
            ],
            "damping": [DAMPING_7520_14, 2.0 * DAMPING_5020, 2.0 * DAMPING_5020],
            "torque_max": [88, 50, 50],
            "default_pos": [0.0] * (3),
        },
        "arm": {
            "stiffness": [
                STIFFNESS_5020,
                STIFFNESS_5020,
                STIFFNESS_5020,
                STIFFNESS_5020,
                STIFFNESS_5020,
                STIFFNESS_4010,
                STIFFNESS_4010,
            ]
            * (2),
            "damping": [
                DAMPING_5020,
                DAMPING_5020,
                DAMPING_5020,
                DAMPING_5020,
                DAMPING_5020,
                DAMPING_4010,
                DAMPING_4010,
            ]
            * (2),
            "torque_max": [25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0] * (2),
            "default_pos": [0.0] * (7 * 2),
        },
    }
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
    leg_tq_max = [88.0, 139.0, 88.0, 139.0, 50.0, 50.0] * (2)

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
    ] * (2)
    arm_D_gains = [
        DAMPING_5020,
        DAMPING_5020,
        DAMPING_5020,
        DAMPING_5020,
        DAMPING_5020,
        DAMPING_4010,
        DAMPING_4010,
    ] * (2)
    arm_tq_max = [25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0] * (2)

    #####################
    # joint default pos #
    #####################
    leg_default_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * (2)
    pelvis_default_pos = [0.0] * (3)
    arm_default_pos = [0.0] * (7 * 2)

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

    def __init__(self):
        super().__init__()


ROBOT_CFG_REGISTRY = {
    "g1": G1RobotCfg,
}


def resolve_robot_name(
    argv: list[str] | None = None,
    env_var: str = "AWESOME_DEPLOY_ROBOT_NAME",
    default: str = "g1",
) -> str:
    argv = argv if argv is not None else sys.argv
    for index, arg in enumerate(argv):
        if arg.startswith("robot_name="):
            return arg.split("=", 1)[1]
        if arg.startswith("--robot_name="):
            return arg.split("=", 1)[1]
        if arg == "--robot_name" and index + 1 < len(argv):
            return argv[index + 1]

    return os.getenv(env_var, default)


def build_robot_cfg(robot_name: str) -> BaseRobotCfg:
    try:
        print(robot_name)
        cfg_cls = ROBOT_CFG_REGISTRY[robot_name]
    except KeyError as exc:
        supported = ", ".join(sorted(ROBOT_CFG_REGISTRY))
        raise ValueError(
            f"Unknown robot_name '{robot_name}'. Supported robots: {supported}"
        ) from exc
    return cfg_cls()


cfg: G1RobotCfg | BaseRobotCfg = build_robot_cfg(resolve_robot_name())


def get_robot_cfg(robot_name=None):
    resolved_robot_name = robot_name or resolve_robot_name()
    return build_robot_cfg(resolved_robot_name)
