"""Robot configuration definitions for deployment-time simulation."""

from awesome_deploy.utils.urdf_graph import UrdfGraph
from awesome_deploy.utils.motor_conf import *
from awesome_deploy import AWESOME_DIR

import os
import sys

current_path = AWESOME_DIR


class BaseRobotCfg:
    """Base configuration shared by all supported robots.

    The class centralizes policy asset paths, timing parameters, and name
    mappings required to translate between MuJoCo ordering and dataset ordering.
    Concrete robot subclasses are expected to provide robot-specific assets and
    motor parameters through class attributes.
    """

    group: dict
    urdf_path: str

    def __init__(self):
        """Initializes paths and shared deployment parameters for one robot."""

        self.simulator_dt = 0.002
        self.policy_dt = 0.02

        # Keep all policy-related assets colocated so a robot can switch model,
        # protocol, and motion files by changing one policy directory entry.
        self.policy_dir = current_path + "/" + self.group["policy"]
        self.policy_path = self.policy_dir + "/policy.onnx"
        self.protocol_path = self.policy_dir + "/policy.protocol.yaml"

        self.motion_file = (
            self.policy_dir + "/" + self.group["motion"] + ".npz"
        )
        # Action scaling is applied after the neural policy output is produced
        # and before the target joint position is sent to the PD controller.
        self.action_clip = 10.0
        self.action_scale = 0.25

        # When enabled, the simulator follows the motion dataset directly
        # instead of applying policy actions through the PD controller.
        self.motion_play = False

        # Build name mappings once so runtime code can cheaply convert among
        # URDF order, MuJoCo order, and motion dataset order.
        self.urdf_graph = UrdfGraph(self.urdf_path)
        self.isaac_sim_joint_name = self.urdf_graph.bfs_joint_order()
        self.isaac_sim_link_name = self.urdf_graph.bfs_link_order()


class G1RobotCfg(BaseRobotCfg):
    """Concrete deployment configuration for the Unitree G1 robot."""

    group = {
        "policy": "policy/g1/2026-02-26_22-16-14_G1_slowly_walk",
        "motion": "03_fast_forward_walk_120Hz",
    }

    asset_path = current_path + "/assets/unitree_g1"
    mjcf_path = asset_path + "/g1_29dof_rev_1_0.xml"
    urdf_path = asset_path + "/g1_29dof_mode_15.urdf"

    # Motor groups are stored separately to match the physical robot's
    # kinematic layout while still allowing runtime code to flatten them into a
    # single MuJoCo joint vector.
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
        """Initializes the base configuration with G1-specific metadata."""
        super().__init__()
class Q1RobotCfg(BaseRobotCfg):
    """Concrete deployment configuration for the Unitree G1 robot."""

    group = {
        "policy": "policy/q1/2026-03-20_15-37-52_xsens_all_fsq_s",
        "motion": "Aeroplane_BR",
        # "motion": "251014_single_action_forward_walk",
    }

    asset_path = current_path + "/assets/Q1"
    mjcf_path = asset_path + "/mjcf/Q1_wo_hand.xml"
    urdf_path = asset_path + "/urdf/Q1_wo_hand_rl.urdf"

    # Motor groups are stored separately to match the physical robot's
    # kinematic layout while still allowing runtime code to flatten them into a
    # single MuJoCo joint vector.
    motor_cfg = {
        "leg": {
            "stiffness": [
                350,
                150,
                450,
                450,
                70.0,
                70.0,
            ]
            * 2,
            "damping": [
                6.0,
                4.5,
                12.0,
                12.0,
                1.5,
                1.5,
            ]
            * 2,
            "torque_max": [158.7, 64.4, 158.7, 158.7, 75.9, 75.9] * (2),
            "default_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * (2),
        },
        "pelvis": {
            "stiffness": [280],
            "damping": [4.5],
            "torque_max": [158.7],
            "default_pos": [0.0],
        },
        "arm": {
            "stiffness": [
                70,
                70,
                70,
                70,
                20,
                20,
                20,
            ]
            * (2),
            "damping": [
                1.5,
                1.5,
                2,
                2,
                1.0,
                1.0,
                1.0,
            ]
            * (2),
            "torque_max": [42.0, 42.0, 23.0, 23.0, 8.3, 3.3, 3.3] * (2),
            "default_pos": [0.0] * (7 * 2),
        },
        "head": {
            "stiffness": [3.0, 1.5],
            "damping": [0.6, 0.3],
            "torque_max": [2.52, 1.26],
            "default_pos": [0.0, 0.0],
        },
    }

    motion_body_names = [
        "pelvis_link",
        "L_hip_yaw_link",
        "L_knee_link",
        "L_ankle_roll_link",
        "R_hip_yaw_link",
        "R_knee_link",
        "R_ankle_roll_link",
        "torso_link",
        "L_shoulder_roll_link",
        "L_elbow_link",
        "L_wrist_pitch_link",
        "R_shoulder_roll_link",
        "R_elbow_link",
        "R_wrist_pitch_link",
        "head_pitch_link",
    ]
    motion_reference_body = "torso_link"

    def __init__(self):
        """Initializes the base configuration with Q1-specific metadata."""
        super().__init__()


ROBOT_CFG_REGISTRY = {
    "g1": G1RobotCfg,
    "q1": Q1RobotCfg,
}


def resolve_robot_name(
    argv: list[str] | None = None,
    env_var: str = "AWESOME_DEPLOY_ROBOT_NAME",
    default: str = "g1",
) -> str:
    """Resolves the active robot name from CLI arguments or environment.

    Args:
        argv: Optional argument list. Defaults to ``sys.argv``.
        env_var: Environment variable checked when CLI arguments do not specify
            the robot.
        default: Fallback robot name.

    Returns:
        The resolved robot identifier, such as ``"g1"``.
    """
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
    """Builds the concrete configuration object for one robot name.

    Args:
        robot_name: Registry key identifying the robot configuration class.

    Returns:
        Instantiated robot configuration.

    Raises:
        ValueError: If the requested robot is not registered.
    """
    try:
        cfg_cls = ROBOT_CFG_REGISTRY[robot_name]
    except KeyError as exc:
        supported = ", ".join(sorted(ROBOT_CFG_REGISTRY))
        raise ValueError(
            f"Unknown robot_name '{robot_name}'. Supported robots: {supported}"
        ) from exc
    return cfg_cls()


cfg: G1RobotCfg | Q1RobotCfg | BaseRobotCfg = build_robot_cfg(resolve_robot_name())


def get_robot_cfg(robot_name=None):
    """Returns the active robot config or resolves a specific robot on demand."""
    resolved_robot_name = robot_name or resolve_robot_name()
    return build_robot_cfg(resolved_robot_name)
