from deploy.utils.urdf_graph import UrdfGraph
from deploy.utils.motor_conf import *
import os


current_path = os.getcwd()

HUMAN = 1
ROBOT = 0


def _normalize_motion_names(motion_names):
    if isinstance(motion_names, str):
        return [motion_names]
    return list(motion_names)


def _build_motion_files(policy_raw_path, motion_names):
    return [
        os.path.join(policy_raw_path, "motion", name + ".npz")
        for name in motion_names
    ]


class HumanCfg:
    desire_human_joint_names: list[str] = [
        "Hips",
        "Spine1",
        "Spine2",
        "Chest",
        "Neck1",
        "Neck2",
        "Head",
        "HeadEnd",
        "LeftShoulder",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightShoulder",
        "RightArm",
        "RightForeArm",
        "RightHand",
        "LeftLeg",
        "LeftShin",
        "LeftFoot",
        "LeftToeBase",
        "LeftToeEnd",
        "RightLeg",
        "RightShin",
        "RightFoot",
        "RightToeBase",
        "RightToeEnd",
    ]
    fsq_human_body_names: list[str] = [
        "Chest",
        "HeadEnd",
        "LeftShoulder",
        "LeftArm",
        "LeftForeArm",
        "RightShoulder",
        "RightArm",
        "RightForeArm",
        "LeftLeg",
        "LeftShin",
        "LeftFoot",
        "RightLeg",
        "RightShin",
        "RightFoot",
    ]
    human_anchor_name: str = "Hips"


class EnvCfg:
    simulator_dt = 0.002
    policy_dt = 0.02


class BasePolicyCfg:
    TOKEN_SELECTOR = ROBOT  # ROBOT, HUMAN

    only_leg_flag = False  # True, False
    with_wrist_flag = True  # True, False

    ################
    # action param #
    ################
    action_clip = 10.0
    action_scale = 0.25

    ####################
    # motion play mode #
    ####################
    """
     if motion_play is true, robots in mujoco will set
     qpos and qvel through the retargeting dataset
    """
    motion_play = False  # False, True


class G1PolicyCfg(BasePolicyCfg):
    # group = {
    #     "policy": "deploy/policy/g1/2026-02-26_22-16-14_G1_slowly_walk",
    #     "motion": "03_fast_forward_walk_120Hz"
    #     # "motion": "01_slowly_forward_walk_120Hz"
    # }
    group = {
        "policy": "deploy/policy/g1/2026-05-22_13-59-19_soma_cus_s",
        "motion": 
        [
            # "big_light_one_hand_pick_up_front_low_R_005__A508",
            # "body_stretch_1_004__A069",
            # "dance_basic_chaines_180_R_001__A306",
            # "dance_hiphop_shuffle_square_R_fast_002__A318",
            # "high_jump_R_001__A277",
            # "item_pick_up_standing_R_001__A410",
            # "Neutral_throw_ball_001__A057",
            # "Neutral_walk_forward_002__A057",
            # "small_light_one_hand_pick_up_front_low_002__A507",
            "wave_R_001__A428"
        ]
    }


class G1PathCfg(G1PolicyCfg):
    policy_raw_path = (
        os.path.join(current_path, G1PolicyCfg.group["policy"])
    )
    policy_path = (
        os.path.join(policy_raw_path, "policy.onnx")
    )
    asset_path = os.path.join(current_path, "deploy/assets/unitree_g1")
    mjcf_path = os.path.join(asset_path, "g1_29dof_rev_1_0.xml")
    urdf_path = os.path.join(asset_path, "g1_29dof_mode_15.urdf")
    
    motion_names = _normalize_motion_names(G1PolicyCfg.group["motion"])
    motion_file = _build_motion_files(policy_raw_path, motion_names)


class G1RobotControlCfg:
    ###################################################
    # stiffness damping and joint maximum torqueparam #
    ###################################################
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

    pelvis_P_gains = [
        STIFFNESS_7520_14,
        2.0 * STIFFNESS_5020,
        2.0 * STIFFNESS_5020,
    ]
    pelvis_D_gains = [
        DAMPING_7520_14,
        2.0 * DAMPING_5020,
        2.0 * DAMPING_5020,
    ]
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

    ########################
    # joint maximum torque #
    ########################

    #####################
    # joint default pos #
    #####################
    leg_default_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * (2)
    pelvis_default_pos = [0.0] * (3)
    arm_default_pos = [0.0] * (7 * 2)


class G1RobotModelCfg(G1PathCfg):
    ###########################################
    # Data conversion of isaac sim and mujoco #
    ###########################################
    motion_body_names = [
        "pelvis",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_pitch_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ]

    motion_reference_body = "torso_link"

    history_frames = 0
    future_frames = 9


class G1Cfg(G1RobotModelCfg, G1RobotControlCfg, HumanCfg, EnvCfg):
    robot_name = "g1"


class MdrxPolicyCfg(BasePolicyCfg):
    # group = {
    #     "policy": "deploy/policy/mdrx/2026-06-24_15-19-38_test",
    #     "motion":  
    #     [
    #         "seed_subsets_acrobatics_flip_roll_bvh_export_mdrx/230324/flip_360_004__A304"
    #     ],
    # }
    group = {
        "policy": "deploy/policy/mdrx/2026-06-24_11-38-56_test",
        "motion":  
        [
            "high_jump_R_001__A277",
            # "Neutral_throw_ball_001__A057",
            # "Neutral_walk_forward_002__A057",
        ],
    }


class MdrxPathCfg(MdrxPolicyCfg):
    policy_raw_path = os.path.join(current_path, MdrxPolicyCfg.group["policy"])
    policy_path = os.path.join(policy_raw_path, "policy.onnx")
    asset_path = os.path.join(current_path, "deploy/assets/rx_27dof")
    mjcf_path = os.path.join(asset_path, "rx_27dof.xml")
    urdf_path = os.path.join(asset_path, "rx_custom_collision_27dof.urdf")

    motion_names = _normalize_motion_names(MdrxPolicyCfg.group["motion"])
    motion_file = _build_motion_files(policy_raw_path, motion_names)


class MdrxRobotControlCfg:
    leg_P_gains = [80.0, 80.0, 80.0, 80.0, 30.0, 30.0] * 2
    leg_D_gains = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0] * 2
    leg_tq_max = [45.0, 35.0, 25.0, 45.0, 50.0, 50.0] * 2

    pelvis_P_gains = [80.0, 80.0, 80.0]
    pelvis_D_gains = [2.0, 2.0, 2.0]
    pelvis_tq_max = [27.0, 27.0, 27.0]

    arm_P_gains = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0] * 2
    arm_D_gains = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] * 2
    arm_tq_max = [14.0, 14.0, 14.0, 14.0, 14.0, 14.0] * 2

    leg_default_pos = [0.0] * (6 * 2)
    pelvis_default_pos = [0.0] * 3
    arm_default_pos = [0.0] * (6 * 2)


class MdrxRobotModelCfg(MdrxPathCfg):
    motion_body_names = [
        "base_link",
        "l_hip_yaw_link",
        "l_knee_link",
        "l_ankle_roll_link",
        "r_hip_yaw_link",
        "r_knee_link",
        "r_ankle_roll_link",
        "waist_pitch_link",
        "l_shoulder_pitch_link",
        "l_shoulder_yaw_link",
        "l_elbow_link",
        "l_wrist_yaw_link",
        "r_shoulder_pitch_link",
        "r_shoulder_yaw_link",
        "r_elbow_link",
        "r_wrist_yaw_link",
    ]

    motion_reference_body = "waist_pitch_link"

    history_frames = 0
    future_frames = 0


class MdrxCfg(MdrxRobotModelCfg, MdrxRobotControlCfg, HumanCfg, EnvCfg):
    robot_name = "mdrx"


ROBOT_CONFIGS = {
    G1Cfg.robot_name: G1Cfg,
    MdrxCfg.robot_name: MdrxCfg,
}


class ConfigProxy:
    def __init__(self, robot_name="g1"):
        self._derived_cache = {}
        self.select_robot(robot_name)

    @property
    def robot_name(self):
        return self._robot_name

    @property
    def config_class(self):
        return self._config_class

    @property
    def available_robots(self):
        return tuple(ROBOT_CONFIGS)

    def select_robot(self, robot_name):
        normalized_name = robot_name.lower()
        if normalized_name not in ROBOT_CONFIGS:
            choices = ", ".join(ROBOT_CONFIGS)
            raise ValueError(
                f"unknown robot '{robot_name}'. Available robots: {choices}"
            )
        self._robot_name = normalized_name
        self._config_class = ROBOT_CONFIGS[normalized_name]
        self._derived_cache.clear()
        return self

    def __getattr__(self, name):
        if name == "urdf_graph":
            return self._get_urdf_graph()
        if name == "isaac_sim_joint_name":
            return self._get_urdf_graph().bfs_joint_order()
        if name == "isaac_sim_link_name":
            return self._get_urdf_graph().bfs_link_order()
        try:
            return getattr(self._config_class, name)
        except AttributeError as exc:
            raise AttributeError(
                f"{self.__class__.__name__} has no attribute '{name}'"
            ) from exc

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        setattr(self._config_class, name, value)
        if name == "urdf_path":
            self._derived_cache.clear()

    def _get_urdf_graph(self):
        if "urdf_graph" not in self._derived_cache:
            self._derived_cache["urdf_graph"] = UrdfGraph(self.urdf_path)
        return self._derived_cache["urdf_graph"]


def available_robots():
    return cfg.available_robots


def select_robot(robot_name):
    return cfg.select_robot(robot_name)


cfg = ConfigProxy("g1")


cfg_human = HumanCfg
cfg_env = EnvCfg

# Legacy class names keep direct imports working for the default G1 robot.
PolicyCfg = G1PolicyCfg
PathCfg = G1PathCfg
RobotControlCfg = G1RobotControlCfg
RobotModelCfg = G1RobotModelCfg
