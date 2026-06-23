from deploy.utils.urdf_graph import UrdfGraph
from deploy.utils.motor_conf import *
import os
current_path = os.getcwd()
print(current_path)
HUMAN = 1
ROBOT = 0

class cfg_human:
    desire_human_joint_names: list[str] = [
        "Hips",
        "Spine1","Spine2","Chest",
        "Neck1","Neck2",
        "Head","HeadEnd",
        "LeftShoulder","LeftArm","LeftForeArm","LeftHand",
        "RightShoulder","RightArm","RightForeArm","RightHand",
        "LeftLeg","LeftShin","LeftFoot","LeftToeBase","LeftToeEnd",
        "RightLeg","RightShin","RightFoot","RightToeBase","RightToeEnd",
    ]
    fsq_human_body_names: list[str] = [
        "Chest",
        "HeadEnd",
        "LeftShoulder", "LeftArm", "LeftForeArm",
        "RightShoulder", "RightArm", "RightForeArm",
        "LeftLeg", "LeftShin", "LeftFoot",
        "RightLeg", "RightShin", "RightFoot",
    ]
    human_anchor_name: str = "Hips"

class cfg_env:
    simulator_dt = 0.002
    policy_dt = 0.02

class cfg(cfg_human, cfg_env):
    TOKEN_SELECTOR = ROBOT # ROBOT， HUMAN
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
    policy_raw_path = (
        current_path
        + "/"
        + group["policy"]
    )
    policy_path = (
        policy_raw_path
        + "/policy.onnx"
    )
    asset_path = current_path + "/deploy/assets/unitree_g1"
    mjcf_path = asset_path + "/g1_29dof_rev_1_0.xml"
    urdf_path = asset_path + "/g1_29dof_mode_15.urdf"
    motion_names = group["motion"]
    if isinstance(motion_names, str):
        motion_names = [motion_names]
    motion_file = []
    for name in motion_names:
        motion_file.append(os.path.join(policy_raw_path, "motion", name + ".npz"))
    only_leg_flag = False  # True, False
    with_wrist_flag = True  # True, False

    ###################################################
    # stiffness damping and joint maximum torqueparam #
    ###################################################
    leg_P_gains = [STIFFNESS_7520_14, STIFFNESS_7520_22, STIFFNESS_7520_14, STIFFNESS_7520_22, 2.0 * STIFFNESS_5020, 2.0 * STIFFNESS_5020] * 2
    leg_D_gains = [DAMPING_7520_14, DAMPING_7520_22, DAMPING_7520_14, DAMPING_7520_22, 2.0 * DAMPING_5020, 2.0 * DAMPING_5020] * 2
    leg_tq_max = [88.0, 139.0, 88.0, 139.0, 50.0, 50.0] * (2)

    pelvis_P_gains = [STIFFNESS_7520_14, 2.0 * STIFFNESS_5020, 2.0 * STIFFNESS_5020]
    pelvis_D_gains = [DAMPING_7520_14, 2.0 * DAMPING_5020, 2.0 * DAMPING_5020]
    pelvis_tq_max = [88, 50, 50]

    arm_P_gains = [STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_4010, STIFFNESS_4010] * (2)
    arm_D_gains = [DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_4010, DAMPING_4010] * (2)
    arm_tq_max = [25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0] * (2)

    ########################
    # joint maximum torque #
    ########################

    #####################
    # joint default pos #
    #####################
    leg_default_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * (2)
    pelvis_default_pos = [0.0] * (3)
    arm_default_pos = [0.0] * (7*2)

    ################
    # action param #
    ################
    action_clip = 10.0
    action_scale = 0.25

    #############
    # obs param #
    #############
    frame_stack = 1
    num_single_obs = 154 #1557 154

    ####################
    # motion play mode #
    ####################
    """
     if motion_play is true, robots in mujoco will set 
     qpos and qvel through the retargeting dataset 
    """
    motion_play = False  # False, True

    ###########################################
    # Data conversion of isaac sim and mujoco #
    ###########################################
    urdf_graph = UrdfGraph(urdf_path)
    isaac_sim_joint_name = urdf_graph.bfs_joint_order()

    isaac_sim_link_name = urdf_graph.bfs_link_order() # env.unwrapped.scene["robot"].body_names

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


