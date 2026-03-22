from awesome_deploy.utils.urdf_graph import UrdfGraph
from awesome_deploy.utils.motor_conf import *
from awesome_deploy import AWESOME_DIR
print(AWESOME_DIR)
current_path = AWESOME_DIR
class cfg:
    simulator_dt = 0.002
    policy_dt = 0.02

    group = {
        "policy": "policy/g1/2026-02-26_22-16-14_G1_slowly_walk",
        "motion": "03_fast_forward_walk_120Hz"
        # "motion": "01_slowly_forward_walk_120Hz"
    }
    policy_path = (
        current_path
        + "/"
        + group["policy"]
        + "/policy.onnx"
    )
    asset_path = current_path + "/assets/unitree_g1"
    mjcf_path = asset_path + "/g1_29dof_rev_1_0.xml"
    urdf_path = asset_path + "/g1_29dof_mode_15.urdf"
    motion_file = (
        current_path
        + "/"
        + group["policy"]
        + "/"
        + group["motion"]
        + ".npz"
    )
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

    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

    motion_reference_body = "torso_link"
