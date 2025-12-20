import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from general_motion_tracker_whole_body_teleoperation.assets import ASSET_DIR
from general_motion_tracker_whole_body_teleoperation.robots import (
    tn_delayed_pd_actuators,
)
from general_motion_tracker_whole_body_teleoperation.robots.tn_delayed_pd_actuators import (
    EncosActuatorCfg_EC_A8112,
    EncosActuatorCfg_EC_A6408,
    EncosActuatorCfg_EC_A10020,
    EncosActuatorCfg_EC_A4310,
    EncosActuatorCfg_EC_A4315,
    Ti5ActuatorCfg_CRA_RI60_80,
    Ti5ActuatorCfg_CRA_RI50_70,
    Ti5ActuatorCfg_CRA_RI40_52,
    Ti5ActuatorCfg_CRA_RI30_40,
    HTActuatorCfg_DMS_6015,
    HTActuatorCfg_DMS_6015_2,
)

ImplicitActuator_actuators = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_roll_joint",
            ".*_hip_yaw_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
        ],
        effort_limit_sim={
            ".*_hip_roll_joint": 60.0, # 保守稳定运行 40.0 峰值 90.0
            ".*_hip_yaw_joint": 40.0,# 保守稳定运行 25.0 峰值 90.0
            ".*_hip_pitch_joint": 160.0,# 保守稳定运行 130.0 峰值 330.0
            ".*_knee_joint": 160.0,# 保守稳定运行 130.0 峰值 330.0
        },
        velocity_limit_sim={
            ".*_hip_roll_joint": 16.441001554,
            ".*_hip_yaw_joint": 15.603243513,
            ".*_hip_pitch_joint": 12.88052988,
            ".*_knee_joint": 12.88052988,
        },
        stiffness={
            ".*_hip_roll_joint": 300,
            ".*_hip_yaw_joint": 200,
            ".*_hip_pitch_joint": 600,
            ".*_knee_joint": 600,
        },
        damping={
            ".*_hip_roll_joint": 2.5,
            ".*_hip_yaw_joint": 1.5,
            ".*_hip_pitch_joint": 3.0,
            ".*_knee_joint": 3.0,
        },
        armature={
            ".*_hip_roll_joint": 51181 * 1e-6,
            ".*_hip_yaw_joint": 58070 * 1e-6,
            ".*_hip_pitch_joint": 277376 * 1e-6,
            ".*_knee_joint": 277376 * 1e-6,
        },
    ),
    "feet": ImplicitActuatorCfg(
        effort_limit_sim=36.0,# 保守稳定运行 12.0*2 峰值 36.0*2
        velocity_limit_sim=9.320058206,
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        stiffness=200.0,
        damping=1.5,
        armature=24222 * 1e-6,
    ),
    "torso": ImplicitActuatorCfg(
        effort_limit_sim=42,
        velocity_limit_sim=8.58701992,
        joint_names_expr=["pelvis_joint"],
        stiffness=280,
        damping=1.5,
        armature=213 * 1e-7 * (51 ^ 2),
    ),
    "heads": ImplicitActuatorCfg(
        joint_names_expr=[
            "head_yaw_joint",
            "head_pitch_joint",
        ],
        effort_limit_sim={
            "head_yaw_joint": 2.52,
            "head_pitch_joint": 1.26,
        },
        velocity_limit_sim={
            "head_yaw_joint": 29.216811679,
            "head_pitch_joint": 58.433623357,
        },
        stiffness={
            "head_yaw_joint": 3.0,
            "head_pitch_joint": 1.5,
        },
        damping={
            "head_yaw_joint": 0.6,
            "head_pitch_joint": 0.3,
        },
        armature={
            "head_yaw_joint": 504 * 1e-7,
            "head_pitch_joint": 504 * 1e-7 * (2 ^ 2),
        },
    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_forearm_yaw_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
        ],
        effort_limit_sim={
            ".*_shoulder_pitch_joint": 42.0,
            ".*_shoulder_roll_joint": 42.0,
            ".*_shoulder_yaw_joint": 23.0,
            ".*_elbow_joint": 23.0,
            ".*_forearm_yaw_joint": 8.3,
            ".*_wrist_roll_joint": 3.3,
            ".*_wrist_pitch_joint": 3.3,
        },
        velocity_limit_sim={
            ".*_shoulder_pitch_joint": 8.58701992,
            ".*_shoulder_roll_joint": 8.58701992,
            ".*_shoulder_yaw_joint": 10.157816247,
            ".*_elbow_joint": 10.157816247,
            ".*_forearm_yaw_joint": 12.356931104,
            ".*_wrist_roll_joint": 12.356931104,
            ".*_wrist_pitch_joint": 12.356931104,
        },
        stiffness={
            ".*_shoulder_pitch_joint": 70,
            ".*_shoulder_roll_joint": 70,
            ".*_shoulder_yaw_joint": 70,
            ".*_elbow_joint": 70,
            ".*_forearm_yaw_joint": 20,
            ".*_wrist_roll_joint": 20,
            ".*_wrist_pitch_joint": 20,
        },
        damping={
            ".*_shoulder_pitch_joint": 1.5,
            ".*_shoulder_roll_joint": 1.5,
            ".*_shoulder_yaw_joint": 2,
            ".*_elbow_joint": 2,
            ".*_forearm_yaw_joint": 1.0,
            ".*_wrist_roll_joint": 1.0,
            ".*_wrist_pitch_joint": 1.0,
        },
        armature={
            ".*_shoulder_pitch_joint": 213 * 1e-7 * (51 ^ 2),
            ".*_shoulder_roll_joint": 213 * 1e-7 * (51 ^ 2),
            ".*_shoulder_yaw_joint": 124 * 1e-7 * (51 ^ 2),
            ".*_elbow_joint": 124 * 1e-7 * (51 ^ 2),
            ".*_forearm_yaw_joint": 80 * 1e-7 * (51 ^ 2),
            ".*_wrist_roll_joint": 16 * 1e-7 * (51 ^ 2),
            ".*_wrist_pitch_joint": 16 * 1e-7 * (51 ^ 2),
        },
    ),
}

IdealPDActuator_actuators = {
    "legs": IdealPDActuatorCfg(
        joint_names_expr=[
            ".*_hip_roll_joint",
            ".*_hip_yaw_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
        ],
        effort_limit_sim={
            ".*_hip_roll_joint": 52.0,
            ".*_hip_yaw_joint": 40.0,
            ".*_hip_pitch_joint": 100.0,
            ".*_knee_joint": 100.0,
        },
        velocity_limit_sim={
            ".*_hip_roll_joint": 16.441001554,
            ".*_hip_yaw_joint": 15.603243513,
            ".*_hip_pitch_joint": 12.88052988,
            ".*_knee_joint": 12.88052988,
        },
        stiffness={
            ".*_hip_roll_joint": 240,
            ".*_hip_yaw_joint": 240,
            ".*_hip_pitch_joint": 380,
            ".*_knee_joint": 380,
        },
        damping={
            ".*_hip_roll_joint": 2.5,
            ".*_hip_yaw_joint": 1.5,
            ".*_hip_pitch_joint": 3.0,
            ".*_knee_joint": 3.0,
        },
        armature={
            ".*_hip_roll_joint": 51181 * 1e-6,
            ".*_hip_yaw_joint": 58070 * 1e-6,
            ".*_hip_pitch_joint": 277376 * 1e-6,
            ".*_knee_joint": 277376 * 1e-6,
        },
    ),
    "feet": IdealPDActuatorCfg(
        effort_limit_sim=30.0,
        velocity_limit_sim=9.320058206,
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        stiffness=70.0,
        damping=1.5,
        armature=24222 * 1e-6,
    ),
    "torso": IdealPDActuatorCfg(
        effort_limit_sim=42,
        velocity_limit_sim=8.58701992,
        joint_names_expr=["pelvis_joint"],
        stiffness=280,
        damping=1.5,
        armature=213 * 1e-7 * (51 ^ 2),
    ),
    "heads": IdealPDActuatorCfg(
        joint_names_expr=[
            "head_yaw_joint",
            "head_pitch_joint",
        ],
        effort_limit_sim={
            "head_yaw_joint": 2.52,
            "head_pitch_joint": 1.26,
        },
        velocity_limit_sim={
            "head_yaw_joint": 29.216811679,
            "head_pitch_joint": 58.433623357,
        },
        stiffness={
            "head_yaw_joint": 3.0,
            "head_pitch_joint": 1.5,
        },
        damping={
            "head_yaw_joint": 0.6,
            "head_pitch_joint": 0.3,
        },
        armature={
            "head_yaw_joint": 504 * 1e-7,
            "head_pitch_joint": 504 * 1e-7 * (2 ^ 2),
        },
    ),
    "arms": IdealPDActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_forearm_yaw_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
        ],
        effort_limit_sim={
            ".*_shoulder_pitch_joint": 42.0,
            ".*_shoulder_roll_joint": 42.0,
            ".*_shoulder_yaw_joint": 23.0,
            ".*_elbow_joint": 23.0,
            ".*_forearm_yaw_joint": 8.3,
            ".*_wrist_roll_joint": 3.3,
            ".*_wrist_pitch_joint": 3.3,
        },
        velocity_limit_sim={
            ".*_shoulder_pitch_joint": 8.58701992,
            ".*_shoulder_roll_joint": 8.58701992,
            ".*_shoulder_yaw_joint": 10.157816247,
            ".*_elbow_joint": 10.157816247,
            ".*_forearm_yaw_joint": 12.356931104,
            ".*_wrist_roll_joint": 12.356931104,
            ".*_wrist_pitch_joint": 12.356931104,
        },
        stiffness={
            ".*_shoulder_pitch_joint": 70,
            ".*_shoulder_roll_joint": 70,
            ".*_shoulder_yaw_joint": 70,
            ".*_elbow_joint": 70,
            ".*_forearm_yaw_joint": 20,
            ".*_wrist_roll_joint": 20,
            ".*_wrist_pitch_joint": 20,
        },
        damping={
            ".*_shoulder_pitch_joint": 1.5,
            ".*_shoulder_roll_joint": 1.5,
            ".*_shoulder_yaw_joint": 2,
            ".*_elbow_joint": 2,
            ".*_forearm_yaw_joint": 1.0,
            ".*_wrist_roll_joint": 1.0,
            ".*_wrist_pitch_joint": 1.0,
        },
        armature={
            ".*_shoulder_pitch_joint": 213 * 1e-7 * (51 ^ 2),
            ".*_shoulder_roll_joint": 213 * 1e-7 * (51 ^ 2),
            ".*_shoulder_yaw_joint": 124 * 1e-7 * (51 ^ 2),
            ".*_elbow_joint": 124 * 1e-7 * (51 ^ 2),
            ".*_forearm_yaw_joint": 80 * 1e-7 * (51 ^ 2),
            ".*_wrist_roll_joint": 16 * 1e-7 * (51 ^ 2),
            ".*_wrist_pitch_joint": 16 * 1e-7 * (51 ^ 2),
        },
    ),
}

FullActuator_actuators = {
    "EC_A8112": EncosActuatorCfg_EC_A8112(
        joint_names_expr=[".*_hip_roll_joint"],
        stiffness=240,
        damping=2.5,
        friction=0.01,
        effort_limit_sim=52,
    ),
    "EC_A6408": EncosActuatorCfg_EC_A6408(
        joint_names_expr=[".*_hip_yaw_joint"],
        stiffness=240,
        damping=1.5,
        friction=0.01,
        effort_limit_sim=40,
    ),
    "EC_A10020": EncosActuatorCfg_EC_A10020(
        joint_names_expr=[
            ".*_hip_pitch_joint",
            ".*_knee_joint",
        ],
        stiffness=430,
        damping=3.0,
        friction=0.01,
        effort_limit_sim=130,
    ),
    "EC_A4310": EncosActuatorCfg_EC_A4315(
        joint_names_expr=[
            ".*_ankle_roll_joint",
            ".*_ankle_pitch_joint",
        ],
        stiffness=70,
        damping=1.5,
        friction=0.01,
        effort_limit_sim=40,
    ),
    "CRA_RI60_80_shoulder": Ti5ActuatorCfg_CRA_RI60_80(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
        ],
        stiffness=70,
        damping=1.5,
        friction=0.01,
        effort_limit_sim=42,
    ),
    "CRA_RI60_80_pelvis": Ti5ActuatorCfg_CRA_RI60_80(
        joint_names_expr=[
            "pelvis_joint",
        ],
        stiffness=280,
        damping=1.5,
        friction=0.01,
        effort_limit_sim=42,
    ),
    "CRA_RI50_70": Ti5ActuatorCfg_CRA_RI50_70(
        joint_names_expr=[
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
        ],
        stiffness=70,
        damping=2.0,
        friction=0.01,
        effort_limit_sim=23,
    ),
    "CRA_RI40_52": Ti5ActuatorCfg_CRA_RI40_52(
        joint_names_expr=[
            ".*_forearm_yaw_joint",
        ],
        stiffness=20,
        damping=1.0,
        friction=0.01,
        effort_limit_sim=8.3,
    ),
    "CRA_RI30_40": Ti5ActuatorCfg_CRA_RI30_40(
        joint_names_expr=[
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
        ],
        stiffness=20,
        damping=1.0,
        friction=0.01,
        effort_limit_sim=3.3,
    ),
    "HT_DMS_6015_2": HTActuatorCfg_DMS_6015_2(
        joint_names_expr=[
            "head_yaw_joint",
        ],
        stiffness=3.0,
        damping=0.6,
        friction=0.01,
        effort_limit_sim=1.26 * 2,
    ),
    "HT_DMS_6015": HTActuatorCfg_DMS_6015(
        joint_names_expr=[
            "head_pitch_joint",
        ],
        stiffness=1.5,
        damping=0.3,
        friction=0.01,
        effort_limit_sim=1.26,
    ),
}
Q1_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=ASSET_DIR + "/Q1/urdf/Q1_wo_hand_rl.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.992),
        joint_pos={
            ".*_hip_pitch_joint": 0,
            ".*_knee_joint": 0,
            ".*_ankle_pitch_joint": 0,
            ".*_elbow_joint": 0.0,
            "L_shoulder_roll_joint": 0,
            "L_shoulder_pitch_joint": 0.0,
            "R_shoulder_roll_joint": 0,
            "R_shoulder_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators=ImplicitActuator_actuators,
    # actuators=IdealPDActuator_actuators,
    # actuators=FullActuator_actuators,
)

Q1_ACTION_SCALE = {}
for a in Q1_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            Q1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
