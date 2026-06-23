import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from general_motion_tracker_whole_body_teleoperation.assets import ASSET_DIR
from general_motion_tracker_whole_body_teleoperation.robots import (
    tn_delayed_pd_actuators,
)
import torch
import os
from isaaclab.utils import configclass

pi = 3.141592653589793
scale = 1.15
ARMATURE_HEAVY = 0.027944  # J4340: hip_roll/pitch, knee, ankle, waist
ARMATURE_LIGHT = 0.001744  # J4310: shoulder, elbow, wrist, hip_yaw

NATURAL_FREQ = 10 * 2.0 * pi  # 10Hz
DAMPING_RATIO = 2.0


ImplicitActuator_actuators = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_pitch_joint",
            ".*_hip_roll_joint",
            ".*_hip_yaw_joint",
            ".*_knee_joint",
        ],
        effort_limit_sim={
            ".*_hip_pitch_joint": 45.0,
            ".*_hip_roll_joint": 35.0,
            ".*_hip_yaw_joint": 25.0,
            ".*_knee_joint": 45.0,
        },
        velocity_limit_sim={
            ".*_hip_pitch_joint": 8.0,
            ".*_hip_roll_joint": 12.9,
            ".*_hip_yaw_joint": 18.0,
            ".*_knee_joint": 12.5,
        },
        stiffness={
            ".*_hip_pitch_joint": 80,
            ".*_hip_roll_joint": 80,
            ".*_hip_yaw_joint": 80,
            ".*_knee_joint": 80,
        },
        damping={
            ".*_hip_pitch_joint": 2.0,
            ".*_hip_roll_joint": 2.0,
            ".*_hip_yaw_joint": 2.0,
            ".*_knee_joint": 2.0,
        },
        armature={
            ".*_hip_pitch_joint": ARMATURE_HEAVY,
            ".*_hip_roll_joint": ARMATURE_HEAVY,
            ".*_hip_yaw_joint": ARMATURE_LIGHT,
            ".*_knee_joint": ARMATURE_HEAVY,
        },
    ),
    "feet": ImplicitActuatorCfg(
        effort_limit_sim=50.0,
        velocity_limit_sim=37.0,
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        stiffness= 30.0,
        damping= 2.0,
        armature=2.0 * ARMATURE_LIGHT,
    ),
    "waist": ImplicitActuatorCfg(
        effort_limit_sim=27,
        velocity_limit_sim=12.0,
        joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
        stiffness=80.0,
        damping=2.0,
        armature=2.0 * ARMATURE_HEAVY,
    ),
    "waist_yaw": ImplicitActuatorCfg(
        effort_limit_sim=27,
        velocity_limit_sim=12.0,
        joint_names_expr=["waist_yaw_joint"],
        stiffness=80,
        damping=2.0,
        armature=ARMATURE_HEAVY,
    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_yaw_joint",
            ".*_wrist_roll_joint",
        ],
        effort_limit_sim={
            ".*_shoulder_pitch_joint": 14.0,
            ".*_shoulder_roll_joint": 14.0,
            ".*_shoulder_yaw_joint": 14.0,
            ".*_elbow_joint": 14.0,
            ".*_wrist_yaw_joint": 14.0,
            ".*_wrist_roll_joint": 14.0,
        },
        velocity_limit_sim={
            ".*_shoulder_pitch_joint": 18.0,
            ".*_shoulder_roll_joint": 18.0,
            ".*_shoulder_yaw_joint": 18.0,
            ".*_elbow_joint": 18.0,
            ".*_wrist_yaw_joint": 18.0,
            ".*_wrist_roll_joint": 18.0,
        },
        stiffness={
            ".*_shoulder_pitch_joint": 4,
            ".*_shoulder_roll_joint": 4,
            ".*_shoulder_yaw_joint": 4,
            ".*_elbow_joint": 4,
            ".*_wrist_yaw_joint": 4,
            ".*_wrist_roll_joint": 4,
        },
        damping={
            ".*_shoulder_pitch_joint": 1,
            ".*_shoulder_roll_joint": 1,
            ".*_shoulder_yaw_joint": 1,
            ".*_elbow_joint": 1,
            ".*_wrist_yaw_joint": 1,
            ".*_wrist_roll_joint": 1,
        },
        armature={
            ".*_shoulder_pitch_joint": ARMATURE_LIGHT,
            ".*_shoulder_roll_joint": ARMATURE_LIGHT,
            ".*_shoulder_yaw_joint": ARMATURE_LIGHT,
            ".*_elbow_joint": ARMATURE_LIGHT,
            ".*_wrist_yaw_joint": ARMATURE_LIGHT,
            ".*_wrist_roll_joint": ARMATURE_LIGHT,
        },
    ),
}

MDRX_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path="assets/rx_27dof/rx_custom_collision_27dof.urdf",
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
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.1,
            rest_offset=-0.001,
        ),  # TODO:检查这个是否对碰撞检测有好的影响
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.523),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators=ImplicitActuator_actuators,
    # actuators=IdealPDActuator_actuators,
    # actuators=FullActuator_actuators,
)

MDRX_ACTION_SCALE = {}
for a in MDRX_CYLINDER_CFG.actuators.values():
    # e = a.Y1
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}

    for n in names:
        if n in e and n in s and s[n]:
            print(f"{n}: \n    el: {e[n]}\n    sf: {s[n]}")
            MDRX_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
            # 是否使用这种action scale的计算方式，具体考量需要参考个人调试笔记12月20日记录
print("MDRX_ACTION_SCALE:", MDRX_ACTION_SCALE)
