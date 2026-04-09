from isaaclab.utils import configclass

from general_motion_tracker_whole_body_teleoperation.tasks.tracking.tracking_env_cfg import (
    TrackingEnvCfg,
)
from general_motion_tracker_whole_body_teleoperation.tasks.tracking.teacher_wo_d_tracking_env_cfg import (
    TrackingEnvCfg as TeacherTrackingEnvCfg,
)
from general_motion_tracker_whole_body_teleoperation.tasks.tracking.pure_tracking_env_cfg import (
    TrackingEnvCfg as PureTrackingEnvCfg,
)
from general_motion_tracker_whole_body_teleoperation.tasks.tracking.distill_tracking_env_cfg import (
    TrackingEnvCfg as DissTrackingEnvCfg,
)

from general_motion_tracker_whole_body_teleoperation.robots.q1 import (
    Q1_ACTION_SCALE,
    Q1_CYLINDER_CFG,
)


@configclass
class Q1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = Q1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = Q1_ACTION_SCALE
        self.commands.motion.reference_body = "torso_link"
        self.commands.motion.body_names = [
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


@configclass
class Q1FlatPureEnvCfg(Q1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.future_frames = 0
        self.observations.proprioception_with_noise_wo_privilege.history_length = 8
        self.observations.proprioception.history_length = 8

        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.pelvis_com = None
        self.events.knee_link_com = None
        self.events.robot_scale_mass = None
        self.events.robot_joint_stiffness_and_damping = None
        self.events.push_robot = None

        self.commands.motion.velocity_range = {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0.0, 0.0),
            "pitch": (-0.0, 0.0),
            "yaw": (-0.0, 0.0),
        }
        self.commands.motion.joint_position_range = (-0.0, 0.0)


@configclass
class Q1FlatDistillEnvCfg(Q1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.future_frames = 10
        self.observations.proprioception_with_noise_wo_privilege.history_length = 8
        self.observations.proprioception.history_length = 8


@configclass
class Q1FlatTeacherEnvCfg(TeacherTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = Q1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = Q1_ACTION_SCALE
        self.commands.motion.reference_body = "torso_link"
        self.commands.motion.body_names = [
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


@configclass
class PureQ1FlatEnvCfg(PureTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = Q1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = Q1_ACTION_SCALE
        self.commands.motion.reference_body = "torso_link"
        self.commands.motion.body_names = [
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


@configclass
class DissQ1FlatEnvCfg(DissTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = Q1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = Q1_ACTION_SCALE
        self.commands.motion.reference_body = "torso_link"
        self.commands.motion.body_names = [
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
