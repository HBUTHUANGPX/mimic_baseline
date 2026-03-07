from isaaclab.utils import configclass

from general_motion_tracker_whole_body_teleoperation.tasks.tracking_g1.tracking_env_cfg import TrackingEnvCfg
from general_motion_tracker_whole_body_teleoperation.tasks.tracking_g1.pure_tracking_env_cfg import TrackingEnvCfg as PureTrackingEnvCfg
from general_motion_tracker_whole_body_teleoperation.tasks.tracking_g1.distill_tracking_env_cfg import TrackingEnvCfg as DissTrackingEnvCfg

from general_motion_tracker_whole_body_teleoperation.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG

reference_body = "torso_link"
body_names = [
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
@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.reference_body = reference_body
        self.commands.motion.body_names = body_names

@configclass
class PureG1FlatEnvCfg(PureTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.reference_body = reference_body
        self.commands.motion.body_names = body_names

@configclass
class DissG1FlatEnvCfg(DissTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.reference_body = reference_body
        self.commands.motion.body_names = body_names