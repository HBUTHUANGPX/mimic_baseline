from isaaclab.utils import configclass

from general_motion_tracker_whole_body_teleoperation.tasks.tracking.tracking_env_cfg import (
    TrackingEnvCfg,
)
from general_motion_tracker_whole_body_teleoperation.robots.mdrx import (
    MDRX_ACTION_SCALE,
    MDRX_CYLINDER_CFG,
)


@configclass
class MDRXTerrainEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = MDRX_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/waist_pitch_link"

        self.actions.joint_pos.scale = MDRX_ACTION_SCALE
        self.commands.motion.anchor_body_name = "waist_pitch_link"
        self.commands.motion.body_names = [
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
            "l_wrist_roll_link",
            "r_shoulder_pitch_link",
            "r_shoulder_yaw_link",
            "r_elbow_link",
            "r_wrist_roll_link",
        ]
        self.rewards.foot_contact_velocity.params["body_names"] = [
            "l_ankle_roll_link",
            "r_ankle_roll_link",
        ]
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^(?!l_ankle_roll_link$)(?!r_ankle_roll_link$)(?!l_wrist_roll_link$)(?!r_wrist_roll_link$).+$"
        ]
        self.events.knee_link_com.params["asset_cfg"].body_names = [
            "l_knee_link",
            "r_knee_link",
        ]
        self.events.pelvis_com.params["asset_cfg"].body_names = ["base_link"]
        self.events.base_com.params["asset_cfg"].body_names = ["waist_pitch_link"]

        self.terminations.ee_body_pos_knee.params["body_names"] = [
            "l_knee_link",
            "r_knee_link",
        ]
        self.terminations.ee_body_pos_ankle.params["body_names"] = [
            "l_ankle_roll_link",
            "r_ankle_roll_link",
        ]
        self.terminations.ee_body_pos_wrist.params["body_names"] = [
            "l_wrist_roll_link",
            "r_wrist_roll_link",
        ]


@configclass
class MDRXTerrainEnvCfg_PLAY(MDRXTerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.terminations = None


@configclass
class MDRXTerrainPureEnvCfg(MDRXTerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.future_frames = 0
        self.observations.proprioception_with_noise_wo_privilege.history_length = 0
        self.observations.proprioception.history_length = 0

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
class MDRXTerrainDistillEnvCfg(MDRXTerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.future_frames = 10
        self.observations.proprioception_with_noise_wo_privilege.history_length = 8
        self.observations.proprioception.history_length = 8
