from isaaclab.utils import configclass

from general_motion_tracker_whole_body_teleoperation.tasks.tracking.tracking_env_cfg import (
    TrackingEnvCfg,
)
from general_motion_tracker_whole_body_teleoperation.robots.g1 import (
    G1_ACTION_SCALE,
    G1_CYLINDER_CFG,
)

@configclass
class G1TerrainEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
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
        self.rewards.foot_contact_velocity.params["body_names"] = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ]
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
        ]
        self.events.knee_link_com.params["asset_cfg"].body_names = [
            "left_knee_link",
            "right_knee_link",
        ]
        self.events.pelvis_com.params["asset_cfg"].body_names = ["pelvis"]

        self.terminations.ee_body_pos_knee.params["body_names"] = [
            "left_knee_link",
            "right_knee_link",
        ]
        self.terminations.ee_body_pos_ankle.params["body_names"] = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ]
        self.terminations.ee_body_pos_wrist.params["body_names"] = [
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ]

@configclass
class G1TerrainEnvCfg_PLAY(G1TerrainEnvCfg):
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
class G1TerrainPureEnvCfg(G1TerrainEnvCfg):
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
class G1TerrainDistillEnvCfg(G1TerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.future_frames = 10
        self.observations.proprioception_with_noise_wo_privilege.history_length = 8
        self.observations.proprioception.history_length = 8
