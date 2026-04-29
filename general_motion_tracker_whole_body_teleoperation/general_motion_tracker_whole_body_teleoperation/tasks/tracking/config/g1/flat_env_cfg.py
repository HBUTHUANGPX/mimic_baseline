from isaaclab.utils import configclass

from general_motion_tracker_whole_body_teleoperation.tasks.tracking.tracking_env_cfg import (
    TrackingEnvCfg,
)
from general_motion_tracker_whole_body_teleoperation.robots.g1 import (
    G1_ACTION_SCALE,
    G1_CYLINDER_CFG,
)
from general_motion_tracker_whole_body_teleoperation.tasks.tracking.config.g1.terrain_env_cfg import (
    G1TerrainEnvCfg,
)


@configclass
class G1FlatEnvCfg(G1TerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None


@configclass
class G1FlatPureEnvCfg(G1FlatEnvCfg):
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
class G1FlatDistillEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.future_frames = 10
        self.observations.proprioception_with_noise_wo_privilege.history_length = 8
        self.observations.proprioception.history_length = 8


@configclass
class G1FlatDualFSQEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.history_frames = 0
        self.commands.motion.future_frames = 9
        self.observations.proprioception_with_noise_wo_privilege.history_length = 8
        self.observations.proprioception.history_length = 8
