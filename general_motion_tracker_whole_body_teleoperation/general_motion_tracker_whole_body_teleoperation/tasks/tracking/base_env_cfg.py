from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import general_motion_tracker_whole_body_teleoperation.tasks.tracking.mdp as mdp

##
# Scene definition
##
VELOCITY_RANGE_ZERO = {
    "x": (-0.0, 0.0),
    "y": (-0.0, 0.0),
    "z": (-0.0, 0.0),
    "roll": (-0.0, 0.0),
    "pitch": (-0.0, 0.0),
    "yaw": (-0.0, 0.0),
}
VELOCITY_RANGE_MID = {
    "x": (-1.2, 1.2),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}
VELOCITY_RANGE_HIGH = {
    "x": (-2.4, 2.4),
    "y": (-1.2, 1.2),
    "z": (-0.5, 0.5),
    "roll": (-1.52, 01.52),
    "pitch": (-1.52, 1.52),
    "yaw": (-1.78, 1.78),
}


POSE_RANGE_ZERO = {
    "x": (-0.0, 0.0),
    "y": (-0.0, 0.0),
    "z": (-0.0, 0.0),
    "roll": (-0.0, 0.0),
    "pitch": (-0.0, 0.0),
    "yaw": (-0.0, 0.0),
}
POSE_RANGE_NORMAL = {
    "x": (-0.1, 0.1),
    "y": (-0.1, 0.1),
    "z": (-0.0, 0.2),
    "roll": (-0.1, 0.1),
    "pitch": (-0.1, 0.1),
    "yaw": (-0.2, 0.2),
}
POSE_RANGE_HIGH = {
    "x": (-0.2, 0.2),
    "y": (-0.2, 0.2),
    "z": (-0.0, 0.3),
    "roll": (-0.2, 0.2),
    "pitch": (-0.2, 0.2),
    "yaw": (-0.3, 0.3),
}


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=True,
    )


##
# MDP settings
##

JOINTS_POSITION_RANGE_NORMAL = (-0.1, 0.1)
JOINTS_POSITION_RANGE_HIGH = (-0.2, 0.2)
JOINTS_POSITION_RANGE_ZERO = (-0.0, 0.0)


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range=POSE_RANGE_NORMAL,
        velocity_range=VELOCITY_RANGE_MID,
        joint_position_range=JOINTS_POSITION_RANGE_NORMAL,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "motion"}
        )
        motion_ref_ori_b = ObsTerm(
            func=mdp.motion_ref_ori_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "motion"}
        )
        motion_ref_pos_b = ObsTerm(
            func=mdp.motion_ref_pos_b, params={"command_name": "motion"}
        )
        motion_ref_ori_b = ObsTerm(
            func=mdp.motion_ref_ori_b, params={"command_name": "motion"}
        )
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

    @configclass
    class ProprioceptionWithNoiseCfg(ObsGroup):  # 有噪 特权 本体
        """Observations for proprioception group with noise."""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.25, n_max=0.25)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.015, n_max=0.015)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.55, n_max=0.55)
        )

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class ProprioceptionWithNoiseWOPrivilegeCfg(ObsGroup):  # 有噪 无特权 本体
        """Observations for proprioception group with noise."""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.015, n_max=0.015)
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.8, n_max=0.8))

        def __post_init__(self):
            self.enable_corruption = True
            # self.history_length = 8

    @configclass
    class ProprioceptionCfg(ObsGroup):  # 无噪 特权 本体
        """Observations for proprioception group without noise."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = True
            # self.history_length = 8

    @configclass
    class ProprioceptionWOPrivilegeCfg(ObsGroup):  # 无噪 无特权 本体
        """Observations for proprioception group without noise."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = True
            # self.history_length = 8

    @configclass
    class CommandWithNoiseCfg(ObsGroup):  # 有噪 特权 cmd
        """Observations for command group with noise."""

        joint_pos_delta = ObsTerm(
            func=mdp.joint_pos_delta,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        target_joint_pos = ObsTerm(
            func=mdp.robot_joint_pos,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        motion_ref_pos_b = ObsTerm(
            func=mdp.motion_ref_pos_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        motion_ref_ori_b = ObsTerm(
            func=mdp.motion_ref_ori_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        body_pos = ObsTerm(
            func=mdp.robot_body_pos_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )
        body_ori = ObsTerm(
            func=mdp.robot_body_ori_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class CommandWithNoiseWOPrivilegeCfg(ObsGroup):  # 有噪 无特权 cmd
        """Observations for command group with noise."""

        joint_pos_delta = ObsTerm(
            func=mdp.joint_pos_delta,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        target_joint_pos = ObsTerm(
            func=mdp.robot_joint_pos,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        motion_ref_ori_b = ObsTerm(
            func=mdp.motion_ref_ori_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class CommandCfg(ObsGroup):  # 无噪 特权 cmd
        """Observations for command group with noise."""

        joint_pos_delta = ObsTerm(
            func=mdp.joint_pos_delta, params={"command_name": "motion"}
        )
        target_joint_pos = ObsTerm(
            func=mdp.robot_joint_pos,
            params={"command_name": "motion"},
        )
        motion_ref_pos_b = ObsTerm(
            func=mdp.motion_ref_pos_b,
            params={"command_name": "motion"},
        )
        motion_ref_ori_b = ObsTerm(
            func=mdp.motion_ref_ori_b,
            params={"command_name": "motion"},
        )
        body_pos = ObsTerm(
            func=mdp.robot_body_pos_b,
            params={"command_name": "motion"},
        )
        body_ori = ObsTerm(
            func=mdp.robot_body_ori_b,
            params={"command_name": "motion"},
        )

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class CommandWOPrivilegeCfg(ObsGroup):  # 无噪 无特权 cmd
        """Observations for command group with noise."""

        joint_pos_delta = ObsTerm(
            func=mdp.joint_pos_delta, params={"command_name": "motion"}
        )

        target_joint_pos = ObsTerm(
            func=mdp.robot_joint_pos, params={"command_name": "motion"}
        )
        motion_ref_ori_b = ObsTerm(
            func=mdp.motion_ref_ori_b,
            params={"command_name": "motion"},
        )

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class CommandWindowWithNoiseWOPrivilegeCfg(ObsGroup):  # 有噪 无特权 cmd
        """Observations for command group with noise."""

        joint_pos_delta = ObsTerm(
            func=mdp.joint_pos_delta_window,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        target_joint_pos = ObsTerm(
            func=mdp.robot_joint_pos_window,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        motion_ref_ori_b = ObsTerm(
            func=mdp.motion_ref_ori_b_window,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class CommandWindowCfg(ObsGroup):  # 无噪 特权 cmd
        """Observations for command group with noise."""

        joint_pos_delta = ObsTerm(
            func=mdp.joint_pos_delta_window, params={"command_name": "motion"}
        )
        target_joint_pos = ObsTerm(
            func=mdp.robot_joint_pos_window,
            params={"command_name": "motion"},
        )
        motion_ref_pos_b = ObsTerm(
            func=mdp.motion_ref_pos_b_window,
            params={"command_name": "motion"},
        )
        motion_ref_ori_b = ObsTerm(
            func=mdp.motion_ref_ori_b_window,
            params={"command_name": "motion"},
        )
        body_pos = ObsTerm(
            func=mdp.robot_body_pos_b_window,
            params={"command_name": "motion"},
        )
        body_ori = ObsTerm(
            func=mdp.robot_body_ori_b_window,
            params={"command_name": "motion"},
        )

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class LastActionCfg(ObsGroup):  # 不带噪声的上一个动作观测组
        """Observations for last action group."""

        actions = ObsTerm(func=mdp.last_action)

    @configclass
    class MotionIdCfg(ObsGroup):  # 不带噪声的上一个动作观测组
        """Observations for last action group."""

        motion_id = ObsTerm(func=mdp.motion_id, params={"command_name": "motion"})

    @configclass
    class MotionGroupCfg(ObsGroup):  # 不带噪声的上一个动作观测组
        """Observations for last action group."""

        motion_group = ObsTerm(func=mdp.motion_group, params={"command_name": "motion"})

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()

    # Use the window-backed command observations. When history_frames and
    # future_frames are both zero, these terms reduce exactly to the original
    # single-frame observations and can be used to validate correctness before
    # enabling larger temporal windows.
    command_with_noise_wo_privilege: CommandWindowWithNoiseWOPrivilegeCfg = (
        CommandWindowWithNoiseWOPrivilegeCfg()
    )  # 有噪 无特权 cmd
    # command_with_noise_wo_privilege : CommandWithNoiseWOPrivilegeCfg = CommandWithNoiseWOPrivilegeCfg()  # 有噪 无特权 cmd
    proprioception_with_noise_wo_privilege: ProprioceptionWithNoiseWOPrivilegeCfg = (
        ProprioceptionWithNoiseWOPrivilegeCfg()
    )  # 有噪 无特权 本体

    command: CommandWindowCfg = CommandWindowCfg()  # 无噪 特权 cmd
    # command: CommandCfg = CommandCfg()  # 无噪 特权 cmd
    proprioception: ProprioceptionCfg = ProprioceptionCfg()  # 无噪 特权 本体
    last_action: LastActionCfg = LastActionCfg()

    motion_id: MotionIdCfg = MotionIdCfg()
    motion_group: MotionGroupCfg = MotionGroupCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.1, 1.6),
            "dynamic_friction_range": (0.1, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.06, 0.06), "y": (-0.025, 0.025), "z": (-0.01, 0.05)},
        },
    )
    pelvis_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis_link"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.02, 0.02), "z": (0.01, 0.01)},
        },
    )
    knee_link_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["L_knee_link", "R_knee_link"]
            ),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.03, 0.03)},
        },
    )
    robot_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.92, 1.08),
            "operation": "scale",
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",  # startup 和 reset 的训练结构没什么区别，反而 reset 会增加训练时间
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (1 / 2.0, 2.0),
            "damping_distribution_params": (1 / 2.0, 2.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE_MID},
    )

    # reset robot
    reset_robot = EventTerm(
        func=mdp.reset_robot_state_by_motioncommand,
        mode="reset",
        params={
            "command_name": "motion",
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    motion_global_root_pos = RewTerm(
        func=mdp.motion_global_ref_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_root_ori = RewTerm(
        func=mdp.motion_global_ref_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    extern_motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.081},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_torques_limit = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!L_ankle_roll_link$)(?!R_ankle_roll_link$)(?!L_wrist_pitch_link$)(?!R_wrist_pitch_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )
    foot_contact_velocity = RewTerm(
        func=mdp.foot_contact_velocity,
        weight=-0.1,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_ankle_roll_link"],
            ),
            "command_name": "motion",
            "clip": 2.0**2,
            "body_names": ["L_ankle_roll_link", "R_ankle_roll_link"],
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    ref_pos = DoneTerm(
        func=mdp.bad_ref_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    ref_ori = DoneTerm(
        func=mdp.bad_ref_ori,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "motion",
            "threshold": 0.8,
        },
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "L_ankle_roll_link",
                "R_ankle_roll_link",
                "L_wrist_pitch_link",
                "R_wrist_pitch_link",
            ],
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass
