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
from isaaclab.sensors import ContactSensorCfg,RayCasterCfg,patterns
from isaaclab.terrains import TerrainImporterCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
import general_motion_tracker_whole_body_teleoperation.tasks.tracking.mdp as mdp

##
# Scene definition
##

VELOCITY_RANGE = {
    "x": (-1.2, 1.2),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=200.0,
    border_height=0.0,
    
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            # mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
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
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=False,
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            # "x": (-0.0, 0.0),
            # "y": (-0.0, 0.0),
            # "z": (-0.0, 0.0),
            # "roll": (-0., 0.),
            # "pitch": (-0., 0.),
            # "yaw": (-0., 0.),
            "x": (-0.1, 0.1),
            "z": (0.05, 0.1),
            "y": (-0.1, 0.1),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        # joint_position_range=(-0., 0.),
        # joint_velocity_range=(-0., 0.),
        joint_position_range=(-0.1, 0.1),
        # joint_velocity_range=(-0.1, 0.1),
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
    class CommandAllCfg(ObsGroup):  # 有噪 特权 window cmd
        """Observations for command group with noise."""

        joint_pos_delta = ObsTerm(# 
            func=mdp.joint_pos_delta,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        joint_pos_delta_window = ObsTerm(
            func=mdp.joint_pos_delta_window,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )

        motion_anchor_pos_b = ObsTerm(# robot motion anchor translation in world frame
            func=mdp.motion_anchor_pos_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        motion_anchor_pos_b_window = ObsTerm(# with future window frame
            func=mdp.motion_anchor_pos_b_window,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )

        motion_anchor_ori_b = ObsTerm(# robot motion anchor orientation in world frame
            func=mdp.motion_anchor_ori_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        motion_anchor_ori_b_window = ObsTerm(# with future window frame
            func=mdp.motion_anchor_ori_b_window,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        robot_body_pos = ObsTerm(# robot motion key body's translation in anchor frame
            func=mdp.robot_body_pos_b, 
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )
        robot_body_ori = ObsTerm(# robot motion key body's orientation in anchor frame
            func=mdp.robot_body_ori_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class ProprioceptionAllCfg(ObsGroup):  # 有噪 特权 本体
        """Observations for proprioception group with noise."""

        root_pos_w = ObsTerm(
            func=mdp.root_pos_w, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        robot_anchor_ori_w = ObsTerm(
            func=mdp.robot_anchor_ori_w,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
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

    @configclass
    class RobotFSQWindowCfg(ObsGroup):
        robot_fsq_window = ObsTerm(func=mdp.robot_fsq_window, params={"command_name": "motion"})

        def __post_init__(self):
            self.enable_corruption = False

    @configclass
    class HumanFSQWindowCfg(ObsGroup):
        human_fsq_window = ObsTerm(func=mdp.human_fsq_window, params={"command_name": "motion"})

        def __post_init__(self):
            self.enable_corruption = False

    command_window_with_noise_wo_privilege: CommandAllCfg = (
        CommandAllCfg()
    )  # 有噪 无特权 cmd
    command_window_with_noise_wo_privilege.joint_pos_delta = None
    command_window_with_noise_wo_privilege.motion_anchor_pos_b = None
    command_window_with_noise_wo_privilege.motion_anchor_ori_b = None
    command_window_with_noise_wo_privilege.motion_anchor_pos_b_window = None
    command_window_with_noise_wo_privilege.robot_body_pos = None
    command_window_with_noise_wo_privilege.robot_body_ori = None

    command_with_noise_wo_privilege: CommandAllCfg = (
        CommandAllCfg()
    )  # 有噪 无特权 cmd
    command_with_noise_wo_privilege.joint_pos_delta_window = None
    command_with_noise_wo_privilege.motion_anchor_pos_b_window = None
    command_with_noise_wo_privilege.motion_anchor_ori_b_window = None
    command_with_noise_wo_privilege.motion_anchor_pos_b = None
    command_with_noise_wo_privilege.robot_body_pos = None
    command_with_noise_wo_privilege.robot_body_ori = None

    proprioception_with_noise_wo_privilege: ProprioceptionAllCfg = (
        ProprioceptionAllCfg()
    )  # 有噪 无特权 本体
    proprioception_with_noise_wo_privilege.base_lin_vel = None
    proprioception_with_noise_wo_privilege.root_pos_w = None

    command_window: CommandAllCfg = CommandAllCfg(enable_corruption = False)  # 无噪 特权 cmd 
    command_window.joint_pos_delta = None
    command_window.motion_anchor_pos_b = None
    command_window.motion_anchor_ori_b = None

    command: CommandAllCfg = CommandAllCfg(enable_corruption = False)  # 无噪 特权 cmd
    command.joint_pos_delta_window = None
    command.motion_anchor_ori_b_window = None
    command.motion_anchor_pos_b_window = None
    proprioception: ProprioceptionAllCfg = ProprioceptionAllCfg(enable_corruption = False)  # 无噪 特权 本体

    last_action: LastActionCfg = LastActionCfg()

    motion_id: MotionIdCfg = MotionIdCfg()
    motion_group: MotionGroupCfg = MotionGroupCfg()
    robot_fsq_window: RobotFSQWindowCfg = RobotFSQWindowCfg()
    human_fsq_window: HumanFSQWindowCfg = HumanFSQWindowCfg()


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
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=2.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
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
    termination = RewTerm(
        func=mdp.is_terminated,
        weight=-200,
    )
    # global_anchor_position_error_z = RewTerm(
    #     func=mdp.motion_global_anchor_position_z_error_sum_square,
    #     weight=-2.0,
    #     params={
    #         "command_name": "motion",
    #     },
    # )
    # xy_anchor_movement_in_recovering = RewTerm(
    #     func=mdp.xy_anchor_movement_in_recovering,
    #     weight=-1.0,
    #     params={
    #         "command_name": "motion",
    #     },
    # )
    # action_rate_l2_in_recovering = RewTerm(
    #     func=mdp.action_rate_l2_in_recovering,
    #     weight=-2.0,
    #     params={
    #         "command_name": "motion",
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # bad_tracking_terminate = DoneTerm(
    #     func=mdp.bad_tracking_terminated,
    #     params={"command_name": "motion"},
    # )
    ref_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    ref_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "motion",
            "threshold": 0.8,
        },
    )
    ee_body_pos_knee = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.28,
            "body_names": [
                "L_knee_link",
                "R_knee_link",
            ],
        },
    )
    ee_body_pos_ankle = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.35,
            "body_names": [
                "L_ankle_roll_link",
                "R_ankle_roll_link",
            ],
        },
    )

    ee_body_pos_wrist = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "L_wrist_pitch_link",
                "R_wrist_pitch_link",
            ],
        },
    )
    # reach_motion_clip_end = DoneTerm(
    #     func=mdp.reached_motion_end, params={"command_name": "motion"}
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    # scene: MySceneCfg = MySceneCfg(num_envs=64, env_spacing=2.5)
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # scene: MySceneCfg = MySceneCfg(num_envs=4096 * 4, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        # self.decimation = 4
        # self.sim.dt = 0.005

        self.decimation = 1
        self.sim.dt = 0.02

        self.observations.proprioception_with_noise_wo_privilege.history_length = 8
        self.commands.motion.future_frames = 10
        # self.decimation = 20
        # self.sim.dt = 0.001
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**16
        # viewer settings
        # self.viewer.eye = (3, 3, 1.5)
        # self.viewer.origin_type = "asset_root"
        # self.viewer.asset_name = "robot"
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
