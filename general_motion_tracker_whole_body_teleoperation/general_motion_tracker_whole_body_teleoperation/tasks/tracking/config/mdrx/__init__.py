import gymnasium as gym

from . import agents, flat_env_cfg, terrain_env_cfg

PROBABILISTIC_TRACKING_ENTRY_POINT = (
    "general_motion_tracker_whole_body_teleoperation.tasks.tracking.probabilistic_tracking_env:"
    "ProbabilisticTrackingRLEnv"
)

##
# Register Gym environments.
##

gym.register(
    id="Tracking-Flat-MDRX-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.MDRXFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MDRXFlatPPORunnerCfg",
    },
)
gym.register(
    id="Tracking-Terrain-MDRX-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": terrain_env_cfg.MDRXTerrainEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MDRXFlatPPORunnerCfg",
    },
)
gym.register(
    id="Tracking-Terrain-MDRX-ProbTerm-v0",
    entry_point=PROBABILISTIC_TRACKING_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": terrain_env_cfg.MDRXTerrainEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MDRXFlatPPORunnerCfg",
    },
)
gym.register(
    id="Tracking-Terrain-MDRX-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": terrain_env_cfg.MDRXTerrainEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MDRXFlatPPORunnerCfg",
    },
)
gym.register(
    id="Tracking-Flat-MDRX-ProbTerm-v0",
    entry_point=PROBABILISTIC_TRACKING_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.MDRXFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MDRXFlatPPORunnerCfg",
    },
)
gym.register(
    id="Tracking-Flat-MDRX-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.MDRXFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MDRXFlatPPOSingleFSQRunnerCfg",
    },
)
gym.register(
    id="Tracking-Flat-MDRX-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.MDRXFlatPureEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MDRXFlatPPOPureRunnerCfg",
    },
)
gym.register(
    id="Tracking-Flat-MDRX-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.MDRXFlatDistillEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MDRXFlatPPODistillSingleFSQRunnerCfg",
    },
)
gym.register(
    id="Tracking-Flat-MDRX-v4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.MDRXFlatDualFSQEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MDRXFlatPPODualFSQRunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-MDRX-v5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.MDRXFlatDualFSQEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MDRXFlatPPODualTokenRunnerCfg",
    },
)
