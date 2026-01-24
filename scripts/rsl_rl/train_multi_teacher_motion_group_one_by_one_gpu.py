# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--max_iterations", type=int, default=None, help="RL Policy training iterations."
)
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Run training with multiple GPUs or nodes.",
)
parser.add_argument(
    "--export_io_descriptors",
    action="store_true",
    default=False,
    help="Export IO descriptors.",
)
parser.add_argument(
    "--ray-proc-id",
    "-rid",
    type=int,
    default=None,
    help="Automatically configured by Ray integration, otherwise None.",
)
parser.add_argument(
    "--motion_file_path",
    type=str,
    default="scripts/rsl_rl/motion_file.yaml",
    help="The name of the motion yaml file_path.",
)
parser.add_argument(
    '--group_name',          # 参数名
    type=str,            # 明确指定类型为 str（实际上不写 type 也默认是 str）
    required=False,      # 是否必须传入（可选）
    default=None,        # 默认值
    help='group_name将使用指定的groupname'
)
parser.add_argument(
    '--time_stamp',          # 参数名
    type=str,            # 明确指定类型为 str（实际上不写 type 也默认是 str）
    required=False,      # 是否必须传入（可选）
    default=None,        # 默认值
    help='time_stamp将指定使用共享的事件戳作为log保存的子路径'
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [
            r".\isaaclab.bat",
            "-p",
            "-m",
            "pip",
            "install",
            f"rsl-rl-lib=={RSL_RL_VERSION}",
        ]
    else:
        cmd = [
            "./isaaclab.sh",
            "-p",
            "-m",
            "pip",
            "install",
            f"rsl-rl-lib=={RSL_RL_VERSION}",
        ]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import logging
import os
import time
import torch
from datetime import datetime
import glob
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from typing import List
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import general_motion_tracker_whole_body_teleoperation.tasks  # noqa: F401
import yaml  # 导入PyYAML库

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

from load_motion_file import collect_npz_paths
'''
这个脚本是用来训练教师模型的
    ‘train_multi_teacher.py’用于多个卡一起进行一个training任务，并且将对‘motion_file.yaml’中motion_group描述的group逐个进行训练。
    这个脚本与之区别在于，每张卡独立进行training任务，并且将每个卡上跑的任务将依次对‘motion_file.yaml’中motion_group描述的group逐个进行训练。

    TODO:
    1. 修改rsl-rl库的log
'''
@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    print(f"motion:{args_cli.group_name}")
    print(f"time_stamp:{args_cli.time_stamp}")
    specify_group_name = args_cli.group_name
    motion_file_group = collect_npz_paths(args_cli.motion_file_path)
    for group_name, paths in motion_file_group.items():
        print(f"\nGroup: {group_name}")
        print(f"[INFO] Collected {len(paths)} motion files for training.")
    # print(motion_file)

    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    agent_cfg.max_iterations = (
        args_cli.max_iterations
        if args_cli.max_iterations is not None
        else agent_cfg.max_iterations
    )
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    
    env_cfg.commands.motion.motion_file = {next(iter(motion_file_group)): next(iter(motion_file_group.values()))}

    # create isaac environment
    _env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    env = RslRlVecEnvWrapper(_env, clip_actions=agent_cfg.clip_actions) # wrap around environment for rsl-rl
    print(f"[INFO] wrap around environment for rsl-rl")

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(
            "/home/hpx/HPX_LOCO_2/mimic_baseline/logs/rsl_rl/pure_q1_flat", agent_cfg.load_run, agent_cfg.load_checkpoint
        )
    print(f"[INFO] save resume path before creating a new log_dir")

    start_time = time.time()

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    elif agent_cfg.class_name == "DistillationRunner":
        print("[INFO]: Creating DistillationRunner")
        runner = DistillationRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    print(f"[INFO] create runner from rsl-rl")
    
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
    print(f"[INFO] write git state to logs,load the checkpoint")
    
    from rsl_rl.utils import resolve_obs_groups

    for group_name, paths in motion_file_group.items():
        if specify_group_name == group_name:
            ...
        else:
            print(f"Group: {group_name} pass")
            continue
        env.unwrapped.command_manager.cfg.motion.motion_file = {group_name: paths}
        env.unwrapped.command_manager._terms['motion'].load_motion({group_name: paths})
        print(f"Group: {group_name} has {len(paths)} motion files.")
        if agent_cfg.run_name:
            log_dir = os.path.join(log_root_path, args_cli.time_stamp+f"_{agent_cfg.run_name}", group_name)
        else:
            log_dir = os.path.join(log_root_path, args_cli.time_stamp,group_name)
        
        # dump the configuration into log-directory
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        print(f"[INFO] dump the configuration into log-directory")

        runner.cfg = agent_cfg.to_dict()
        runner.policy_cfg = agent_cfg.to_dict()["policy"]
        runner.alg_cfg = agent_cfg.to_dict()["algorithm"]
        _obs = runner.env.get_observations()
        runner.cfg["obs_groups"] = resolve_obs_groups(_obs, runner.cfg["obs_groups"], runner._get_default_obs_sets())
        runner.alg = runner._construct_algorithm(_obs)
        runner.init_logger(log_dir)
        # write git state to logs
        runner.add_git_repo_to_log(__file__)
        runner.current_learning_iteration = 0
        # run training
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
        )
        print(f"Training time: {round(time.time() - start_time, 2)} seconds")
        if not runner.logger.disable_logs:
            if runner.logger.logger_type == "wandb":
                runner.logger.writer.stop()
        # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
