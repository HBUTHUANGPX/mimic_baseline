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
from rsl_rl.runners import (
    DistillationRunner,
    OnPolicyRunner,
    MultiTeacherDistillationRunner,
)
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
from isaaclab_tasks.utils.hydra import hydra_task_config
import general_motion_tracker_whole_body_teleoperation.tasks  # noqa: F401
import yaml  # 导入PyYAML库
import re

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

from load_motion_file import collect_npz_paths

def get_checkpoint_path(
    log_path: str, run_dir: str = ".*", sort_alpha: bool = True
) -> List[str]:
    """Get paths to the latest model checkpoints in all motion subdirectories under the input run directory.

    The run directory is resolved as: ``<log_path>/<run_dir>``, where :attr:`run_dir` can be a regex expression.
    If :attr:`run_dir` is a regex, the most recent (highest alphabetical order) run is selected.

    Under the run directory, all subdirectories (assumed to be motion-named folders) are traversed.
    For each subdirectory, the latest checkpoint file (matching 'model_*.pt' and with the highest number) is selected.

    Args:
        log_path: The log directory path to find runs in.
        run_dir: The regex expression for the name of the run directory. Defaults to the most
            recent directory created inside :attr:`log_path`.
        sort_alpha: Whether to sort the runs by alphabetical order. Defaults to True.
            If False, the folders in :attr:`run_dir` are sorted by the last modified time.

    Returns:
        A list of paths to the latest model checkpoints in each motion subdirectory.

    Raises:
        ValueError: When no runs are found in the input directory.
        ValueError: When no checkpoints are found in a motion subdirectory.

    """
    # Find all runs in the directory that match the regex expression
    runs = []  # 初始化一个空列表
    print(f"[INFO]: Searching for runs in: '{log_path}' matching regex: '{run_dir}'")
    for run in os.scandir(log_path):  # 遍历log_path目录下的所有条目
        if run.is_dir() and re.match(
            run_dir, run.name
        ):  # 检查是否为目录且名称匹配正则表达式
            print(f"[INFO]: Found matching run: '{run.name}'")
            runs.append(
                os.path.join(log_path, run)
            )  # 如果条件满足，则将完整路径追加到列表中

    # Sort matched runs by alphabetical order (latest run should be last) or by modification time
    if sort_alpha:
        runs.sort()
    else:
        runs = sorted(runs, key=os.path.getmtime)

    # Select the latest run path
    try:
        run_path = runs[-1]
    except IndexError:
        raise ValueError(
            f"No runs present in the directory: '{log_path}' match: '{run_dir}'."
        )

    # Collect all motion subdirectories under the run path (exclude non-dirs and special dirs like 'params')
    motion_subdirs = []
    for sub in os.scandir(run_path):
        if (
            sub.is_dir() and sub.name != "params"
        ):  # 假设 'params' 非 motion 目录，可根据需要调整过滤
            motion_subdirs.append(os.path.join(run_path, sub.name))

    if len(motion_subdirs) == 0:
        raise ValueError(
            f"No motion subdirectories found in the run directory: '{run_path}'."
        )

    # For each motion subdirectory, find the latest checkpoint
    checkpoint_paths = []
    for subdir in motion_subdirs:
        # List all model checkpoints matching 'model_*.pt'
        model_checkpoints = [
            f for f in os.listdir(subdir) if re.match(r"model_.*\.pt", f)
        ]
        # Check if any checkpoints are present
        if len(model_checkpoints) == 0:
            raise ValueError(
                f"No checkpoints in the subdirectory: '{subdir}' match 'model_*.pt'."
            )
        # Sort alphabetically while ensuring that *_10 comes after *_9
        model_checkpoints.sort(key=lambda m: f"{m:0>15}")
        # Get latest matched checkpoint file
        checkpoint_file = model_checkpoints[-1]
        checkpoint_paths.append(os.path.join(subdir, checkpoint_file))

    return checkpoint_paths


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
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

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    # check for invalid combination of CPU device with distributed training
    if (
        args_cli.distributed
        and args_cli.device is not None
        and "cpu" in args_cli.device
    ):
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    motion_file_group = collect_npz_paths(args_cli.motion_file_path)
    for group_name, paths in motion_file_group.items():
        print(f"\nGroup: {group_name}")
        print(f"[INFO] Collected {len(paths)} motion files for training.")
    # print(motion_file)
    env_cfg.commands.motion.motion_file = motion_file_group
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "PPO_Distil":
        resume_path = get_checkpoint_path(
            "/home/hpx/HPX_LOCO_2/mimic_baseline/logs/rsl_rl/pure_q1_flat",
            agent_cfg.load_run,
        )
    print(f"[INFO]: Resuming training from checkpoint: {resume_path}")
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
        )
    elif agent_cfg.class_name == "DistillationRunner":
        print("[INFO]: Creating DistillationRunner")
        runner = DistillationRunner(
            env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
        )
    elif agent_cfg.class_name == "MultiTeacherDistillationRunner":
        print("[INFO]: Creating MultiTeacherDistillationRunner")
        motion_run_names = env.unwrapped.command_manager._terms[
            "motion"
        ].motion.group_names
        runner = MultiTeacherDistillationRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=log_dir,
            device=agent_cfg.device,
            motion_run_names=motion_run_names,
            teacher_names=resume_path,
        )
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # runner.alg = runner._construct_algorithm(obs)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "PPO_Distil":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path, map_location=agent_cfg.device)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
    )

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
