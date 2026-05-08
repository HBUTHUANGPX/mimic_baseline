# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

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
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
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
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)
parser.add_argument(
    "--motion_file_path",
    type=str,
    default="scripts/rsl_rl/motion_file.yaml",
    help="The name of the motion yaml file_path.",
)
parser.add_argument(
    "--domain_randomization",
    action="store_true",
    default=False,
    help="Enable domain randomization during evaluation.",
)
parser.add_argument(
    "--other_dirs",
    type=str,
    default=None,
    help="Comma-separated list of other directories to include.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args = parser.parse_args()
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import glob
from typing import List

from rsl_rl.runners import DistillationRunner, OnPolicyRunner, OnPolicyRunnerFSQ, OnPolicyDisstillationRunnerFSQ

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import general_motion_tracker_whole_body_teleoperation.tasks  # noqa: F401
from general_motion_tracker_whole_body_teleoperation.utils.exporter import (
    attach_onnx_metadata,
    export_motion_policy_as_onnx,
)

# PLACEHOLDER: Extension template (do not remove this comment)

from load_motion_file import collect_npz_paths

import onnxruntime as ort
import numpy as np


def _to_onnx_input(value, expected_rank: int = 2):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float32)
    if expected_rank == 2 and value.ndim == 1:
        value = value[None, :]
    return value


def _run_onnx_inference(session, actor_obs, human_fsq_obs, robot_fsq_obs, selector):
    actor_obs = _to_onnx_input(actor_obs)
    human_fsq_obs = _to_onnx_input(human_fsq_obs)
    robot_fsq_obs = _to_onnx_input(robot_fsq_obs)
    selector = _to_onnx_input(selector)

    actor_obs_name = session.get_inputs()[0].name
    human_fsq_obs_name = session.get_inputs()[1].name
    robot_fsq_obs_name = session.get_inputs()[2].name
    selector_name = session.get_inputs()[3].name

    (
        actions,q_human,q_robot
    ) = session.run(
        None,
        {
            actor_obs_name: actor_obs,
            human_fsq_obs_name: human_fsq_obs,
            robot_fsq_obs_name: robot_fsq_obs,
            selector_name: selector,
        },
    )
    return actions,q_human,q_robot

def _onnx_policy_reasoning(actor_obs, human_fsq_obs, robot_fsq_obs, selector, onnx_policy):
    (
        act,q_human,q_robot
    ) = _run_onnx_inference(
        onnx_policy, actor_obs, human_fsq_obs, robot_fsq_obs, selector
    )
    return act,q_human,q_robot


def _load_onnx_model(onnx_path, device="cpu"):
    providers = (
        ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]
    )
    session = ort.InferenceSession(onnx_path, providers=providers)
    return session

@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")
    retain_events = ['reset_robot']  # 指定要保留的事件名称列表
    if args_cli.domain_randomization:
        # 获取所有事件参数，并剔除非保留项
        for event_name in dir(env_cfg.events):
            print(f"[INFO] Checking event: {event_name}")
            if not event_name.startswith('_') and event_name not in retain_events:
                setattr(env_cfg.events, event_name, None)
    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print(
                "[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task."
            )
            return
    # elif args_cli.checkpoint:
    #     resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        if args.other_dirs is not None:
            resume_path = get_checkpoint_path(
                log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint, other_dirs=[args_cli.other_dirs]
            )   
        else:
            resume_path = get_checkpoint_path(
                log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
            )
    print(f"[INFO] Resuming from checkpoint: {resume_path}")

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir
    motion_file_group = collect_npz_paths(args_cli.motion_file_path)
    for group_name, paths in motion_file_group.items():
        print(f"\nGroup: {group_name}")
        print(f"[INFO] Collected {len(paths)} motion files for training.")
    env_cfg.commands.motion.motion_file = motion_file_group
    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        _runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    elif agent_cfg.class_name == "OnPolicyRunnerFSQ":
        _runner = OnPolicyRunnerFSQ(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    elif agent_cfg.class_name == "OnPolicyDisstillationRunnerFSQ":
        _runner = OnPolicyDisstillationRunnerFSQ(
            env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
        )
    elif agent_cfg.class_name == "DistillationRunner":
        _runner = DistillationRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner: OnPolicyRunner | OnPolicyRunnerFSQ | OnPolicyDisstillationRunnerFSQ | DistillationRunner = _runner
    if agent_cfg.class_name == "OnPolicyDisstillationRunnerFSQ":
        runner.load(resume_path, map_location=agent_cfg.device,is_eval = True)
    else:
        runner.load(resume_path, map_location=agent_cfg.device)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if (
        runner.alg.policy.__class__.__name__ == "ActorCriticSingleFSQ"
        or runner.alg.policy.__class__.__name__ == "ActorCriticSingleFSQDistillation"
        or runner.alg.policy.__class__.__name__ == "ActorCriticDualFSQ"
    ):
        _policy = runner.alg.policy
        _policy.export_policy_as_onnx(
            env,
            path=export_model_dir,
            filename="policy.onnx",
            verbose=False,
        )
    else:
        export_motion_policy_as_onnx(
            env.unwrapped,
            policy_nn,
            normalizer=normalizer,
            path=export_model_dir,
            filename="policy.onnx",
        )
    onnx_policy = _load_onnx_model(
        os.path.join(export_model_dir, "policy.onnx"), device=agent_cfg.device
    )

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs,only_action=True)
            human_fsq_obs = _policy.get_actor_human_fsq_obs(obs)
            robot_fsq_obs = _policy.get_actor_robot_fsq_obs(obs)
            actor_obs = _policy.get_actor_obs(obs)
            selector = torch.zeros(1, 1, device=human_fsq_obs.device)
            act,q_human,q_robot = _onnx_policy_reasoning(
                actor_obs[0:1, :],
                human_fsq_obs[0:1, :],
                robot_fsq_obs[0:1, :],
                selector,
                onnx_policy,
            )
            # print("================================")
            # print("human_fsq_obs:\r\n",human_fsq_obs[0:1, :].tolist())
            # print("robot_fsq_obs:\r\n",robot_fsq_obs[0:1, :].tolist())
            # print("q_human:\r\n",q_human)
            # print("q_robot:\r\n",q_robot)
            actions[0, :] = torch.from_numpy(act)

            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
