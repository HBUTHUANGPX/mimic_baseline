from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat

from general_motion_tracker_whole_body_teleoperation.tasks.tracking_q1.mdp.commands import MotionCommand
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_body_names,
    record_dtype,
    record_joint_names,
    record_joint_pos_offsets,
    record_joint_vel_offsets,
    record_shape,
)
def motion_id(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.motion_id

def motion_group(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.motion_group

def robot_ref_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = command._robot_ref_ori_w_mat
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_ref_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_ref_lin_vel_w[:, :3].view(env.num_envs, -1)


def robot_ref_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_ref_ang_vel_w[:, :3].view(env.num_envs, -1)

def robot_ref_vx_vy_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_ref_lin_vel_w[:, 0:2].view(env.num_envs, -1) # tag

def robot_ref_wz_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_ref_ang_vel_w.view(env.num_envs, -1) # tag

def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command._robot_body_pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command._robot_body_ori_b_mat[..., :2].reshape(env.num_envs, -1)


def motion_ref_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command._motion_ref_pos_b.view(env.num_envs, -1)


def motion_ref_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command._motion_ref_ori_b_mat[..., :2].reshape(env.num_envs, -1)

def joint_pos_delta(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:

    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.joint_pos - command.robot_joint_pos
