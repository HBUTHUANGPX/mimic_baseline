from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

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
    mat = matrix_from_quat(command.robot_ref_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_ref_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_ref_vel_w[:, :3].view(env.num_envs, -1)


def robot_ref_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_ref_vel_w[:, 3:6].view(env.num_envs, -1)

def robot_ref_vx_vy_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_ref_lin_vel_w[:, 0:2].view(env.num_envs, -1)

def robot_ref_wz_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_ref_ang_vel_w[:, 5:6].view(env.num_envs, -1)

def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_ref_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, ori  = subtract_frame_transforms(
        command.robot_ref_pos_w,
        command.robot_ref_quat_w,
        command.ref_pos_w,
        command.ref_quat_w,
    )
    r1 = pos.view(env.num_envs, -1)
    mat = matrix_from_quat(ori)
    r2 = mat[..., :2].reshape(mat.shape[0], -1)
    return pos.view(env.num_envs, -1)


def motion_ref_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_ref_pos_w,
        command.robot_ref_quat_w,
        command.ref_pos_w,
        command.ref_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)

def joint_pos_delta(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:

    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.joint_pos - command.robot_joint_pos