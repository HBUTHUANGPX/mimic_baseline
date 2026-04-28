from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from general_motion_tracker_whole_body_teleoperation.tasks.tracking.mdp.commands import MotionCommand
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def ref_human_anchor_rot6d_in_sim_anchor(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.ref_human_anchor_rot6d_in_sim_anchor

def sim_robot_anchor_rot6d_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.sim_robot_anchor_rot6d_w


def sim_robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.sim_robot_anchor_spatial_vel_w[:, :3].view(env.num_envs, -1)


def sim_robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.sim_robot_anchor_spatial_vel_w[:, 3:6].view(env.num_envs, -1)


def sim_robot_body_pos_in_sim_anchor(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.sim_robot_body_pos_in_sim_anchor.view(env.num_envs, -1)


def sim_robot_body_rot6d_in_sim_anchor(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.sim_robot_body_rot6d_in_sim_anchor


def ref_robot_anchor_pos_in_sim_anchor(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.ref_robot_anchor_pos_in_sim_anchor.view(env.num_envs, -1)


def ref_robot_anchor_rot6d_in_sim_anchor(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.ref_robot_anchor_rot6d_in_sim_anchor

def ref_robot_minus_sim_joint_angle_rad(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.ref_robot_minus_sim_joint_angle_rad

def sim_robot_joint_angle_rad(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.sim_robot_joint_angle_rad

def ref_motion_id(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.ref_motion_id

def ref_motion_group(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.ref_motion_group

def actor_ref_robot_fsq_feature_window(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.actor_ref_robot_fsq_feature_window

def actor_ref_human_fsq_feature_window(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.actor_ref_human_fsq_feature_window

def critic_ref_robot_fsq_feature_window(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.critic_ref_robot_fsq_feature_window

def critic_ref_human_fsq_feature_window(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.critic_ref_human_fsq_feature_window


def human_motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return ref_human_anchor_rot6d_in_sim_anchor(env, command_name)


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return sim_robot_anchor_rot6d_w(env, command_name)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return sim_robot_anchor_lin_vel_w(env, command_name)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return sim_robot_anchor_ang_vel_w(env, command_name)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return sim_robot_body_pos_in_sim_anchor(env, command_name)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return sim_robot_body_rot6d_in_sim_anchor(env, command_name)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return ref_robot_anchor_pos_in_sim_anchor(env, command_name)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return ref_robot_anchor_rot6d_in_sim_anchor(env, command_name)


def joint_pos_delta(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return ref_robot_minus_sim_joint_angle_rad(env, command_name)


def motion_id(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return ref_motion_id(env, command_name)


def motion_group(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return ref_motion_group(env, command_name)


def joint_pos_delta_window(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return ref_robot_minus_sim_joint_angle_rad(env, command_name)


def motion_anchor_ori_b_window(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return ref_robot_anchor_rot6d_in_sim_anchor(env, command_name)


def motion_anchor_pos_b_window(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return ref_robot_anchor_pos_in_sim_anchor(env, command_name)


def robot_fsq_window(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return actor_ref_robot_fsq_feature_window(env, command_name)


def human_fsq_window(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return actor_ref_human_fsq_feature_window(env, command_name)


def actor_robot_fsq_window(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return actor_ref_robot_fsq_feature_window(env, command_name)


def actor_human_fsq_window(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return actor_ref_human_fsq_feature_window(env, command_name)


def critic_robot_fsq_window(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return critic_ref_robot_fsq_feature_window(env, command_name)


def critic_human_fsq_window(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return critic_ref_human_fsq_feature_window(env, command_name)
