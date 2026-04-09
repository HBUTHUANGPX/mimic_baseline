from __future__ import annotations

"""Observation helpers for the tracking motion command.

The module preserves the original single-frame observation semantics and adds
new window-based observation functions. Window observations always follow the
time order `[t - n, ..., t, ..., t + m]`.
"""

import torch
from typing import TYPE_CHECKING

from general_motion_tracker_whole_body_teleoperation.tasks.tracking.mdp.commands import (
    MotionCommand,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _get_command(env: ManagerBasedEnv, command_name: str) -> MotionCommand:
    """Fetch the motion command term from the environment.

    Args:
        env: Environment instance.
        command_name: Name of the command term.

    Returns:
        The resolved :class:`MotionCommand` instance.
    """
    return env.command_manager.get_term(command_name)


def motion_id(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the motion id of the current center frame.

    Args:
        env: Environment instance.
        command_name: Name of the motion command term.

    Returns:
        Tensor containing the motion id for each environment.
    """
    return _get_command(env, command_name).motion_id


def motion_group(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the motion group id of the current center frame.

    Args:
        env: Environment instance.
        command_name: Name of the motion command term.

    Returns:
        Tensor containing the motion group id for each environment.
    """
    return _get_command(env, command_name).motion_group


def robot_ref_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the robot reference orientation matrix projected to 6D.

    Args:
        env: Environment instance.
        command_name: Name of the motion command term.

    Returns:
        Tensor with the first two columns of the robot reference rotation
        matrix, flattened per environment.
    """
    command = _get_command(env, command_name)
    mat = command._robot_ref_ori_w_mat
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_ref_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return robot reference linear velocity in world frame.

    Args:
        env: Environment instance.
        command_name: Name of the motion command term.

    Returns:
        Flattened linear velocity tensor per environment.
    """
    command = _get_command(env, command_name)
    return command.robot_ref_lin_vel_w[:, :3].view(env.num_envs, -1)


def robot_ref_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return robot reference angular velocity in world frame.

    Args:
        env: Environment instance.
        command_name: Name of the motion command term.

    Returns:
        Flattened angular velocity tensor per environment.
    """
    command = _get_command(env, command_name)
    return command.robot_ref_ang_vel_w[:, :3].view(env.num_envs, -1)


def robot_ref_vx_vy_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return planar linear velocity of the robot reference body.

    Args:
        env: Environment instance.
        command_name: Name of the motion command term.

    Returns:
        Flattened tensor containing `vx` and `vy`.
    """
    command = _get_command(env, command_name)
    return command.robot_ref_lin_vel_w[:, 0:2].view(env.num_envs, -1)


def robot_ref_wz_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return yaw angular velocity of the robot reference body.

    Args:
        env: Environment instance.
        command_name: Name of the motion command term.

    Returns:
        Flattened tensor containing the reference angular velocity.
    """
    command = _get_command(env, command_name)
    return command.robot_ref_ang_vel_w.view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return robot body positions in the current robot reference frame.

    Args:
        env: Environment instance.
        command_name: Name of the motion command term.

    Returns:
        Flattened body-position tensor per environment.
    """
    command = _get_command(env, command_name)
    return command._robot_body_pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return robot body orientations in 6D representation.

    Args:
        env: Environment instance.
        command_name: Name of the motion command term.

    Returns:
        Flattened 6D rotation representation per body and environment.
    """
    command = _get_command(env, command_name)
    return command._robot_body_ori_b_mat[..., :2].reshape(env.num_envs, -1)


def motion_ref_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return motion reference-body position relative to the current robot pose.

    Args:
        env: Environment instance.
        command_name: Name of the motion command term.

    Returns:
        Flattened position tensor per environment.
    """
    command = _get_command(env, command_name)
    return command._motion_ref_pos_b.view(env.num_envs, -1)


def motion_ref_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return motion reference-body orientation relative to the robot pose.

    Args:
        env: Environment instance.
        command_name: Name of the motion command term.

    Returns:
        Flattened 6D rotation representation per environment.
    """
    command = _get_command(env, command_name)
    return command._motion_ref_ori_b_mat[..., :2].reshape(env.num_envs, -1)


def joint_pos_delta(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return joint-position delta between motion center frame and robot state.

    Args:
        env: Environment instance.
        command_name: Name of the motion command term.

    Returns:
        Joint-position delta tensor per environment.
    """
    command = _get_command(env, command_name)
    return command.joint_pos - command.robot_joint_pos


def robot_joint_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return motion-command center-frame joint positions.

    Args:
        env: Environment instance.
        command_name: Name of the motion command term.

    Returns:
        Joint-position tensor per environment.
    """
    command = _get_command(env, command_name)
    return command.joint_pos


def joint_pos_delta_window(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return joint-position deltas for the full temporal window.

    Args:
        env: Environment instance.
        command_name: Name of the motion command term.

    Returns:
        Flattened tensor with time order `[t - n, ..., t, ..., t + m]`.
    """
    command = _get_command(env, command_name)
    return command._flatten_window(command._joint_pos_delta_window)


def robot_joint_pos_window(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return target joint positions for the full temporal window.

    The historical single-frame function `robot_joint_pos()` returns the target
    motion joint positions at the center frame. The window version extends that
    behavior to the full command window.

    Args:
        env: Environment instance.
        command_name: Name of the motion command term.

    Returns:
        Flattened target joint-position tensor in temporal order.
    """
    command = _get_command(env, command_name)
    return command._flatten_window(command.joint_pos_window)


def motion_ref_pos_b_window(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return motion reference-body positions for the full temporal window.

    Args:
        env: Environment instance.
        command_name: Name of the motion command term.

    Returns:
        Flattened tensor in temporal order.
    """
    command = _get_command(env, command_name)
    return command._flatten_window(command._motion_ref_pos_b_window)


def motion_ref_ori_b_window(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return motion reference-body orientations for the full temporal window.

    Args:
        env: Environment instance.
        command_name: Name of the motion command term.

    Returns:
        Flattened 6D rotation representation in temporal order.
    """
    command = _get_command(env, command_name)
    return command._flatten_window(command._motion_ref_ori_b_mat_window[..., :2])


def robot_body_pos_b_window(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return body positions for the full temporal window.

    The command module caches a window-aligned body-position tensor, so the
    window observation should consume that cache directly instead of repeating
    the center-frame tensor.

    Args:
        env: Environment instance.
        command_name: Name of the motion command term.

    Returns:
        Flattened body-position tensor in temporal order.
    """
    command = _get_command(env, command_name)
    return command._flatten_window(command._robot_body_pos_b_window)


def robot_body_ori_b_window(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return body orientations for the full temporal window.

    Args:
        env: Environment instance.
        command_name: Name of the motion command term.

    Returns:
        Flattened 6D rotation representation in temporal order.
    """
    command = _get_command(env, command_name)
    return command._flatten_window(command._robot_body_ori_b_mat_window[..., :2])
