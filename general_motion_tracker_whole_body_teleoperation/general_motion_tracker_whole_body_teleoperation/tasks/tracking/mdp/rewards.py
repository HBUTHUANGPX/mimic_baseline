from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from general_motion_tracker_whole_body_teleoperation.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_error), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = command.anchor_rot_error**2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_error[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def motion_global_anchor_position_z_error_sum_square(
    env: ManagerBasedRLEnv, command_name: str,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_error[..., 2:3]), dim=-1)
    return error

def xy_anchor_movement_in_recovering(
    env: ManagerBasedRLEnv, command_name: str,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    xy_vel_sum = torch.sum(torch.square(command.robot_anchor_lin_vel_w[:, :2]), dim=-1)
    return command.is_recovering*xy_vel_sum

def action_rate_l2_in_recovering(env: ManagerBasedRLEnv, command_name: str,) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.is_recovering*torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)

def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = command.body_rot_error[:, body_indexes] ** 2
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_error[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_error[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward

def foot_contact_velocity(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    clip: float,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    # 用于检测足端与地面接触时的速度，希望与地面接触时，速度尽可能为0
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)

    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = (
        torch.max(
            torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1
        )[0]
        > threshold
    )
    foot_vel = torch.square(command.robot_body_lin_vel_w[:, body_indexes]).sum(
        -1
    )  # tag
    clipped_vel = torch.clip(foot_vel, min=0, max=clip)
    r = ((torch.exp(clipped_vel) - 1.0)) * is_contact

    return r.sum(1)
