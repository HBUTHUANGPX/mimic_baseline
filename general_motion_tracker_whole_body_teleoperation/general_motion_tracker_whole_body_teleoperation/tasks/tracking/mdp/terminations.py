from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import SceneEntityCfg

from general_motion_tracker_whole_body_teleoperation.tasks.tracking.mdp.commands import MotionCommand
from general_motion_tracker_whole_body_teleoperation.tasks.tracking.mdp.rewards import _get_body_indexes


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.anchor_pos_error_norm > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_error[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return (
        command.motion_projected_gravity_b[:, 2]
        - command.robot_projected_gravity_b[:, 2]
    ).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = command.body_pos_error_norm[:, body_indexes]
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_error[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)

def reached_motion_end(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    # return command.reached_motion_end
    return command.time_steps >= command.motion.time_step_total

def bad_tracking_terminated(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.consecutive_bad_steps >= command.bad_steps_threshold