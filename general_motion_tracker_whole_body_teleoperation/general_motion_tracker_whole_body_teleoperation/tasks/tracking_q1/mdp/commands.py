from __future__ import annotations

import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
import re


def extract_part(path):
    # 使用正则匹配 '/([^:/]+):' 模式，捕获组1为所需部分
    match = re.search(r"/([^:/]+):", path)
    if match:
        return match.group(1)
    return None


class MotionLoader:
    def __init__(
        self,
        motion_file: str | Sequence[str],
        body_indexes: Sequence[int],
        device: str = "cpu",
    ):
        # 支持单个字符串或列表，统一转换为列表
        if isinstance(motion_file, str):
            self.motion_file = [motion_file]
        else:
            self.motion_file = motion_file
        for file in self.motion_file:
            assert os.path.isfile(file), f"Invalid file path: {file}"
        self.extracted_list = [
            extract_part(p) for p in self.motion_file if extract_part(p) is not None
        ]
        self.num_motions = len(self.motion_file)
        assert self.num_motions > 0, "At least one motion file is required."

        # Load and concatenate data from all files
        joint_pos_list = []
        joint_vel_list = []
        body_pos_w_list = []
        body_quat_w_list = []
        body_lin_vel_w_list = []
        body_ang_vel_w_list = []
        self.motion_lengths = []  # Length of each motion segment
        self.fps = None  # Assume all files have the same fps

        for _file in self.motion_file:
            data = np.load(_file)
            if self.fps is None:
                self.fps = data["fps"]
            else:
                assert (
                    self.fps == data["fps"]
                ), "All motion files must have the same fps."

            joint_pos_list.append(
                torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
            )
            joint_vel_list.append(
                torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
            )
            body_pos_w_list.append(
                torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
            )
            body_quat_w_list.append(
                torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
            )
            body_lin_vel_w_list.append(
                torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
            )
            body_ang_vel_w_list.append(
                torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
            )
            self.motion_lengths.append(data["joint_pos"].shape[0])

        # Concatenate along time dimension (dim=0)
        self.joint_pos = torch.cat(joint_pos_list, dim=0)
        self.joint_vel = torch.cat(joint_vel_list, dim=0)
        self._body_pos_w = torch.cat(body_pos_w_list, dim=0)
        self._body_quat_w = torch.cat(body_quat_w_list, dim=0)
        self._body_lin_vel_w = torch.cat(body_lin_vel_w_list, dim=0)
        self._body_ang_vel_w = torch.cat(body_ang_vel_w_list, dim=0)

        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

        # New: new_data_flag (bool tensor, True at start of each new segment except the first)
        self.new_data_flag = torch.zeros(
            self.time_step_total, dtype=torch.bool, device=device
        )  # torch.Size([time_step_total])
        cumulative_len = 0
        for i, length in enumerate(self.motion_lengths):
            if i > 0:  # Skip the first segment
                self.new_data_flag[cumulative_len] = True
            cumulative_len += length

        # New: motion_indices (num_motions, 2), [start, end] exclusive end
        self.motion_indices = torch.zeros(
            self.num_motions, 2, dtype=torch.long, device=device
        )  # torch.Size([num_motions, 2])
        start = 0
        for i, length in enumerate(self.motion_lengths):
            end = start + length
            self.motion_indices[i] = torch.tensor(
                [start, end], dtype=torch.long, device=device
            )
            start = end

        # New: motion_distribution (1, num_motions), initialized uniformly
        self.motion_distribution = torch.full(
            (1, self.num_motions),
            1.0 / self.num_motions,
            dtype=torch.float32,
            device=device,
        )  # torch.Size([1, num_motions])

        # Target distribution based on lengths
        total_length = sum(self.motion_lengths)
        self.target_dist = torch.tensor(
            [length / total_length for length in self.motion_lengths],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(
            0
        )  # torch.Size([1, num_motions])

        a = 1

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_ref_body_index = self.robot.body_names.index(self.cfg.reference_body)
        self.motion_ref_body_index = self.cfg.body_names.index(self.cfg.reference_body)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        self.motion = MotionLoader(
            self.cfg.motion_file, self.body_indexes, device=self.device
        )
        self.counts = torch.zeros(
            self.motion.num_motions, dtype=torch.float32, device=self.device
        )
        self.time_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, device=self.device
        )
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 4, device=self.device
        )
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.metrics["error_ref_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_ref_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_ref_lin_vel"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_ref_ang_vel"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        for name in self.motion.extracted_list:
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
    
    @property
    def command(
        self,
    ) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return (
            self.motion.body_pos_w[self.time_steps]
            + self._env.scene.env_origins[:, None, :]
        )

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def ref_pos_w(self) -> torch.Tensor:
        return (
            self.motion.body_pos_w[self.time_steps, self.motion_ref_body_index]
            + self._env.scene.env_origins
        )

    @property
    def ref_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_ref_body_index]

    @property
    def ref_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_ref_body_index]

    @property
    def ref_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_ref_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_ref_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_ref_body_index]

    @property
    def robot_ref_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_ref_body_index]

    @property
    def robot_ref_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_ref_body_index]

    @property
    def robot_ref_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_ref_body_index]

    def _update_metrics(self):
        self.metrics["error_ref_pos"] = torch.norm(
            self.ref_pos_w - self.robot_ref_pos_w, dim=-1
        )
        self.metrics["error_ref_rot"] = quat_error_magnitude(
            self.ref_quat_w, self.robot_ref_quat_w
        )
        self.metrics["error_ref_lin_vel"] = torch.norm(
            self.ref_lin_vel_w - self.robot_ref_lin_vel_w, dim=-1
        )
        self.metrics["error_ref_ang_vel"] = torch.norm(
            self.ref_ang_vel_w - self.robot_ref_ang_vel_w, dim=-1
        )

        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)

        self.metrics["error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)

        self.metrics["error_joint_pos"] = torch.norm(
            self.joint_pos - self.robot_joint_pos, dim=-1
        )
        self.metrics["error_joint_vel"] = torch.norm(
            self.joint_vel - self.robot_joint_vel, dim=-1
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # phase = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
        # self.time_steps[env_ids] = (phase * (self.motion.time_step_total - 1)).long()

        if len(env_ids) == 0:
            return

        # Compute adjusted probabilities for balancing based on current distribution
        epsilon = 1e-6

        current_dist = self.motion.motion_distribution.squeeze(0)  # (num_motions,)
        target_dist = self.motion.target_dist.squeeze(0)  # (num_motions,)
        weights = target_dist / (current_dist + epsilon)
        probs = weights / weights.sum()  # Normalized probabilities for dynamic balance

        # Sample motion indices based on adjusted probs
        motion_ids = torch.multinomial(
            probs, len(env_ids), replacement=True
        )  # (len(env_ids),)

        # For each env, sample local phase in the selected motion
        selected_starts = self.motion.motion_indices[motion_ids, 0]  # (len(env_ids),)
        selected_lengths = torch.tensor(
            [self.motion.motion_lengths[mid.item()] for mid in motion_ids],
            device=self.device,
        )
        local_phases = torch.rand((len(env_ids),), device=self.device)  # Uniform [0,1)
        local_steps = (local_phases * (selected_lengths - 1)).long()
        self.time_steps[env_ids] = selected_starts + local_steps

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [
            self.cfg.pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
        )
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [
            self.cfg.velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(
            *self.cfg.joint_position_range, joint_pos.shape, joint_pos.device
        )
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids],
            soft_joint_pos_limits[:, :, 0],
            soft_joint_pos_limits[:, :, 1],
        )

        # joint_vel += sample_uniform(*self.cfg.joint_velocity_range, joint_vel.shape, joint_vel.device)
        # soft_joint_vel_limits = self.robot.data.soft_joint_vel_limits[env_ids]
        # joint_vel[env_ids] = torch.clip(
        #     joint_vel[env_ids], soft_joint_vel_limits[:, :, 0], soft_joint_vel_limits[:, :, 1]
        # )
        self.robot.write_joint_state_to_sim(
            joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
        )
        self.robot.write_root_state_to_sim(
            torch.cat(
                [
                    root_pos[env_ids],
                    root_ori[env_ids],
                    root_lin_vel[env_ids],
                    root_ang_vel[env_ids],
                ],
                dim=-1,
            ),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        for i in range(self.motion.num_motions):
            start, end = self.motion.motion_indices[i]
            map = (self.time_steps >= start) & (self.time_steps < end)
            self.metrics[self.motion.extracted_list[i]] = map.clone().float()
            self.counts[i] = map.sum().float()
        self.motion.motion_distribution = (self.counts / self.num_envs).unsqueeze(0)

        self._resample_command(env_ids)

        ref_pos_w_repeat = self.ref_pos_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        ref_quat_w_repeat = self.ref_quat_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        robot_ref_pos_w_repeat = self.robot_ref_pos_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        robot_ref_quat_w_repeat = self.robot_ref_quat_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )

        delta_pos_w = ref_pos_w_repeat - robot_ref_pos_w_repeat
        delta_pos_w[..., :2] = 0.0
        delta_ori_w = yaw_quat(
            quat_mul(robot_ref_quat_w_repeat, quat_inv(ref_quat_w_repeat))
        )

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = (
            robot_ref_pos_w_repeat
            + delta_pos_w
            + quat_apply(delta_ori_w, self.body_pos_w - ref_pos_w_repeat)
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_ref_visualizer"):
                self.current_ref_visualizer = VisualizationMarkers(
                    self.cfg.ref_visualizer_cfg.replace(
                        prim_path="/Visuals/Command/current/ref"
                    )
                )
                self.goal_ref_visualizer = VisualizationMarkers(
                    self.cfg.ref_visualizer_cfg.replace(
                        prim_path="/Visuals/Command/goal/ref"
                    )
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(
                                prim_path="/Visuals/Command/current/" + name
                            )
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(
                                prim_path="/Visuals/Command/goal/" + name
                            )
                        )
                    )

            self.current_ref_visualizer.set_visibility(True)
            self.goal_ref_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_ref_visualizer"):
                self.current_ref_visualizer.set_visibility(False)
                self.goal_ref_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_ref_visualizer.visualize(
            self.robot_ref_pos_w, self.robot_ref_quat_w
        )
        self.goal_ref_visualizer.visualize(self.ref_pos_w, self.ref_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(
                self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i]
            )
            self.goal_body_visualizers[i].visualize(
                self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i]
            )


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    reference_body: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    joint_velocity_range: tuple[float, float] = (-0.52, 0.52)

    ref_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    ref_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
