from __future__ import annotations

import math
import time
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
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from general_motion_tracker_whole_body_teleoperation.tasks.tracking.mdp.motion_loader import (
    MotionLoader_robot as MotionLoader,
)


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(
            self.cfg.anchor_body_name
        )
        self.motion_anchor_body_index = self.cfg.body_names.index(
            self.cfg.anchor_body_name
        )
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        self.motion = MotionLoader(
            self.cfg.motion_file, self.body_indexes, device=self.device
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

        self.bin_count = (
            int(
                self.motion.time_step_total
                // (1 / (env.cfg.decimation * env.cfg.sim.dt))
            )
            + 1
        )
        self.bin_failed_count = torch.zeros(
            self.bin_count, dtype=torch.float, device=self.device
        )
        self._current_bin_failed = torch.zeros(
            self.bin_count, dtype=torch.float, device=self.device
        )
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()
        self._perpare_metrics()

        self.body_pos_start_w = self.motion.body_pos_w[self.time_steps]*torch.tensor([1,1,0], device=self.device)[None,...]

        self._update_motion_cache()
        self._update_robot_state_cache()
        self._make_calculate()
        range_list = [
            self.cfg.velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        self.velocity_ranges = torch.tensor(range_list, device=self.device)
        range_list = [
            self.cfg.pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        self.pose_ranges = torch.tensor(range_list, device=self.device)
        self.soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits.clone()

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _perpare_metrics(self):
        self.metrics["error_anchor_pos"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_anchor_rot"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_anchor_lin_vel"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_anchor_ang_vel"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["sampling_top1_prob"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["sampling_top1_bin"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_body_lin_vel"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_body_ang_vel"] = torch.zeros(
            self.num_envs, device=self.device
        )

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = self.anchor_pos_error_norm
        self.metrics["error_anchor_rot"] = self.anchor_rot_error
        self.metrics["error_anchor_lin_vel"] = self.anchor_lin_vel_error_norm
        self.metrics["error_anchor_ang_vel"] = self.anchor_ang_vel_error_norm
        self.metrics["error_body_pos"] = self.body_pos_error_norm.mean(dim=-1)
        self.metrics["error_body_rot"] = self.body_rot_error.mean(dim=-1)
        self.metrics["error_body_lin_vel"] = self.body_lin_vel_error_norm.mean(dim=-1)
        self.metrics["error_body_ang_vel"] = self.body_ang_vel_error_norm.mean(dim=-1)
        self.metrics["error_joint_pos"] = self.joint_pos_error_norm
        self.metrics["error_joint_vel"] = self.joint_vel_error_norm

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count)
                // max(self.motion.time_step_total, 1),
                0,
                self.bin_count - 1,
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(
                fail_bins, minlength=self.bin_count
            )

        # Sample
        sampling_probabilities = (
            self.bin_failed_count
            + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        )
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(
            sampling_probabilities, self.kernel.view(1, 1, -1)
        ).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(
            sampling_probabilities, len(env_ids), replacement=True
        )

        self.time_steps[env_ids] = (
            (
                sampled_bins
                + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
            )
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)  # 对time_stamps进行自适应采样
        self.body_pos_start_w[env_ids] = (self.motion.body_pos_w[self.time_steps]*torch.tensor([1,1,0], device=self.device)[None,...])[env_ids]
        self._update_motion_cache()
        self._reset_env_by_motion(
            env_ids
        )  # 根据采样的time_stamps对应的motion数据重置环境状态

    def _motion_command_reset(self, env_ids: Sequence[int]): ...

    def _update_motion_cache(self):
        # 在time_stamps更新后，更新缓存的motion数据,因为_resample_command在_update_command中被调用,所以当需要reset的env_ids数量为0时也要触发一次
        assert (
            self.time_steps.max() < self.motion.time_step_total
        ), f"time_steps: {self.time_steps}, motion time_step_total: {self.motion.time_step_total}"

        self.body_pos_w = (
            self.motion.body_pos_w[self.time_steps]
            - self.body_pos_start_w 
            + self._env.scene.env_origins[:, None, :]
        )
        self.body_quat_w = self.motion.body_quat_w[self.time_steps]
        self.body_lin_vel_w = self.motion.body_lin_vel_w[self.time_steps]
        self.body_ang_vel_w = self.motion.body_ang_vel_w[self.time_steps]
        self.joint_pos = self.motion.joint_pos[self.time_steps]
        self.joint_vel = self.motion.joint_vel[self.time_steps]
        self.anchor_pos_w = (
            self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
            - self.body_pos_start_w[:, self.motion_anchor_body_index]
            + self._env.scene.env_origins
        )
        self.anchor_quat_w = self.motion.body_quat_w[
            self.time_steps, self.motion_anchor_body_index
        ]
        self.anchor_lin_vel_w = self.motion.body_lin_vel_w[
            self.time_steps, self.motion_anchor_body_index
        ]
        self.anchor_ang_vel_w = self.motion.body_ang_vel_w[
            self.time_steps, self.motion_anchor_body_index
        ]
        self.motion_id = self.motion._motion_id[self.time_steps]
        self.motion_group = self.motion._motion_group[self.time_steps]

    def _update_robot_state_cache(self):
        self.robot_body_pos_w = self.robot.data.body_pos_w[:, self.body_indexes].clone()
        self.robot_body_quat_w = self.robot.data.body_quat_w[
            :, self.body_indexes
        ].clone()
        self.robot_body_lin_vel_w = self.robot.data.body_lin_vel_w[
            :, self.body_indexes
        ].clone()
        self.robot_body_ang_vel_w = self.robot.data.body_ang_vel_w[
            :, self.body_indexes
        ].clone()
        self.robot_joint_pos = self.robot.data.joint_pos.clone()
        self.robot_joint_vel = self.robot.data.joint_vel.clone()
        self.robot_anchor_pos_w = self.robot.data.body_pos_w[
            :, self.robot_anchor_body_index
        ].clone()
        self.robot_anchor_quat_w = self.robot.data.body_quat_w[
            :, self.robot_anchor_body_index
        ].clone()
        self.robot_anchor_lin_vel_w = self.robot.data.body_lin_vel_w[
            :, self.robot_anchor_body_index
        ].clone()
        self.robot_anchor_ang_vel_w = self.robot.data.body_ang_vel_w[
            :, self.robot_anchor_body_index
        ].clone()

    def _make_calculate(self):
        num_bodies = len(self.cfg.body_names)
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].expand(-1, num_bodies, -1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].expand(-1, num_bodies, -1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].expand(
            -1, num_bodies, -1
        )
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].expand(
            -1, num_bodies, -1
        )

        # Build legacy command tensor once per step.
        self._command = torch.cat([self.joint_pos, self.joint_vel], dim=1)
        self.robot_anchor_vel_w = torch.cat(
            [self.robot_anchor_lin_vel_w, self.robot_anchor_ang_vel_w], dim=-1
        )
        self.joint_pos_delta = self.joint_pos - self.robot_joint_pos
        # Global-anchor alignment used by rewards/terminations.
        delta_pos_w = robot_anchor_pos_w_repeat.clone()
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(
            quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
        )
        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(
            delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
        )

        # Robot anchor orientation in 6D representation.
        robot_anchor_ori_mat = matrix_from_quat(self.robot_anchor_quat_w)
        self.robot_anchor_ori_w = robot_anchor_ori_mat[..., :2].reshape(
            self.num_envs, -1
        )
        # self.robot_anchor_ori_w = self.robot_anchor_quat_w
        # Robot body pose in robot-anchor frame.
        robot_body_pos_b, robot_body_ori_b = subtract_frame_transforms(
            robot_anchor_pos_w_repeat,
            robot_anchor_quat_w_repeat,
            self.robot_body_pos_w,
            self.robot_body_quat_w,
        )
        self.robot_body_pos_b = robot_body_pos_b
        # self.robot_body_ori_b = robot_body_ori_b.reshape(self.num_envs, -1)
        self.robot_body_ori_b = matrix_from_quat(robot_body_ori_b)[..., :2].reshape(
            self.num_envs, -1
        )
        # Motion anchor pose in robot-anchor frame.
        motion_anchor_pos_b, motion_anchor_ori_b = subtract_frame_transforms(
            self.robot_anchor_pos_w,
            self.robot_anchor_quat_w,
            self.anchor_pos_w,
            self.anchor_quat_w,
        )
        self.motion_anchor_pos_b = motion_anchor_pos_b
        self.motion_anchor_ori_b = matrix_from_quat(motion_anchor_ori_b)[
            ..., :2
        ].reshape(self.num_envs, -1)
        # self.motion_anchor_ori_b = motion_anchor_ori_b
        # Shared error tensors used by rewards/terminations/metrics.
        self.anchor_pos_error = self.anchor_pos_w - self.robot_anchor_pos_w
        self.anchor_lin_vel_error = self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w
        self.anchor_ang_vel_error = self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w
        self.anchor_rot_error = quat_error_magnitude(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        self.body_pos_error = self.body_pos_relative_w - self.robot_body_pos_w
        self.body_rot_error = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        )
        self.body_lin_vel_error = self.body_lin_vel_w - self.robot_body_lin_vel_w
        self.body_ang_vel_error = self.body_ang_vel_w - self.robot_body_ang_vel_w
        self.joint_pos_error = self.joint_pos - self.robot_joint_pos
        self.joint_vel_error = self.joint_vel - self.robot_joint_vel
        self.anchor_pos_error_norm = torch.norm(self.anchor_pos_error, dim=-1)
        self.anchor_lin_vel_error_norm = torch.norm(self.anchor_lin_vel_error, dim=-1)
        self.anchor_ang_vel_error_norm = torch.norm(self.anchor_ang_vel_error, dim=-1)
        self.body_pos_error_norm = torch.norm(self.body_pos_error, dim=-1)
        self.body_lin_vel_error_norm = torch.norm(self.body_lin_vel_error, dim=-1)
        self.body_ang_vel_error_norm = torch.norm(self.body_ang_vel_error, dim=-1)
        self.joint_pos_error_norm = torch.norm(self.joint_pos_error, dim=-1)
        self.joint_vel_error_norm = torch.norm(self.joint_vel_error, dim=-1)

        gravity_w = self.robot.data.GRAVITY_VEC_W
        self.motion_projected_gravity_b = quat_apply_inverse(
            self.anchor_quat_w, gravity_w
        )
        self.robot_projected_gravity_b = quat_apply_inverse(
            self.robot_anchor_quat_w, gravity_w
        )

    def _reset_env_by_motion(self, env_ids: Sequence[int]):
        root_pos = self.body_pos_w[env_ids, 0]
        root_ori = self.body_quat_w[env_ids, 0]
        root_lin_vel = self.body_lin_vel_w[env_ids, 0]
        root_ang_vel = self.body_ang_vel_w[env_ids, 0]
        joint_pos = self.joint_pos[env_ids]
        joint_vel = self.joint_vel[env_ids]

        rand_samples = sample_uniform(
            self.pose_ranges[:, 0],
            self.pose_ranges[:, 1],
            (len(env_ids), 6),
            device=self.device,
        )
        root_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
        )
        root_ori = quat_mul(orientations_delta, root_ori)
        rand_samples = sample_uniform(
            self.velocity_ranges[:, 0],
            self.velocity_ranges[:, 1],
            (len(env_ids), 6),
            device=self.device,
        )
        root_lin_vel += rand_samples[:, :3]
        root_ang_vel += rand_samples[:, 3:]
        joint_pos += sample_uniform(
            *self.cfg.joint_position_range, joint_pos.shape, joint_pos.device
        )
        joint_pos = torch.clip(
            joint_pos,
            self.soft_joint_pos_limits[env_ids, :, 0],
            self.soft_joint_pos_limits[env_ids, :, 1],
        )
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat(
                [
                    root_pos,
                    root_ori,
                    root_lin_vel,
                    root_ang_vel,
                ],
                dim=-1,
            ),
            env_ids=env_ids,
        )

    def _update_command(self):
        # TODO: 这地方太糟糕了,time_steps的增加和环境重置不应该放在此处.
        # 具体需要分析IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py中的step函数中的流程
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)
        self._update_motion_cache()
        self._update_robot_state_cache()
        self._make_calculate()

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed
            + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()
        # self.reached_motion_end = self.time_steps > self.motion.time_step_total

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(
                        prim_path="/Visuals/Command/current/anchor"
                    )
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(
                        prim_path="/Visuals/Command/goal/anchor"
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

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(
            self.robot_anchor_pos_w, self.robot_anchor_quat_w
        )
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

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

    motion_file: dict[str, list[str] | str] | str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    enable_timing_metrics: bool = True
    timing_sync_cuda: bool = False
    timing_ema_alpha: float = 0.1

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
