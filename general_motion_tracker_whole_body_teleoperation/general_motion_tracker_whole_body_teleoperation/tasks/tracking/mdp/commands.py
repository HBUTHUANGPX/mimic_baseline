from __future__ import annotations

import math
import time
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    matrix_from_quat,
    euler_xyz_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
    wrap_to_pi,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from general_motion_tracker_whole_body_teleoperation.utils.motion_loader import (
    MotionLoader_human as MotionLoader,
)
from general_motion_tracker_whole_body_teleoperation.tasks.tracking.mdp.adaptive_sample import (
    AdaptiveSamplingModule,
    AdaptiveSamplingModuleCfg,
    LegacyBinAdaptiveSampling,
    LegacyBinAdaptiveSamplingCfg,
    SonicBinAdaptiveSampling,
    SonicBinAdaptiveSamplingCfg,
    StratifiedLegacyBinAdaptiveSampling,
    StratifiedLegacyBinAdaptiveSamplingCfg,
)
from general_motion_tracker_whole_body_teleoperation.tasks.tracking.mdp.motion_debug_visualizer import (
    MotionDebugVisualizer,
)


@torch.jit.script
def rot6d_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to the flattened first two rotation-matrix columns.

    Args:
        quaternions: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Flattened 6D rotation representation. Shape is (..., 6).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (6,))


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
        self.human_body_indexes = torch.tensor(
            self.cfg.desire_human_joint_names.index(self.cfg.human_anchor_name),
            dtype=torch.long,
            device=self.device,
        )
        self.human_anchor_body_index = self.cfg.desire_human_joint_names.index(
            self.cfg.human_anchor_name
        )
        self.fsq_human_body_indexes = torch.tensor(
            [
                self.cfg.desire_human_joint_names.index(name)
                for name in self.cfg.fsq_human_body_names
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.motion = MotionLoader(
            motion_file_group=self.cfg.motion_file,
            robot_body_names=self.robot.body_names,
            robot_joint_names=self.robot.joint_names,
            body_indexes=self.body_indexes,
            desire_human_joint_names=self.cfg.desire_human_joint_names,
            history_frames=self.cfg.history_frames,
            future_frames=self.cfg.future_frames,
            device=self.device,
            enable_distributed_sharding=self.cfg.enable_distributed_motion_sharding,
            use_token = self.cfg.use_token
        )
        num_robot_bodies = len(self.cfg.body_names)
        num_human_bodies = len(self.cfg.fsq_human_body_names)
        self.actor_ref_robot_fsq_feature_window = torch.zeros(
            self.num_envs,
            self.motion.window_size
            * (6 + self.motion.joint_pos.shape[-1] + 9 * num_robot_bodies),
            device=self.device,
        )
        self.critic_ref_robot_fsq_feature_window = torch.zeros(
            self.num_envs,
            self.motion.window_size
            * (9 + self.motion.joint_pos.shape[-1] + 9 * num_robot_bodies),
            device=self.device,
        )
        self.actor_ref_human_fsq_feature_window = torch.zeros(
            self.num_envs,
            self.motion.window_size * (6 + 15 * num_human_bodies),
            device=self.device,
        )
        self.critic_ref_human_fsq_feature_window = torch.zeros(
            self.num_envs,
            self.motion.window_size * (9 + 15 * num_human_bodies),
            device=self.device,
        )
        self.time_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._previous_time_steps = None
        self.yaw_aligned_ref_robot_body_pos_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, device=self.device
        )
        self.yaw_aligned_ref_robot_body_quat_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 4, device=self.device
        )
        self.yaw_aligned_ref_robot_body_quat_w[:, :, 0] = 1.0
        self.consecutive_bad_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        self.bin_count = (
            int(
                self.motion.time_step_total
                // (1 / (env.cfg.decimation * env.cfg.sim.dt))
            )
            + 1
        )
        self.bin_frame_count = max(
            int(1.0 / (env.cfg.decimation * env.cfg.sim.dt)),
            1,
        )
        self.valid_center_bin_ids = torch.clamp(
            self.motion.valid_center_indices // self.bin_frame_count,
            0,
            self.bin_count - 1,
        )
        self.valid_center_count_per_bin = torch.bincount(
            self.valid_center_bin_ids, minlength=self.bin_count
        )
        self.valid_sampling_bin_mask = self.valid_center_count_per_bin > 0
        max_valid_centers_per_bin = max(
            int(self.valid_center_count_per_bin.max().item()),
            1,
        )
        self.bin_valid_center_indices = torch.full(
            (self.bin_count, max_valid_centers_per_bin),
            self.motion.time_step_total,
            dtype=torch.long,
            device=self.device,
        )
        for bin_id in range(self.bin_count):
            valid_count = int(self.valid_center_count_per_bin[bin_id].item())
            if valid_count == 0:
                continue
            bin_valid_centers = self.motion.valid_center_indices[
                self.valid_center_bin_ids == bin_id
            ]
            self.bin_valid_center_indices[bin_id, :valid_count] = bin_valid_centers
        self.adaptive_sampler = self._build_adaptive_sampler()
        self._perpare_metrics()

        self._xy_plane_mask = torch.tensor([1.0, 1.0, 0.0], device=self.device)
        self.body_pos_start_w = (
            self.motion.body_pos_w[self.time_steps] * self._xy_plane_mask[None, None, :]
        )
        self.human_body_pos_start_w = (
            self.motion.human_body_pos_w[self.time_steps]
            * self._xy_plane_mask[None, None, :]
        )
        self.bad_steps_threshold = torch.zeros(self.num_envs, device=self.device)

        self._update_motion_cache(full=True)
        self._update_robot_state_cache()
        self._make_calculate()
        # self._update_termination_cache()
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
        self.scale_difficulty = torch.zeros(1, device=self.device)
        self.soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits.clone()

    @property
    def command(self) -> torch.Tensor:
        return self.ref_robot_joint_state_cmd

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
        self.metrics["scale_difficulty"] = torch.zeros(
            self.num_envs, device=self.device
        )

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = self.ref_robot_anchor_pos_error_norm
        self.metrics["error_anchor_rot"] = self.ref_robot_anchor_angle_error
        self.metrics["error_anchor_lin_vel"] = self.ref_robot_anchor_lin_vel_error_norm
        self.metrics["error_anchor_ang_vel"] = self.ref_robot_anchor_ang_vel_error_norm
        self.metrics["error_body_pos"] = (
            self.yaw_aligned_ref_robot_body_pos_error_norm.mean(dim=-1)
        )
        self.metrics["error_body_rot"] = (
            self.yaw_aligned_ref_robot_body_angle_error.mean(dim=-1)
        )
        self.metrics["error_body_lin_vel"] = (
            self.ref_robot_body_lin_vel_error_norm.mean(dim=-1)
        )
        self.metrics["error_body_ang_vel"] = (
            self.ref_robot_body_ang_vel_error_norm.mean(dim=-1)
        )
        self.metrics["error_joint_pos"] = self.ref_robot_joint_angle_error_norm
        self.metrics["error_joint_vel"] = self.ref_robot_joint_vel_error_norm
        self.metrics["scale_difficulty"] = self.scale_difficulty * torch.ones(
            self.num_envs, device=self.device
        )

    def _build_adaptive_sampler(self) -> AdaptiveSamplingModule:
        sampler_cfg = self.cfg.adaptive_sampler
        sampler_class = sampler_cfg.class_type
        if not issubclass(sampler_class, AdaptiveSamplingModule):
            raise ValueError(
                "adaptive_sampler.class_type must inherit from "
                "AdaptiveSamplingModule."
            )
        return sampler_class(self, sampler_cfg)

    def _update_sampling_metrics(self, sampling_probabilities: torch.Tensor):
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / max(math.log(self.bin_count), 1e-12)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _sample_time_steps_from_bins(self, sampled_bins: torch.Tensor) -> torch.Tensor:
        candidate_time_steps = (
            sampled_bins * self.bin_frame_count
            + sample_uniform(
                0.0,
                float(self.bin_frame_count),
                sampled_bins.shape,
                device=self.device,
            ).long()
        )
        candidate_time_steps = torch.clamp(
            candidate_time_steps, 0, self.motion.time_step_total - 1
        )
        valid_counts = self.valid_center_count_per_bin[sampled_bins]
        valid_centers = self.bin_valid_center_indices[sampled_bins]
        right_indices = torch.searchsorted(
            valid_centers, candidate_time_steps.unsqueeze(-1)
        ).squeeze(-1)
        right_indices = torch.clamp(right_indices, max=valid_counts - 1)
        left_indices = torch.clamp(right_indices - 1, min=0)

        left_centers = torch.gather(valid_centers, 1, left_indices.view(-1, 1)).squeeze(
            -1
        )
        right_centers = torch.gather(
            valid_centers, 1, right_indices.view(-1, 1)
        ).squeeze(-1)
        choose_right = torch.abs(right_centers - candidate_time_steps) < torch.abs(
            candidate_time_steps - left_centers
        )
        return torch.where(choose_right, right_centers, left_centers)

    def _resample_time_steps(
        self,
        env_ids: Sequence[int],
        update_failure_statistics: bool = True,
    ):
        if len(env_ids) == 0:
            return
        self.adaptive_sampler.on_resample_start(env_ids, update_failure_statistics)
        sampling_probabilities = self.adaptive_sampler.build_sampling_probabilities()
        sampled_bins = torch.multinomial(
            sampling_probabilities, len(env_ids), replacement=True
        )
        self.time_steps[env_ids] = self._sample_time_steps_from_bins(sampled_bins)
        self.adaptive_sampler.on_resample_complete(
            env_ids, sampled_bins, update_failure_statistics
        )
        self._update_sampling_metrics(sampling_probabilities)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._resample_time_steps(env_ids)
        self.body_pos_start_w[env_ids] = (
            self.motion.body_pos_w[self.time_steps] * self._xy_plane_mask[None, None, :]
        )[env_ids]
        self.human_body_pos_start_w[env_ids] = (
            self.motion.human_body_pos_w[self.time_steps]
            * self._xy_plane_mask[None, None, :]
        )[env_ids]
        self.consecutive_bad_steps[env_ids] = 0  # 重置坏跟踪连续计数器
        self._update_motion_cache(full=False)
        self._reset_env_by_motion(
            env_ids
        )  # 根据采样的time_stamps对应的motion数据重置环境状态

    def _get_env_ids_to_resample(self) -> torch.Tensor:
        overflow_mask = (self.time_steps < 0) | (
            self.time_steps >= self.motion.time_step_total
        )
        valid_center_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        non_overflow_ids = torch.nonzero(~overflow_mask, as_tuple=False).squeeze(-1)
        if non_overflow_ids.numel() > 0:
            valid_center_mask[non_overflow_ids] = self.motion.valid_center_mask[
                self.time_steps[non_overflow_ids]
            ]
        return torch.nonzero(
            overflow_mask | (~valid_center_mask), as_tuple=False
        ).squeeze(-1)

    def _update_motion_cache(self, full=False):
        # 在time_stamps更新后，更新缓存的motion数据,因为_resample_command在_update_command中被调用,所以当需要reset的env_ids数量为0时也要触发一次
        assert (
            self.time_steps.max() < self.motion.time_step_total
        ), f"time_steps: {self.time_steps}, motion time_step_total: {self.motion.time_step_total}"

        self.ref_robot_body_pos_w = (
            self.motion.body_pos_w[self.time_steps]
            - self.body_pos_start_w[
                :, self.motion_anchor_body_index : self.motion_anchor_body_index + 1, :
            ]
            + self._env.scene.env_origins[:, None, :]
        )
        self.ref_robot_body_quat_w = self.motion.body_quat_w[self.time_steps]
        self.ref_robot_body_lin_vel_w = self.motion.body_lin_vel_w[self.time_steps]
        self.ref_robot_body_ang_vel_w = self.motion.body_ang_vel_w[self.time_steps]
        self.ref_robot_joint_angle_rad = self.motion.joint_pos[self.time_steps]
        self.ref_robot_joint_vel_rad_s = self.motion.joint_vel[self.time_steps]
        if not full:
            return
        self.ref_robot_anchor_pos_w = (
            self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
            - self.body_pos_start_w[:, self.motion_anchor_body_index]
            + self._env.scene.env_origins
        )
        self.ref_robot_anchor_quat_w = self.motion.body_quat_w[
            self.time_steps, self.motion_anchor_body_index
        ]
        self.ref_robot_anchor_lin_vel_w = self.motion.body_lin_vel_w[
            self.time_steps, self.motion_anchor_body_index
        ]
        self.ref_robot_anchor_ang_vel_w = self.motion.body_ang_vel_w[
            self.time_steps, self.motion_anchor_body_index
        ]
        self.ref_human_body_pos_w = (
            self.motion.human_body_pos_w[self.time_steps]
            - self.human_body_pos_start_w[
                :, self.human_body_indexes : self.human_body_indexes + 1, :
            ]
            + self._env.scene.env_origins[:, None, :]
        )
        self.ref_human_body_quat_w = self.motion.human_body_quat_w[self.time_steps]
        self.ref_human_anchor_pos_w = (
            self.motion.human_body_pos_w[self.time_steps, self.human_body_indexes]
            - self.human_body_pos_start_w[:, self.human_body_indexes]
            + self._env.scene.env_origins
        )
        self.ref_human_anchor_quat_w = self.motion.human_body_quat_w[
            self.time_steps, self.human_body_indexes
        ]
        self.ref_motion_id = self.motion._motion_id[self.time_steps]
        self.ref_motion_group = self.motion._motion_group[self.time_steps]
        self.actor_q_human_latent = self.motion.actor_q_human[self.time_steps]
        self.actor_q_robot_latent = self.motion.actor_q_robot[self.time_steps]
        self.critic_q_human_latent = self.motion.critic_q_human[self.time_steps]
        self.critic_q_robot_latent = self.motion.critic_q_robot[self.time_steps]
        
    def _get_window_time_steps(self) -> torch.Tensor:
        window_time_steps = (
            self.time_steps[:, None] + self.motion.window_offsets[None, :]
        )
        return torch.clamp(window_time_steps, 0, self.motion.time_step_total - 1)

    def _make_calculate(self):
        num_bodies = len(self.cfg.body_names)
        ref_robot_anchor_pos_w_repeat = self.ref_robot_anchor_pos_w[:, None, :].expand(
            -1, num_bodies, -1
        )
        ref_robot_anchor_quat_w_repeat = self.ref_robot_anchor_quat_w[
            :, None, :
        ].expand(-1, num_bodies, -1)
        sim_robot_anchor_pos_w_repeat = self.sim_robot_anchor_pos_w[:, None, :].expand(
            -1, num_bodies, -1
        )
        sim_robot_anchor_quat_w_repeat = self.sim_robot_anchor_quat_w[
            :, None, :
        ].expand(-1, num_bodies, -1)

        # 基础命令和本体观测缓存。
        self.ref_robot_joint_state_cmd = torch.cat(
            [self.ref_robot_joint_angle_rad, self.ref_robot_joint_vel_rad_s], dim=1
        )
        self.sim_robot_anchor_spatial_vel_w = torch.cat(
            [self.sim_robot_anchor_lin_vel_w, self.sim_robot_anchor_ang_vel_w], dim=-1
        )
        self.ref_robot_minus_sim_joint_angle_rad = (
            self.ref_robot_joint_angle_rad - self.sim_robot_joint_angle_rad
        )
        self.sim_robot_anchor_rot6d_w = rot6d_from_quat(self.sim_robot_anchor_quat_w)

        # 参考机器人动作对齐到当前机器人 anchor yaw 后的世界系目标。
        delta_pos_w = torch.cat(
            [
                sim_robot_anchor_pos_w_repeat[..., :2],
                ref_robot_anchor_pos_w_repeat[..., 2:3],
            ],
            dim=-1,
        )
        delta_ori_w = yaw_quat(
            quat_mul(
                sim_robot_anchor_quat_w_repeat, quat_inv(ref_robot_anchor_quat_w_repeat)
            )
        )
        self.yaw_aligned_ref_robot_body_quat_w = quat_mul(
            delta_ori_w, self.ref_robot_body_quat_w
        )
        self.yaw_aligned_ref_robot_body_pos_w = delta_pos_w + quat_apply(
            delta_ori_w, self.ref_robot_body_pos_w - ref_robot_anchor_pos_w_repeat
        )

        # 当前机器人关键 body 在当前机器人 anchor frame 下的位姿，供 observation 使用。
        sim_robot_body_pos_in_sim_anchor, sim_robot_body_quat_in_sim_anchor = (
            subtract_frame_transforms(
                sim_robot_anchor_pos_w_repeat,
                sim_robot_anchor_quat_w_repeat,
                self.sim_robot_body_pos_w,
                self.sim_robot_body_quat_w,
            )
        )
        self.sim_robot_body_pos_in_sim_anchor = sim_robot_body_pos_in_sim_anchor
        self.sim_robot_body_rot6d_in_sim_anchor = rot6d_from_quat(
            sim_robot_body_quat_in_sim_anchor
        ).reshape(self.num_envs, -1)

        # 参考机器人 anchor 在当前机器人 anchor frame 下的相对位姿，供 observation 使用。
        ref_robot_anchor_pos_in_sim_anchor, ref_robot_anchor_quat_in_sim_anchor = (
            subtract_frame_transforms(
                self.sim_robot_anchor_pos_w,
                self.sim_robot_anchor_quat_w,
                self.ref_robot_anchor_pos_w,
                self.ref_robot_anchor_quat_w,
            )
        )
        self.ref_robot_anchor_pos_in_sim_anchor = ref_robot_anchor_pos_in_sim_anchor
        self.ref_robot_anchor_rot6d_in_sim_anchor = rot6d_from_quat(
            ref_robot_anchor_quat_in_sim_anchor
        ).reshape(self.num_envs, -1)

        # 参考 human anchor 在当前机器人 anchor frame 下的相对姿态，供 observation 使用。
        _, ref_human_anchor_quat_in_sim_anchor = subtract_frame_transforms(
            self.sim_robot_anchor_pos_w,
            self.sim_robot_anchor_quat_w,
            self.ref_human_anchor_pos_w,
            self.ref_human_anchor_quat_w,
        )
        self.ref_human_anchor_rot6d_in_sim_anchor = rot6d_from_quat(
            ref_human_anchor_quat_in_sim_anchor
        ).reshape(self.num_envs, -1)

        # FSQ 使用的 history/current/future 窗口特征。
        window_time_steps = self._get_window_time_steps()

        robot_anchor_quat = self.motion.body_quat_w[
            window_time_steps, self.motion_anchor_body_index
        ]
        robot_anchor_rot6d = rot6d_from_quat(robot_anchor_quat)
        robot_anchor_pos = self.motion.body_pos_w[
            window_time_steps, self.motion_anchor_body_index
        ]
        robot_joint_pos = self.motion.joint_pos[window_time_steps]
        num_robot_bodies = len(self.cfg.body_names)
        robot_body_pos = self.motion.body_pos_w[window_time_steps]
        robot_body_quat = self.motion.body_quat_w[window_time_steps]
        robot_anchor_pos_repeat = robot_anchor_pos[:, :, None, :].expand(
            -1, -1, num_robot_bodies, -1
        )
        robot_anchor_quat_repeat = robot_anchor_quat[:, :, None, :].expand(
            -1, -1, num_robot_bodies, -1
        )
        ref_robot_body_pos_in_ref_anchor, ref_robot_body_quat_in_ref_anchor = (
            subtract_frame_transforms(
                robot_anchor_pos_repeat.reshape(-1, 3),
                robot_anchor_quat_repeat.reshape(-1, 4),
                robot_body_pos.reshape(-1, 3),
                robot_body_quat.reshape(-1, 4),
            )
        )
        ref_robot_body_pos_in_ref_anchor = ref_robot_body_pos_in_ref_anchor.reshape(
            self.num_envs, self.motion.window_size, -1
        )
        ref_robot_body_rot6d_in_ref_anchor = rot6d_from_quat(
            ref_robot_body_quat_in_ref_anchor
        ).reshape(self.num_envs, self.motion.window_size, -1)
        actor_robot_feature = torch.cat(
            (
                robot_anchor_rot6d,
                robot_joint_pos,
                # ref_robot_body_pos_in_ref_anchor,
                # ref_robot_body_rot6d_in_ref_anchor,
            ),
            dim=-1,
        )
        critic_robot_feature = torch.cat(
            (
                robot_anchor_rot6d,
                robot_anchor_pos,
                robot_joint_pos,
                ref_robot_body_pos_in_ref_anchor,
                ref_robot_body_rot6d_in_ref_anchor,
            ),
            dim=-1,
        )
        self.actor_ref_robot_fsq_feature_window = actor_robot_feature.reshape(
            self.num_envs, -1
        )
        self.critic_ref_robot_fsq_feature_window = critic_robot_feature.reshape(
            self.num_envs, -1
        )

        human_anchor_quat = self.motion.human_body_quat_w[
            window_time_steps, self.human_anchor_body_index
        ]
        human_anchor_rot6d = rot6d_from_quat(human_anchor_quat)
        human_anchor_pos = self.motion.human_body_pos_w[
            window_time_steps, self.human_anchor_body_index
        ]
        human_body_pos = self.motion.human_body_pos_w[window_time_steps][
            :, :, self.fsq_human_body_indexes, :
        ]
        human_body_quat = self.motion.human_body_quat_w[window_time_steps][
            :, :, self.fsq_human_body_indexes, :
        ]
        human_joint_quat = self.motion.human_joint_quat[window_time_steps][
            :, :, self.fsq_human_body_indexes, :
        ]
        ref_human_body_pos_from_ref_anchor_w = (
            human_body_pos - human_anchor_pos[:, :, None, :]
        )
        num_human_bodies = self.fsq_human_body_indexes.numel()
        human_anchor_quat_w = human_anchor_quat[:, :, None, :].expand(
            -1, -1, num_human_bodies, -1
        )
        ref_human_body_pos_in_ref_anchor = quat_apply_inverse(
            human_anchor_quat_w.reshape(-1, 4),
            ref_human_body_pos_from_ref_anchor_w.reshape(-1, 3),
        ).reshape(self.num_envs, self.motion.window_size, -1)
        ref_human_body_quat_in_ref_anchor = quat_mul(
            quat_inv(human_anchor_quat_w.reshape(-1, 4)),
            human_body_quat.reshape(-1, 4),
        )
        ref_human_body_rot6d_in_ref_anchor = rot6d_from_quat(
            ref_human_body_quat_in_ref_anchor
        ).reshape(self.num_envs, self.motion.window_size, -1)
        human_joint_rot6d = rot6d_from_quat(human_joint_quat).reshape(
            self.num_envs, self.motion.window_size, -1
        )
        actor_human_feature = torch.cat(
            (
                human_anchor_rot6d,
                # human_joint_rot6d,
                ref_human_body_pos_in_ref_anchor,
                # ref_human_body_rot6d_in_ref_anchor,
            ),
            dim=-1,
        )
        critic_human_feature = torch.cat(
            (
                human_anchor_rot6d,
                human_anchor_pos,
                human_joint_rot6d,
                ref_human_body_pos_in_ref_anchor,
                ref_human_body_rot6d_in_ref_anchor,
            ),
            dim=-1,
        )
        self.actor_ref_human_fsq_feature_window = actor_human_feature.reshape(
            self.num_envs, -1
        )
        self.critic_ref_human_fsq_feature_window = critic_human_feature.reshape(
            self.num_envs, -1
        )

        # 奖励、终止和指标共用的误差缓存。
        self.ref_robot_anchor_pos_error_w = (
            self.ref_robot_anchor_pos_w - self.sim_robot_anchor_pos_w
        )
        self.ref_robot_anchor_lin_vel_error_w = (
            self.ref_robot_anchor_lin_vel_w - self.sim_robot_anchor_lin_vel_w
        )
        self.ref_robot_anchor_ang_vel_error_w = (
            self.ref_robot_anchor_ang_vel_w - self.sim_robot_anchor_ang_vel_w
        )
        self.ref_robot_anchor_angle_error = quat_error_magnitude(
            self.ref_robot_anchor_quat_w, self.sim_robot_anchor_quat_w
        )
        self.yaw_aligned_ref_robot_body_pos_error_w = (
            self.yaw_aligned_ref_robot_body_pos_w - self.sim_robot_body_pos_w
        )
        self.yaw_aligned_ref_robot_body_angle_error = quat_error_magnitude(
            self.yaw_aligned_ref_robot_body_quat_w, self.sim_robot_body_quat_w
        )
        self.ref_robot_body_lin_vel_error_w = (
            self.ref_robot_body_lin_vel_w - self.sim_robot_body_lin_vel_w
        )
        self.ref_robot_body_ang_vel_error_w = (
            self.ref_robot_body_ang_vel_w - self.sim_robot_body_ang_vel_w
        )
        self.ref_robot_joint_angle_error_rad = (
            self.ref_robot_joint_angle_rad - self.sim_robot_joint_angle_rad
        )
        self.ref_robot_joint_vel_error_rad_s = (
            self.ref_robot_joint_vel_rad_s - self.sim_robot_joint_vel_rad_s
        )
        self.ref_robot_anchor_pos_error_norm = torch.norm(
            self.ref_robot_anchor_pos_error_w, dim=-1
        )
        self.ref_robot_anchor_lin_vel_error_norm = torch.norm(
            self.ref_robot_anchor_lin_vel_error_w, dim=-1
        )
        self.ref_robot_anchor_ang_vel_error_norm = torch.norm(
            self.ref_robot_anchor_ang_vel_error_w, dim=-1
        )
        self.yaw_aligned_ref_robot_body_pos_error_norm = torch.norm(
            self.yaw_aligned_ref_robot_body_pos_error_w, dim=-1
        )
        self.ref_robot_body_lin_vel_error_norm = torch.norm(
            self.ref_robot_body_lin_vel_error_w, dim=-1
        )
        self.ref_robot_body_ang_vel_error_norm = torch.norm(
            self.ref_robot_body_ang_vel_error_w, dim=-1
        )
        self.ref_robot_joint_angle_error_norm = torch.norm(
            self.ref_robot_joint_angle_error_rad, dim=-1
        )
        self.ref_robot_joint_vel_error_norm = torch.norm(
            self.ref_robot_joint_vel_error_rad_s, dim=-1
        )

        gravity_w = self.robot.data.GRAVITY_VEC_W
        self.ref_robot_projected_gravity_in_anchor = quat_apply_inverse(
            self.ref_robot_anchor_quat_w, gravity_w
        )
        self.sim_robot_projected_gravity_in_anchor = quat_apply_inverse(
            self.sim_robot_anchor_quat_w, gravity_w
        )

    def _update_robot_state_cache(self):
        self.sim_robot_body_pos_w = self.robot.data.body_pos_w[
            :, self.body_indexes
        ].clone()
        self.sim_robot_body_quat_w = self.robot.data.body_quat_w[
            :, self.body_indexes
        ].clone()
        self.sim_robot_body_lin_vel_w = self.robot.data.body_lin_vel_w[
            :, self.body_indexes
        ].clone()
        self.sim_robot_body_ang_vel_w = self.robot.data.body_ang_vel_w[
            :, self.body_indexes
        ].clone()
        self.sim_robot_joint_angle_rad = self.robot.data.joint_pos.clone()
        self.sim_robot_joint_vel_rad_s = self.robot.data.joint_vel.clone()
        self.sim_robot_anchor_pos_w = self.robot.data.body_pos_w[
            :, self.robot_anchor_body_index
        ].clone()
        self.sim_robot_anchor_quat_w = self.robot.data.body_quat_w[
            :, self.robot_anchor_body_index
        ].clone()
        self.sim_robot_anchor_lin_vel_w = self.robot.data.body_lin_vel_w[
            :, self.robot_anchor_body_index
        ].clone()
        self.sim_robot_anchor_ang_vel_w = self.robot.data.body_ang_vel_w[
            :, self.robot_anchor_body_index
        ].clone()
        self.sim_robot_root_lin_vel_b = self.robot.data.root_lin_vel_b.clone()

    def _reset_env_by_motion(self, env_ids: Sequence[int]):
        root_pos = self.ref_robot_body_pos_w[env_ids, 0]
        root_ori = self.ref_robot_body_quat_w[env_ids, 0]
        root_lin_vel = self.ref_robot_body_lin_vel_w[env_ids, 0]
        root_ang_vel = self.ref_robot_body_ang_vel_w[env_ids, 0]
        joint_pos = self.ref_robot_joint_angle_rad[env_ids]
        joint_vel = self.ref_robot_joint_vel_rad_s[env_ids]

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
        self._previous_time_steps = self.time_steps.clone()
        self.time_steps += 1
        env_ids = self._get_env_ids_to_resample()
        self._resample_command(env_ids)
        self._update_motion_cache(full=True)
        self._update_robot_state_cache()
        self._make_calculate()
        # self._update_termination_cache()
        self.adaptive_sampler.on_step_end()
        # self.reached_motion_end = self.time_steps > self.motion.time_step_total

    def _set_debug_vis_impl(self, debug_vis: bool):
        if not hasattr(self, "_motion_debug_visualizer"):
            self._motion_debug_visualizer = MotionDebugVisualizer(self.cfg)
        self._motion_debug_visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        if hasattr(self, "_motion_debug_visualizer"):
            self._motion_debug_visualizer.visualize(self)


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: dict[str, list[str] | str] | str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    desire_human_joint_names: list[str] = [
        "Hips",
        "Spine1", "Spine2", "Chest",
        "Neck1", "Neck2", "Head", "HeadEnd",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
        "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
    ]
    desire_human_joint_names_for_human_bodys: list[str] = [
        "Hips",
        "Spine1", "Spine2", "Chest",
        "Neck1", "Neck2", "Head", "HeadEnd",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
        "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
    ]
    desire_human_joint_names_for_six_point_human_bodys: list[str] = [
        "Hips",
        "HeadEnd",
        "LeftHand",
        "RightHand",
        "LeftToeEnd",
        "RightToeEnd",
    ]
    fsq_human_body_names: list[str] = [
        "Chest",
        "HeadEnd",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
        "RightLeg", "RightShin", "RightFoot", "RightToeBase",
    ]

    human_anchor_name: str = "Hips"

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    history_frames: int = 0
    future_frames: int = 0
    enable_distributed_motion_sharding: bool = True
    use_token: bool = False

    adaptive_sampler: AdaptiveSamplingModuleCfg = LegacyBinAdaptiveSamplingCfg()
    # adaptive_sampler: AdaptiveSamplingModuleCfg = StratifiedLegacyBinAdaptiveSamplingCfg()
    # adaptive_sampler: AdaptiveSamplingModuleCfg = SonicBinAdaptiveSamplingCfg()

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    human_anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    human_anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
    human_body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    human_body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
