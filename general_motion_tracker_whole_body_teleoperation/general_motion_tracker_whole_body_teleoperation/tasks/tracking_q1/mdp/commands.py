from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING
from typing import Dict, List
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
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
import re


# def extract_part(path):
#     # 使用正则匹配 '/([^:/]+):' 模式，捕获组1为所需部分
#     match = re.search(r"/([^:/]+):", path)
#     if match:
#         return match.group(1)
#     return None
def extract_part(path):
    # 假设路径以 'artifacts/' 开头，提取其后的相对路径（包括子文件夹和文件名）
    if path.startswith("artifacts/"):
        # 去除 'artifacts/' 前缀，并返回剩余部分
        relative_path = path[len("artifacts/") :]
        # 验证是否为有效的NPZ文件路径
        if relative_path.endswith(".npz"):
            return relative_path
    return None


def get_run_name(mf: str) -> str | None:
    if mf.startswith("artifacts/"):
        path = mf[len("artifacts/") :]
    path = path.replace("/", "_")
    if path.endswith(".npz"):
        path = path[:-4]
    return path


class MotionLoader:
    def __init__(
        self,
        motion_file_group: Dict[str, List[str]],
        body_indexes: Sequence[int],
        device: str = "cpu",
    ):
        self.group_names = []
        self.extracted_list = []
        self.num_motions = 0
        motion_file_group_index = 0

        # Load and concatenate data from all files
        joint_pos_list = []
        joint_vel_list = []
        body_pos_w_list = []
        body_quat_w_list = []
        body_lin_vel_w_list = []
        body_ang_vel_w_list = []
        motion_id_list = []
        motion_group_list = []
        self.motion_lengths = []  # Length of each motion segment
        self.fps = None  # Assume all files have the same fps

        for group_name, paths in motion_file_group.items():
            print(f"\nGroup: {group_name}")
            print(f"[INFO] Loading {len(paths)} motion files for training.")

            # 支持单个字符串或列表，统一转换为列表
            if isinstance(paths, str):
                paths = [paths]
            print(f"[INFO] load motion file: {paths}")
            for file in paths:
                assert os.path.isfile(file), f"Invalid file path: {file}"
            extracted_list = [
                extract_part(p) for p in paths if extract_part(p) is not None
            ]
            num_motions = len(extracted_list)

            # for _file in self.motion_file:
            for i, _file in enumerate(paths):
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
                    torch.tensor(
                        data["body_quat_w"], dtype=torch.float32, device=device
                    )
                )
                body_lin_vel_w_list.append(
                    torch.tensor(
                        data["body_lin_vel_w"], dtype=torch.float32, device=device
                    )
                )
                body_ang_vel_w_list.append(
                    torch.tensor(
                        data["body_ang_vel_w"], dtype=torch.float32, device=device
                    )
                )
                motion_group_list.append(
                    torch.tensor(
                        motion_file_group_index, dtype=torch.float32, device=device
                    )
                    * torch.ones(
                        data["joint_pos"].shape[0],
                        1,
                        dtype=torch.float32,
                        device=device,
                    )
                )
                motion_id_list.append(
                    torch.tensor(
                        self.num_motions + i, dtype=torch.float32, device=device
                    )
                    * torch.ones(
                        data["joint_pos"].shape[0],
                        1,
                        dtype=torch.float32,
                        device=device,
                    )
                )
                self.motion_lengths.append(data["joint_pos"].shape[0])
            motion_file_group_index += 1
            self.extracted_list.extend(extracted_list)
            print(self.extracted_list)
            self.num_motions += num_motions
            self.group_names.append(group_name)
        assert self.num_motions > 0, "At least one motion file is required."
        # Concatenate along time dimension (dim=0)
        self.joint_pos = torch.cat(joint_pos_list, dim=0)
        self.joint_vel = torch.cat(joint_vel_list, dim=0)
        self._body_pos_w = torch.cat(body_pos_w_list, dim=0)
        self._body_quat_w = torch.cat(body_quat_w_list, dim=0)
        self._body_lin_vel_w = torch.cat(body_lin_vel_w_list, dim=0)
        self._body_ang_vel_w = torch.cat(body_ang_vel_w_list, dim=0)
        self._motion_id = torch.cat(motion_id_list, dim=0)
        self._motion_group = torch.cat(motion_group_list, dim=0)

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

        a = 1
        self.body_ang_vel_w = self._body_ang_vel_w[:, self._body_indexes]
        self.body_pos_w = self._body_pos_w[:, self._body_indexes]
        self.body_quat_w = self._body_quat_w[:, self._body_indexes]
        self.body_lin_vel_w = self._body_lin_vel_w[:, self._body_indexes]


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
        self.load_motion(self.cfg.motion_file)

    def load_motion(self, motion_file: Dict[str, List[str]]):
        self.motion = MotionLoader(motion_file, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._motion_ends = self.motion.motion_indices[:, 1].contiguous()
        ts = torch.clamp(self.time_steps, 0, self.motion.time_step_total - 1)
        self.motion_ids = torch.bucketize(
            ts, self._motion_ends, right=True
        )  # Intervals are [start, end); right=True ensures ts==end maps to next motion
        # Cache env-level motion ids as the single source of truth
        self.env_motion_ids = self.motion_ids.clone()
        # per-step cached tensors (computed in _update_state_data)
        self._ref_pos_w = None
        self._ref_quat_w = None
        self._robot_ref_pos_w = None
        self._robot_ref_quat_w = None
        self._robot_ref_lin_vel_w = None
        self._robot_ref_ang_vel_w = None
        self._robot_joint_pos = None
        self._robot_joint_vel = None
        self._robot_body_pos_w = None
        self._robot_body_quat_w = None
        self._robot_body_lin_vel_w = None
        self._robot_body_ang_vel_w = None
        self._robot_body_pos_b = None
        self._robot_body_ori_b_mat = None
        self._motion_ref_pos_b = None
        self._motion_ref_ori_b_mat = None
        self._robot_ref_ori_w_mat = None
        self._body_pos_w = None
        self._body_quat_w = None
        self._body_lin_vel_w = None
        self._body_ang_vel_w = None
        self._motion_body_pos_w_timestep = None
        self._motion_body_quat_w_timestep = None
        self._motion_body_lin_vel_w_timestep = None
        self._motion_body_ang_vel_w_timestep = None
        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(self.cfg.body_names), 3, device=self.device
        )
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(self.cfg.body_names), 4, device=self.device
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
        self.metrics["sampling_entropy"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["sampling_top1_prob"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["sampling_top1_bin"] = torch.zeros(
            self.num_envs, device=self.device
        )
        # for name in self.motion.extracted_list:
        #     self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        # timing metrics removed

        # Global timeline adaptive sampling (bin-based), aligned with commands_3.
        self.bin_count = (
            int(
                self.motion.time_step_total
                // (1 / (self._env.cfg.decimation * self._env.cfg.sim.dt))
            )
            + 1
        )
        self.bin_failed_count = torch.zeros(
            self.bin_count, dtype=torch.float32, device=self.device
        )
        self._current_bin_failed = torch.zeros(
            self.bin_count, dtype=torch.float32, device=self.device
        )
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
            dtype=torch.float32,
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()
        self._update_motion_data()
        self._update_state_data()

    @property
    def motion_id(self) -> torch.Tensor:
        return self.motion._motion_id[self.time_steps]

    @property
    def motion_group(self) -> torch.Tensor:
        return self.motion._motion_group[self.time_steps]

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
        return self._body_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w

    @property
    def ref_pos_w(self) -> torch.Tensor:
        return self._ref_pos_w

    @property
    def ref_quat_w(self) -> torch.Tensor:
        return self._ref_quat_w

    @property
    def ref_lin_vel_w(self) -> torch.Tensor:  # tag 2.05ms
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_ref_body_index]

    @property
    def ref_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_ref_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self._robot_joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self._robot_joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:  # tag 8.2ms
        return self._robot_body_pos_w

    @property
    def robot_body_quat_w(self) -> torch.Tensor:  # tag 10.66ms
        return self._robot_body_quat_w

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:  # tag 10.2ms
        return self._robot_body_lin_vel_w

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:  # tag 10.5ms
        return self._robot_body_ang_vel_w

    @property
    def robot_ref_pos_w(self) -> torch.Tensor:  # tag 14.5ms
        return self._robot_ref_pos_w

    @property
    def robot_ref_quat_w(self) -> torch.Tensor:  # tag 20ms
        return self._robot_ref_quat_w

    @property
    def robot_ref_lin_vel_w(self) -> torch.Tensor:  # tag 2.05ms
        return self._robot_ref_lin_vel_w

    @property
    def robot_ref_ang_vel_w(self) -> torch.Tensor:  # tag 2.05ms
        return self._robot_ref_ang_vel_w

    def _update_metrics(self):
        # self.metrics["error_ref_pos"] = torch.norm(
        #     self.ref_pos_w - self.robot_ref_pos_w, dim=-1
        # )
        # self.metrics["error_ref_rot"] = quat_error_magnitude(
        #     self.ref_quat_w, self.robot_ref_quat_w
        # )
        # self.metrics["error_ref_lin_vel"] = torch.norm(
        #     self.ref_lin_vel_w - self.robot_ref_lin_vel_w, dim=-1
        # )
        # self.metrics["error_ref_ang_vel"] = torch.norm(
        #     self.ref_ang_vel_w - self.robot_ref_ang_vel_w, dim=-1
        # )

        # self.metrics["error_body_pos"] = torch.norm(
        #     self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        # ).mean(dim=-1)
        # self.metrics["error_body_rot"] = quat_error_magnitude(
        #     self.body_quat_relative_w, self.robot_body_quat_w
        # ).mean(dim=-1)

        # self.metrics["error_body_lin_vel"] = torch.norm(
        #     self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        # ).mean(dim=-1)
        # self.metrics["error_body_ang_vel"] = torch.norm(
        #     self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        # ).mean(dim=-1)

        # self.metrics["error_joint_pos"] = torch.norm(
        #     self.joint_pos - self.robot_joint_pos, dim=-1
        # )
        # self.metrics["error_joint_vel"] = torch.norm(
        #     self.joint_vel - self.robot_joint_vel, dim=-1
        # )
        # for i in range(self.motion.num_motions):
        #     self.metrics[self.motion.extracted_list[i]] = (self.motion_ids == i).float()
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        # phase = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
        # self.time_steps[env_ids] = (phase * (self.motion.time_step_total - 1)).long()

        if len(env_ids) == 0:
            return
        self._resample_adaptive_sampling(env_ids)
        self._update_motion_data()
        self._resample_reset_robot_state(env_ids)

    def _resample_adaptive_sampling(self, env_ids: Sequence[int]):
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

        sampling_probabilities = (
            self.bin_failed_count
            + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        )
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),
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
        ts = torch.clamp(self.time_steps[env_ids], 0, self.motion.time_step_total - 1)
        self.env_motion_ids[env_ids] = torch.bucketize(
            ts, self._motion_ends, right=True
        )

        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / max(math.log(self.bin_count), 1e-12)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed
            + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _resample_reset_robot_state(self, env_ids: Sequence[int]):
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

    def _update_command(self):  # 入口
        self.time_steps += 1
        env_ids = self._get_env_ids_to_resample()
        self._post_update_command()
        # 根据动态平衡策略为需要重采样的 env 重新分配 time_steps
        self._resample_command(env_ids)
        self._update_state_data()

    def _update_motion_data(self):
        ts = torch.clamp(self.time_steps, 0, self.motion.time_step_total - 1)
        self._motion_body_pos_w_timestep = self.motion.body_pos_w[self.time_steps]
        self._motion_body_quat_w_timestep = self.motion.body_quat_w[self.time_steps]
        self._motion_body_lin_vel_w_timestep = self.motion.body_lin_vel_w[
            self.time_steps
        ]
        self._motion_body_ang_vel_w_timestep = self.motion.body_ang_vel_w[
            self.time_steps
        ]

        self._body_pos_w = (
            self._motion_body_pos_w_timestep + self._env.scene.env_origins[:, None, :]
        )
        self._body_quat_w = self._motion_body_quat_w_timestep
        self._body_lin_vel_w = self._motion_body_lin_vel_w_timestep
        self._body_ang_vel_w = self._motion_body_ang_vel_w_timestep

    def _get_env_ids_to_resample(self) -> torch.Tensor:
        overflow_mask = self.time_steps >= self.motion.time_step_total  # 溢出掩码
        valid_mask = ~overflow_mask  # 有效索引掩码 (time_steps < time_step_total)
        cross_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )  # 跨越掩码初始化
        if valid_mask.any():  # 仅对有效部分检查 new_data_flag
            valid_ids = torch.nonzero(valid_mask, as_tuple=False).squeeze(
                -1
            )  # 获取有效 env_ids
            cross_flags = self.motion.new_data_flag[
                self.time_steps[valid_ids]
            ]  # 检查对应 time_steps 的 flag
            cross_mask[valid_ids] = cross_flags  # 更新跨越掩码

        total_mask = overflow_mask | cross_mask  # 合并掩码：溢出或跨越
        env_ids = torch.nonzero(total_mask, as_tuple=False).squeeze(
            -1
        )  # 获取需要重采样的 env_ids
        return env_ids

    def _update_state_data(self):
        # Compute and cache frequently used tensors once per step.
        ref_pos_w = (
            self._motion_body_pos_w_timestep[:, self.motion_ref_body_index]
            + self._env.scene.env_origins
        )
        ref_quat_w = self._motion_body_quat_w_timestep[:, self.motion_ref_body_index]
        robot_data_body_pos_w = self.robot.data.body_pos_w.clone()
        robot_data_body_quat_w = self.robot.data.body_quat_w.clone()
        robot_data_body_lin_vel_w = self.robot.data.body_lin_vel_w.clone()
        robot_data_body_ang_vel_w = self.robot.data.body_ang_vel_w.clone()
        robot_joint_pos = self.robot.data.joint_pos.clone()
        robot_joint_vel = self.robot.data.joint_vel.clone()

        robot_ref_pos_w = robot_data_body_pos_w[:, self.robot_ref_body_index]
        robot_ref_quat_w = robot_data_body_quat_w[:, self.robot_ref_body_index]
        robot_body_pos_w = robot_data_body_pos_w[:, self.body_indexes]
        robot_body_quat_w = robot_data_body_quat_w[:, self.body_indexes]
        robot_body_lin_vel_w = robot_data_body_lin_vel_w[:, self.body_indexes]
        robot_body_ang_vel_w = robot_data_body_ang_vel_w[:, self.body_indexes]
        robot_ref_lin_vel_w = robot_data_body_lin_vel_w[:, self.robot_ref_body_index]
        robot_ref_ang_vel_w = robot_data_body_ang_vel_w[:, self.robot_ref_body_index]

        self._ref_pos_w = ref_pos_w
        self._ref_quat_w = ref_quat_w
        self._robot_ref_pos_w = robot_ref_pos_w
        self._robot_ref_quat_w = robot_ref_quat_w
        self._robot_body_pos_w = robot_body_pos_w
        self._robot_body_quat_w = robot_body_quat_w
        self._robot_body_lin_vel_w = robot_body_lin_vel_w
        self._robot_body_ang_vel_w = robot_body_ang_vel_w
        self._robot_ref_lin_vel_w = robot_ref_lin_vel_w
        self._robot_ref_ang_vel_w = robot_ref_ang_vel_w
        self._robot_joint_pos = robot_joint_pos
        self._robot_joint_vel = robot_joint_vel
        self._robot_ref_ori_w_mat = matrix_from_quat(robot_ref_quat_w)

        num_bodies = len(self.cfg.body_names)
        ref_pos_w_repeat = ref_pos_w[:, None, :].expand(-1, num_bodies, -1)
        ref_quat_w_repeat = ref_quat_w[:, None, :].expand(-1, num_bodies, -1)
        robot_ref_pos_w_repeat = robot_ref_pos_w[:, None, :].expand(-1, num_bodies, -1)
        robot_ref_quat_w_repeat = robot_ref_quat_w[:, None, :].expand(
            -1, num_bodies, -1
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

        # Cache commonly used frame transforms for observations
        pos_b, ori_b = subtract_frame_transforms(
            robot_ref_pos_w_repeat,
            robot_ref_quat_w_repeat,
            robot_body_pos_w,
            robot_body_quat_w,
        )
        self._robot_body_pos_b = pos_b
        self._robot_body_ori_b_mat = matrix_from_quat(ori_b)

        pos_m, ori_m = subtract_frame_transforms(
            robot_ref_pos_w,
            robot_ref_quat_w,
            ref_pos_w,
            ref_quat_w,
        )
        self._motion_ref_pos_b = pos_m
        self._motion_ref_ori_b_mat = matrix_from_quat(ori_m)

    def _post_update_command(self):
        # 预留接口，供子类在更新 time_steps 后、重采样前进行额外处理
        pass

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
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    # profile property access time
    profile_properties: bool = True

    ref_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    ref_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
