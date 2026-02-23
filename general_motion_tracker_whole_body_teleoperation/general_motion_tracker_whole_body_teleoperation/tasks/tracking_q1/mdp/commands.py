from __future__ import annotations

import numpy as np
import os
import torch
import time
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

        # 动态平衡采样: 记录“当前被采样到的 motion 比例”
        # - 形状: (1, num_motions)
        # - 初始均匀分布，后续在 _update_command 中按当前 time_steps 统计更新
        # - 用于 _resample_command 中与 target_dist 计算权重，避免某些 motion 被过度采样
        self.motion_distribution = torch.full(
            (1, self.num_motions),
            1.0 / self.num_motions,
            dtype=torch.float32,
            device=device,
        )  # torch.Size([1, num_motions])

        # 动态平衡采样: 目标分布（按 motion 长度占比）
        # - 长序列在时间维度上覆盖更多步数，因此目标分布按长度归一化
        # - 采样时希望“当前分布”逼近该目标分布
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
        self._motion_lengths_tensor = torch.tensor(
            self.motion.motion_lengths, dtype=torch.long, device=self.device
        )
        self.counts = torch.zeros(
            self.motion.num_motions, dtype=torch.float32, device=self.device
        )
        # per-step cached tensors (computed in _update_state_data)
        self._ref_pos_w = None
        self._ref_quat_w = None
        self._robot_ref_pos_w = None
        self._robot_ref_quat_w = None
        self._robot_body_pos_w = None
        self._robot_body_quat_w = None
        self._robot_body_pos_b = None
        self._robot_body_ori_b_mat = None
        self._motion_ref_pos_b = None
        self._motion_ref_ori_b_mat = None
        self._robot_ref_ori_w_mat = None
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
        # for name in self.motion.extracted_list:
        #     self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        # property timing metrics (ms)
        self._prop_names = [
            "motion_id",
            "motion_group",
            "command",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "ref_pos_w",
            "ref_quat_w",
            "ref_lin_vel_w",
            "ref_ang_vel_w",
            "robot_joint_pos",
            "robot_joint_vel",
            "robot_body_pos_w",
            "robot_body_quat_w",
            "robot_body_lin_vel_w",
            "robot_body_ang_vel_w",
            "robot_ref_pos_w",
            "robot_ref_quat_w",
            "robot_ref_lin_vel_w",
            "robot_ref_ang_vel_w",
        ]
        for name in self._prop_names:
            # self.metrics[f"time_prop_{name}_pre_ms"] = torch.zeros(
            #     self.num_envs, device=self.device
            # )
            self.metrics[f"time_prop_{name}_post_ms"] = torch.zeros(
                self.num_envs, device=self.device
            )
        # self._prop_time_pre_sum = {name: 0.0 for name in self._prop_names}
        self._prop_time_post_sum = {name: 0.0 for name in self._prop_names}
        # timing metrics (ms)
        self.metrics["time_resample_adaptive_ms"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["time_get_env_ids_ms"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["time_update_dist_ms"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["time_record_failures_ms"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["time_update_fail_weights_ms"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["time_write_joint_state_ms"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["time_write_root_state_ms"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self._timing_ms: Dict[str, float] = {}

        # Failure-weighted motion sampling (improved)
        self.motion_fail_counts = torch.zeros(
            self.motion.num_motions, dtype=torch.float32, device=self.device
        )
        self.motion_fail_weights = torch.ones(
            self.motion.num_motions, dtype=torch.float32, device=self.device
        )
        self._fail_update_step = 0
        self._fail_buf_size = max(1, int(self.cfg.fail_update_interval))
        self._fail_buf_ptr = 0
        self._fail_buf_count = 0
        self._fail_term_buf = torch.zeros(
            self._fail_buf_size, self.num_envs, dtype=torch.bool, device=self.device
        )
        self._fail_motion_buf = torch.zeros(
            self._fail_buf_size, self.num_envs, dtype=torch.long, device=self.device
        )
        self._update_state_data()

    @property
    def motion_id(self) -> torch.Tensor:
        return self._profile_property("motion_id", lambda: self.motion._motion_id[self.time_steps])

    @property
    def motion_group(self) -> torch.Tensor:
        return self._profile_property("motion_group", lambda: self.motion._motion_group[self.time_steps])

    @property
    def command(
        self,
    ) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return self._profile_property(
            "command", lambda: torch.cat([self.joint_pos, self.joint_vel], dim=1)
        )

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._profile_property("joint_pos", lambda: self.motion.joint_pos[self.time_steps])

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._profile_property("joint_vel", lambda: self.motion.joint_vel[self.time_steps])

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._profile_property(
            "body_pos_w",
            lambda: self.motion.body_pos_w[self.time_steps]
            + self._env.scene.env_origins[:, None, :],
        )

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._profile_property("body_quat_w", lambda: self.motion.body_quat_w[self.time_steps])

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._profile_property("body_lin_vel_w", lambda: self.motion.body_lin_vel_w[self.time_steps])

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._profile_property("body_ang_vel_w", lambda: self.motion.body_ang_vel_w[self.time_steps])

    @property
    def ref_pos_w(self) -> torch.Tensor:
        if self.cfg.profile_properties:
            return self._profile_property("ref_pos_w", lambda: self._ref_pos_w)
        return self._ref_pos_w

    @property
    def ref_quat_w(self) -> torch.Tensor:
        if self.cfg.profile_properties:
            return self._profile_property("ref_quat_w", lambda: self._ref_quat_w)
        return self._ref_quat_w

    @property
    def ref_lin_vel_w(self) -> torch.Tensor: # tag 2.05ms
        return self._profile_property(
            "ref_lin_vel_w",
            lambda: self.motion.body_lin_vel_w[self.time_steps, self.motion_ref_body_index],
        )

    @property
    def ref_ang_vel_w(self) -> torch.Tensor:
        return self._profile_property(
            "ref_ang_vel_w",
            lambda: self.motion.body_ang_vel_w[self.time_steps, self.motion_ref_body_index],
        )

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self._profile_property("robot_joint_pos", lambda: self.robot.data.joint_pos)

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self._profile_property("robot_joint_vel", lambda: self.robot.data.joint_vel)

    @property
    def robot_body_pos_w(self) -> torch.Tensor: # tag 8.2ms
        if self.cfg.profile_properties:
            return self._profile_property("robot_body_pos_w", lambda: self._robot_body_pos_w)
        return self._robot_body_pos_w

    @property
    def robot_body_quat_w(self) -> torch.Tensor: # tag 10.66ms
        if self.cfg.profile_properties:
            return self._profile_property("robot_body_quat_w", lambda: self._robot_body_quat_w)
        return self._robot_body_quat_w

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor: # tag 10.2ms
        return self._profile_property(
            "robot_body_lin_vel_w", lambda: self.robot.data.body_lin_vel_w[:, self.body_indexes]
        )

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor: # tag 10.5ms
        return self._profile_property(
            "robot_body_ang_vel_w", lambda: self.robot.data.body_ang_vel_w[:, self.body_indexes]
        )

    @property
    def robot_ref_pos_w(self) -> torch.Tensor: # tag 14.5ms
        if self.cfg.profile_properties:
            return self._profile_property("robot_ref_pos_w", lambda: self._robot_ref_pos_w)
        return self._robot_ref_pos_w

    @property
    def robot_ref_quat_w(self) -> torch.Tensor: # tag 20ms
        if self.cfg.profile_properties:
            return self._profile_property("robot_ref_quat_w", lambda: self._robot_ref_quat_w)
        return self._robot_ref_quat_w

    @property
    def robot_ref_lin_vel_w(self) -> torch.Tensor: # tag 2.05ms
        return self._profile_property(
            "robot_ref_lin_vel_w", lambda: self.robot.data.body_lin_vel_w[:, self.robot_ref_body_index]
        )

    @property
    def robot_ref_ang_vel_w(self) -> torch.Tensor: # tag 2.05ms
        return self._profile_property(
            "robot_ref_ang_vel_w", lambda: self.robot.data.body_ang_vel_w[:, self.robot_ref_body_index]
        )

    def _profile_property(self, name: str, fn):
        if not self.cfg.profile_properties:
            return fn()
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        dt_pre = (t1 - t0) * 1000.0
        dt_post = (t2 - t1) * 1000.0
        # if name in self._prop_time_pre_sum:
        #     self._prop_time_pre_sum[name] += dt_pre
        if name in self._prop_time_post_sum:
            self._prop_time_post_sum[name] += dt_post
        return out

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
        # timing metrics (ms)
        self.metrics["time_resample_adaptive_ms"].fill_(
            float(self._timing_ms.get("resample_adaptive", 0.0))
        )
        self.metrics["time_get_env_ids_ms"].fill_(
            float(self._timing_ms.get("get_env_ids", 0.0))
        )
        self.metrics["time_update_dist_ms"].fill_(
            float(self._timing_ms.get("update_dist", 0.0))
        )
        self.metrics["time_record_failures_ms"].fill_(
            float(self._timing_ms.get("record_failures", 0.0))
        )
        self.metrics["time_update_fail_weights_ms"].fill_(
            float(self._timing_ms.get("update_fail_weights", 0.0))
        )
        self.metrics["time_write_joint_state_ms"].fill_(
            float(self._timing_ms.get("write_joint_state", 0.0))
        )
        self.metrics["time_write_root_state_ms"].fill_(
            float(self._timing_ms.get("write_root_state", 0.0))
        )
        for name in self._prop_names:
            # pre = self._prop_time_pre_sum.get(name, 0.0)
            post = self._prop_time_post_sum.get(name, 0.0)
            # self.metrics[f"time_prop_{name}_pre_ms"].fill_(pre)
            self.metrics[f"time_prop_{name}_post_ms"].fill_(post)
            # self._prop_time_pre_sum[name] = 0.0
            self._prop_time_post_sum[name] = 0.0

    def _resample_command(self, env_ids: Sequence[int]):
        # phase = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
        # self.time_steps[env_ids] = (phase * (self.motion.time_step_total - 1)).long()

        if len(env_ids) == 0:
            return
        self._resample_adaptive_sampling(env_ids)
        self._resample_reset_robot_state(env_ids)

    def _resample_adaptive_sampling(self, env_ids: Sequence[int]):
        t0 = time.perf_counter()
        # 动态平衡采样核心:
        # 1) current_dist: 当前 time_steps 覆盖的 motion 分布（由 _update_command 统计）
        # 2) target_dist: 期望分布（按 motion 长度占比）
        # 3) 权重 = target / current，当某个 motion 被“采少了”，权重会变大
        # 4) 对权重再归一化，得到采样概率 probs
        epsilon = 1e-6

        current_dist = self.motion.motion_distribution.squeeze(0)
        target_dist = self.motion.target_dist.squeeze(0)
        base_weights = target_dist / (current_dist + epsilon)
        weights = base_weights * self.motion_fail_weights
        
        # 按动态平衡后的 probs 采样 motion id（每个 env 独立采样）
        motion_ids = torch.multinomial(
            weights, len(env_ids), replacement=True
        )  # (len(env_ids),)

        # 对每个 env 在“选中的 motion 区间”内再采样局部相位（时间步）
        selected_starts = self.motion.motion_indices[motion_ids, 0]  # (len(env_ids),)
        selected_lengths = self._motion_lengths_tensor[motion_ids]
        local_phases = torch.rand((len(env_ids),), device=self.device)  # Uniform [0,1)
        local_steps = (local_phases * (selected_lengths - 1)).long()
        self.time_steps[env_ids] = selected_starts + local_steps
        self.env_motion_ids[env_ids] = motion_ids
        torch.cuda.synchronize()
        self._timing_ms["resample_adaptive"] = (time.perf_counter() - t0) * 1000.0

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
        # timing: write_joint_state_to_sim
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.robot.write_joint_state_to_sim(
            joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
        )
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        self._timing_ms["write_joint_state"] = (t2 - t1) * 1000.0
        self._timing_ms["write_joint_state_pre"] = (t1 - t0) * 1000.0

        # timing: write_root_state_to_sim
        t3 = time.perf_counter()
        torch.cuda.synchronize()
        t4 = time.perf_counter()
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
        torch.cuda.synchronize()
        t5 = time.perf_counter()
        self._timing_ms["write_root_state"] = (t5 - t4) * 1000.0
        self._timing_ms["write_root_state_pre"] = (t4 - t3) * 1000.0

    def _update_command(self): # 入口
        self.time_steps += 1

        env_ids = self._get_env_ids_to_resample()
        self._post_update_command()
        # 根据动态平衡策略为需要重采样的 env 重新分配 time_steps
        self._resample_command(env_ids)
        self._update_state_data()

    def _get_env_ids_to_resample(self) -> torch.Tensor:
        t0 = time.perf_counter()
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
        torch.cuda.synchronize()
        self._timing_ms["get_env_ids"] = (time.perf_counter() - t0) * 1000.0
        return env_ids

    def _update_state_data(self):
        # Compute and cache frequently used tensors once per step.
        ref_pos_w = (
            self.motion.body_pos_w[self.time_steps, self.motion_ref_body_index]
            + self._env.scene.env_origins
        )
        ref_quat_w = self.motion.body_quat_w[
            self.time_steps, self.motion_ref_body_index
        ]
        robot_data_body_pos_w = self.robot.data.body_pos_w.clone()
        robot_data_body_quat_w = self.robot.data.body_quat_w.clone()

        robot_ref_pos_w = robot_data_body_pos_w[:, self.robot_ref_body_index]
        robot_ref_quat_w = robot_data_body_quat_w[:, self.robot_ref_body_index]
        robot_body_pos_w = robot_data_body_pos_w[:, self.body_indexes]
        robot_body_quat_w = robot_data_body_quat_w[:, self.body_indexes]

        self._ref_pos_w = ref_pos_w
        self._ref_quat_w = ref_quat_w
        self._robot_ref_pos_w = robot_ref_pos_w
        self._robot_ref_quat_w = robot_ref_quat_w
        self._robot_body_pos_w = robot_body_pos_w
        self._robot_body_quat_w = robot_body_quat_w
        self._robot_ref_ori_w_mat = matrix_from_quat(robot_ref_quat_w)

        num_bodies = len(self.cfg.body_names)
        ref_pos_w_repeat = ref_pos_w[:, None, :].expand(-1, num_bodies, -1)
        ref_quat_w_repeat = ref_quat_w[:, None, :].expand(-1, num_bodies, -1)
        robot_ref_pos_w_repeat = robot_ref_pos_w[:, None, :].expand(-1, num_bodies, -1)
        robot_ref_quat_w_repeat = robot_ref_quat_w[:, None, :].expand(-1, num_bodies, -1)

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
        self._update_distribution_vectorized()  # 使用向量化方法更新分布统计
        self._record_failures()
        self._update_failure_weights()
        pass

    def _update_distribution_vectorized(self):
        t0 = time.perf_counter()
        # Vectorized: use cached env motion ids
        self.counts = torch.bincount(
            self.env_motion_ids, minlength=self.motion.num_motions
        ).float()
        self.motion.motion_distribution = (self.counts / self.num_envs).unsqueeze(0)
        self.motion_ids = self.env_motion_ids
        torch.cuda.synchronize()
        self._timing_ms["update_dist"] = (time.perf_counter() - t0) * 1000.0

    def _record_failures(self):
        t0 = time.perf_counter()
        # record current step terminated + motion ids
        self._fail_term_buf[self._fail_buf_ptr].copy_(
            self._env.termination_manager.terminated
        )
        self._fail_motion_buf[self._fail_buf_ptr].copy_(self.env_motion_ids)
        self._fail_buf_ptr = (self._fail_buf_ptr + 1) % self._fail_buf_size
        self._fail_buf_count = min(self._fail_buf_count + 1, self._fail_buf_size)
        torch.cuda.synchronize()
        self._timing_ms["record_failures"] = (time.perf_counter() - t0) * 1000.0

    def _update_failure_weights(self):
        t0 = time.perf_counter()
        # low-frequency update of motion-level failure weights
        if self.cfg.fail_update_interval <= 0:
            torch.cuda.synchronize()
            self._timing_ms["update_fail_weights"] = (time.perf_counter() - t0) * 1000.0
            return
        if (self._fail_update_step % self.cfg.fail_update_interval) != 0:
            self._fail_update_step += 1
            torch.cuda.synchronize()
            self._timing_ms["update_fail_weights"] = (time.perf_counter() - t0) * 1000.0
            return

        if self._fail_buf_count > 0:
            term = self._fail_term_buf[: self._fail_buf_count]
            mids = self._fail_motion_buf[: self._fail_buf_count]
            fail_motion_ids = mids[term]
            if fail_motion_ids.numel() > 0:
                counts = torch.bincount(
                    fail_motion_ids, minlength=self.motion.num_motions
                ).float()
            else:
                counts = torch.zeros(
                    self.motion.num_motions, dtype=torch.float32, device=self.device
                )
        else:
            counts = torch.zeros(
                self.motion.num_motions, dtype=torch.float32, device=self.device
            )

        # EMA update of motion-level failure counts
        alpha = float(self.cfg.fail_weight_momentum)
        self.motion_fail_counts = alpha * self.motion_fail_counts + (1.0 - alpha) * counts

        # normalize to weights (avoid zero)
        eps = 1e-6
        w = self.motion_fail_counts + eps
        self.motion_fail_weights = w / (w.sum() / float(self.motion.num_motions))

        # reset buffer window
        self._fail_buf_count = 0
        self._fail_update_step += 1
        torch.cuda.synchronize()
        self._timing_ms["update_fail_weights"] = (time.perf_counter() - t0) * 1000.0

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
    # failure-weighted motion sampling
    fail_update_interval: int = 48
    fail_weight_momentum: float = 0.1
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
