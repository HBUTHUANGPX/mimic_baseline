from __future__ import annotations

"""Motion command definitions for tracking tasks.

This module implements a motion loader and a command term that samples
reference motion data from a dataset of trajectories. The implementation now
supports sampling a valid center frame together with a configurable temporal
window `[t - n, ..., t, ..., t + m]` while preserving the original single-frame
interfaces for backward compatibility.
"""

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


def extract_part(path):
    """Extract the artifact-relative npz path from a motion file path.

    Args:
        path: Raw motion file path.

    Returns:
        The artifact-relative path when the input points to an `.npz` file under
        `artifacts/`. Otherwise, returns `None`.
    """
    if path.startswith("artifacts/"):
        relative_path = path[len("artifacts/") :]
        if relative_path.endswith(".npz"):
            return relative_path
    return None


def get_run_name(mf: str) -> str | None:
    """Convert a motion file path into a compact run name.

    Args:
        mf: Motion file path.

    Returns:
        A filesystem-friendly run name without the `.npz` suffix. Returns
        `None` only when the input handling is invalid upstream.
    """
    if mf.startswith("artifacts/"):
        path = mf[len("artifacts/") :]
    path = path.replace("/", "_")
    if path.endswith(".npz"):
        path = path[:-4]
    return path


class MotionLoader:
    """Load motion trajectories and precompute sampling metadata.

    The loader concatenates multiple motion trajectories into a single global
    timeline for efficient batched indexing. In addition to the original
    per-frame tensors, it precomputes the set of valid center frames that can be
    used for temporal window sampling.

    Attributes:
        valid_center_mask: Boolean mask over the concatenated timeline. A frame
            is `True` only if it can serve as the center of the configured
            temporal window.
        valid_center_indices: Global frame indices that are legal center frames.
        valid_center_lookup: Reverse index from global frame index to position in
            `valid_center_indices`, or `-1` when the frame is invalid.
        window_offsets: Relative offsets that define the window order
            `[t - n, ..., t, ..., t + m]`.
    """

    def __init__(
        self,
        motion_file_group: Dict[str, List[str]],
        body_indexes: Sequence[int],
        history_frames: int,
        future_frames: int,
        device: str = "cpu",
    ):
        """Initialize the motion loader.

        Args:
            motion_file_group: Mapping from group name to motion file paths.
            body_indexes: Robot body indices used to slice motion body tensors.
            history_frames: Number of historical frames in the sampling window.
            future_frames: Number of future frames in the sampling window.
            device: Device used to store tensors.
        """
        self.group_names = []
        self.extracted_list = []
        self.num_motions = 0
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.window_size = history_frames + future_frames + 1
        motion_file_group_index = 0

        # Load and concatenate data from all files into a single global timeline.
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

        self.new_data_flag = torch.zeros(
            self.time_step_total, dtype=torch.bool, device=device
        )
        cumulative_len = 0
        for i, length in enumerate(self.motion_lengths):
            if i > 0:
                self.new_data_flag[cumulative_len] = True
            cumulative_len += length

        self.motion_indices = torch.zeros(
            self.num_motions, 2, dtype=torch.long, device=device
        )
        start = 0
        for i, length in enumerate(self.motion_lengths):
            end = start + length
            self.motion_indices[i] = torch.tensor(
                [start, end], dtype=torch.long, device=device
            )
            start = end

        self.window_offsets = torch.arange(
            -self.history_frames, self.future_frames + 1, dtype=torch.long, device=device
        )
        self.body_ang_vel_w = self._body_ang_vel_w[:, self._body_indexes]
        self.body_pos_w = self._body_pos_w[:, self._body_indexes]
        self.body_quat_w = self._body_quat_w[:, self._body_indexes]
        self.body_lin_vel_w = self._body_lin_vel_w[:, self._body_indexes]
        self._build_valid_center_metadata(device)

    def _build_valid_center_metadata(self, device: str) -> None:
        """Build global metadata for valid temporal window center frames.

        Args:
            device: Device on which the metadata tensors are stored.
        """
        self.valid_center_mask = torch.zeros(
            self.time_step_total, dtype=torch.bool, device=device
        )
        self.motion_valid_lengths = torch.zeros(
            self.num_motions, dtype=torch.long, device=device
        )

        for motion_id in range(self.num_motions):
            start, end = self.motion_indices[motion_id]
            valid_start = start + self.history_frames
            valid_end = end - self.future_frames
            valid_length = max(int((valid_end - valid_start).item()), 0)
            self.motion_valid_lengths[motion_id] = valid_length
            if valid_start < valid_end:
                self.valid_center_mask[valid_start:valid_end] = True
            else:
                print(
                    f"[WARN] Motion {motion_id} is shorter than the configured "
                    f"window size {self.window_size} and will be excluded from sampling."
                )

        self.valid_center_indices = torch.nonzero(
            self.valid_center_mask, as_tuple=False
        ).squeeze(-1)
        assert (
            self.valid_center_indices.numel() > 0
        ), "No valid center frames found for the configured window size."

        self.valid_center_lookup = torch.full(
            (self.time_step_total,), -1, dtype=torch.long, device=device
        )
        self.valid_center_lookup[self.valid_center_indices] = torch.arange(
            self.valid_center_indices.shape[0], dtype=torch.long, device=device
        )
        self.valid_center_motion_ids = self._motion_id[
            self.valid_center_indices
        ].long().view(-1)


class MotionCommand(CommandTerm):
    """Command term that provides motion references for tracking.

    The command term keeps `time_steps` as the center-frame index for each
    environment. It exposes the original single-frame properties for backward
    compatibility while internally caching temporal windows for newly added
    observation functions.
    """

    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the motion command term.

        Args:
            cfg: Configuration object for the command term.
            env: Environment that owns the command term.
        """
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
        """Load motion data and initialize command-side caches.

        Args:
            motion_file: Mapping from group name to motion file paths.
        """
        self.motion = MotionLoader(
            motion_file,
            self.body_indexes,
            history_frames=self.cfg.history_frames,
            future_frames=self.cfg.future_frames,
            device=self.device,
        )
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
        self._window_time_steps = None
        self._motion_joint_pos_window = None
        self._motion_joint_vel_window = None
        self._motion_body_pos_w_window = None
        self._motion_body_quat_w_window = None
        self._motion_body_lin_vel_w_window = None
        self._motion_body_ang_vel_w_window = None
        self._motion_ref_pos_b_window = None
        self._motion_ref_ori_b_mat_window = None
        self._motion_body_pos_b_window = None
        self._motion_body_ori_b_mat_window = None
        self._joint_pos_delta_window = None
        self._previous_time_steps = None
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

        self.valid_center_count = self.motion.valid_center_indices.shape[0]
        self.center_failed_count = torch.zeros(
            self.valid_center_count, dtype=torch.float32, device=self.device
        )
        self._current_center_failed = torch.zeros(
            self.valid_center_count, dtype=torch.float32, device=self.device
        )
        self._initialize_time_steps()
        self._update_motion_data()
        self._update_state_data()

    def _initialize_time_steps(self) -> None:
        """Initialize environment center frames from the valid sampling space."""
        initial_center_ids = torch.randint(
            0, self.valid_center_count, (self.num_envs,), device=self.device
        )
        self.time_steps = self.motion.valid_center_indices[initial_center_ids]
        self.env_motion_ids = self.motion.valid_center_motion_ids[initial_center_ids]

    def _flatten_window(self, tensor: torch.Tensor) -> torch.Tensor:
        """Flatten a window tensor into `[num_envs, -1]`.

        Args:
            tensor: Tensor whose leading dimension is `num_envs`.

        Returns:
            The flattened tensor preserving the original time order.
        """
        return tensor.reshape(self.num_envs, -1)

    @property
    def motion_id(self) -> torch.Tensor:
        """Return the motion id of the current center frame."""
        return self.motion._motion_id[self.time_steps]

    @property
    def motion_group(self) -> torch.Tensor:
        """Return the motion group id of the current center frame."""
        return self.motion._motion_group[self.time_steps]

    @property
    def command(
        self,
    ) -> torch.Tensor:
        """Return the legacy single-frame command representation."""
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        """Return joint positions at the center frame."""
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        """Return joint velocities at the center frame."""
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Return world-space body positions at the center frame."""
        return self._body_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Return world-space body orientations at the center frame."""
        return self._body_quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Return world-space body linear velocities at the center frame."""
        return self._body_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Return world-space body angular velocities at the center frame."""
        return self._body_ang_vel_w

    @property
    def ref_pos_w(self) -> torch.Tensor:
        """Return the reference body position at the center frame."""
        return self._ref_pos_w

    @property
    def ref_quat_w(self) -> torch.Tensor:
        """Return the reference body orientation at the center frame."""
        return self._ref_quat_w

    @property
    def ref_lin_vel_w(self) -> torch.Tensor:
        """Return the reference body linear velocity at the center frame."""
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_ref_body_index]

    @property
    def ref_ang_vel_w(self) -> torch.Tensor:
        """Return the reference body angular velocity at the center frame."""
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_ref_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        """Return the robot joint positions from the current simulator state."""
        return self._robot_joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        """Return the robot joint velocities from the current simulator state."""
        return self._robot_joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        """Return robot body positions from the current simulator state."""
        return self._robot_body_pos_w

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        """Return robot body orientations from the current simulator state."""
        return self._robot_body_quat_w

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        """Return robot body linear velocities from the current simulator state."""
        return self._robot_body_lin_vel_w

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        """Return robot body angular velocities from the current simulator state."""
        return self._robot_body_ang_vel_w

    @property
    def robot_ref_pos_w(self) -> torch.Tensor:
        """Return the robot reference body position from the simulator state."""
        return self._robot_ref_pos_w

    @property
    def robot_ref_quat_w(self) -> torch.Tensor:
        """Return the robot reference body orientation from the simulator state."""
        return self._robot_ref_quat_w

    @property
    def robot_ref_lin_vel_w(self) -> torch.Tensor:
        """Return the robot reference body linear velocity."""
        return self._robot_ref_lin_vel_w

    @property
    def robot_ref_ang_vel_w(self) -> torch.Tensor:
        """Return the robot reference body angular velocity."""
        return self._robot_ref_ang_vel_w

    @property
    def joint_pos_window(self) -> torch.Tensor:
        """Return motion joint positions for the full temporal window."""
        return self._motion_joint_pos_window

    @property
    def joint_vel_window(self) -> torch.Tensor:
        """Return motion joint velocities for the full temporal window."""
        return self._motion_joint_vel_window

    def _update_metrics(self):
        """Update diagnostic metrics.

        The detailed tracking metrics are intentionally left disabled here to
        keep the command term lightweight during training runs.
        """
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
        """Resample center frames for the specified environments.

        Args:
            env_ids: Environment indices that need a new valid center frame.
        """
        if len(env_ids) == 0:
            return
        self._resample_adaptive_sampling(env_ids)
        self._update_motion_data()
        self._resample_reset_robot_state(env_ids)

    def _resample_adaptive_sampling(self, env_ids: Sequence[int]):
        """Sample new valid center frames from a discrete adaptive distribution.

        Args:
            env_ids: Environment indices that need resampling.
        """
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            # Use the previous valid center frame for failure accounting. The
            # current `time_steps` may already have advanced into an invalid
            # center frame, in which case reverse lookup would return `-1`.
            failed_time_steps = self._previous_time_steps[env_ids][episode_failed]
            failed_center_ids = self.motion.valid_center_lookup[failed_time_steps]
            valid_failed_mask = failed_center_ids >= 0
            if torch.any(valid_failed_mask):
                failed_center_ids = failed_center_ids[valid_failed_mask]
                self._current_center_failed.index_add_(
                    0,
                    failed_center_ids,
                    torch.ones_like(failed_center_ids, dtype=torch.float32),
                )

        sampling_probabilities = (
            self.center_failed_count
            + self.cfg.adaptive_uniform_ratio / float(self.valid_center_count)
        )
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_center_ids = torch.multinomial(
            sampling_probabilities, len(env_ids), replacement=True
        )
        self.time_steps[env_ids] = self.motion.valid_center_indices[sampled_center_ids]
        self.env_motion_ids[env_ids] = self.motion.valid_center_motion_ids[
            sampled_center_ids
        ]

        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / max(math.log(self.valid_center_count), 1e-12)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.valid_center_count

    def _resample_reset_robot_state(self, env_ids: Sequence[int]):
        """Reset simulator state around the newly sampled center frame.

        Args:
            env_ids: Environment indices being reset.
        """
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
        """Advance the center frame and resample invalid environments."""
        # Snapshot the current valid centers before advancing so adaptive
        # failure accounting can still reference the frame that produced the
        # transition outcome.
        self._previous_time_steps = self.time_steps.clone()
        self.time_steps += 1
        env_ids = self._get_env_ids_to_resample()
        self._post_update_command()
        self._resample_command(env_ids)
        self._update_state_data()
        self.center_failed_count = (
            self.cfg.adaptive_alpha * self._current_center_failed
            + (1 - self.cfg.adaptive_alpha) * self.center_failed_count
        )
        self._current_center_failed.zero_()

    def _update_motion_data(self):
        """Update both center-frame and temporal-window motion caches."""
        # Build the ordered window indices `[t - n, ..., t, ..., t + m]`.
        self._window_time_steps = (
            self.time_steps[:, None] + self.motion.window_offsets[None, :]
        )

        # Legacy single-frame caches remain centered at `time_steps`.
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

        # Window caches are used only by the newly added window observations.
        self._motion_joint_pos_window = self.motion.joint_pos[self._window_time_steps]
        self._motion_joint_vel_window = self.motion.joint_vel[self._window_time_steps]
        self._motion_body_pos_w_window = self.motion.body_pos_w[self._window_time_steps]
        self._motion_body_quat_w_window = self.motion.body_quat_w[
            self._window_time_steps
        ]
        self._motion_body_lin_vel_w_window = self.motion.body_lin_vel_w[
            self._window_time_steps
        ]
        self._motion_body_ang_vel_w_window = self.motion.body_ang_vel_w[
            self._window_time_steps
        ]

    def _get_env_ids_to_resample(self) -> torch.Tensor:
        """Return environments whose center frame is no longer valid.

        Returns:
            Tensor of environment indices that must be resampled.
        """
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
        resample_mask = overflow_mask | (~valid_center_mask)
        return torch.nonzero(resample_mask, as_tuple=False).squeeze(-1)

    def _update_state_data(self):
        """Update simulator-derived state caches and window-relative targets."""
        # Cache current robot state once to avoid repeated property lookups.
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

        # Cache legacy single-frame transforms for existing observations.
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

        self._update_window_state_data(robot_ref_pos_w, robot_ref_quat_w, robot_joint_pos)

    def _update_window_state_data(
        self,
        robot_ref_pos_w: torch.Tensor,
        robot_ref_quat_w: torch.Tensor,
        robot_joint_pos: torch.Tensor,
    ) -> None:
        """Update window-relative motion caches using the current robot pose.

        Args:
            robot_ref_pos_w: Current robot reference-body positions.
            robot_ref_quat_w: Current robot reference-body orientations.
            robot_joint_pos: Current robot joint positions.
        """
        window_size = self.motion.window_size
        num_bodies = len(self.cfg.body_names)

        motion_ref_pos_w_window = (
            self._motion_body_pos_w_window[:, :, self.motion_ref_body_index]
            + self._env.scene.env_origins[:, None, :]
        )
        motion_ref_quat_w_window = self._motion_body_quat_w_window[
            :, :, self.motion_ref_body_index
        ]

        robot_ref_pos_w_window = robot_ref_pos_w[:, None, :].expand(-1, window_size, -1)
        robot_ref_quat_w_window = robot_ref_quat_w[:, None, :].expand(
            -1, window_size, -1
        )
        pos_mw, ori_mw = subtract_frame_transforms(
            robot_ref_pos_w_window,
            robot_ref_quat_w_window,
            motion_ref_pos_w_window,
            motion_ref_quat_w_window,
        )
        self._motion_ref_pos_b_window = pos_mw
        self._motion_ref_ori_b_mat_window = matrix_from_quat(ori_mw)

        motion_body_pos_w_window = (
            self._motion_body_pos_w_window + self._env.scene.env_origins[:, None, None, :]
        )
        motion_body_quat_w_window = self._motion_body_quat_w_window
        robot_ref_pos_w_body = robot_ref_pos_w[:, None, None, :].expand(
            -1, window_size, num_bodies, -1
        )
        robot_ref_quat_w_body = robot_ref_quat_w[:, None, None, :].expand(
            -1, window_size, num_bodies, -1
        )
        pos_bw, ori_bw = subtract_frame_transforms(
            robot_ref_pos_w_body,
            robot_ref_quat_w_body,
            motion_body_pos_w_window,
            motion_body_quat_w_window,
        )
        self._motion_body_pos_b_window = pos_bw
        self._motion_body_ori_b_mat_window = matrix_from_quat(ori_bw)

        robot_joint_pos_window = robot_joint_pos[:, None, :].expand(
            -1, window_size, -1
        )
        self._joint_pos_delta_window = (
            self._motion_joint_pos_window - robot_joint_pos_window
        )

    def _post_update_command(self):
        """Hook for subclasses to inject logic between step advance and resampling."""
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Enable or disable debug visualization markers.

        Args:
            debug_vis: Whether debug visualization should be enabled.
        """
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
        """Render debug visualization for the current center frame.

        Args:
            event: Visualization callback event object.
        """
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
    """Configuration for :class:`MotionCommand`.

    Attributes:
        asset_name: Name of the articulated robot asset in the scene.
        motion_file: Motion file mapping grouped by semantic source.
        reference_body: Reference body shared by robot and motion coordinates.
        body_names: Ordered body names tracked by the command term.
        pose_range: Randomization range applied to the sampled root pose.
        velocity_range: Randomization range applied to the sampled root velocity.
        joint_position_range: Randomization range applied to joint positions.
        joint_velocity_range: Reserved joint velocity randomization range.
        adaptive_kernel_size: Legacy adaptive smoothing parameter retained for
            configuration compatibility.
        adaptive_lambda: Legacy adaptive smoothing decay retained for
            configuration compatibility.
        adaptive_uniform_ratio: Uniform prior mixed into the discrete center
            frame sampling distribution.
        adaptive_alpha: Exponential moving average factor used to update the
            failure-based sampling weights.
        history_frames: Number of frames before the center frame in the window.
        future_frames: Number of frames after the center frame in the window.
        profile_properties: Whether property profiling is enabled upstream.
    """

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
    history_frames: int = 2
    future_frames: int = 8
    profile_properties: bool = True

    ref_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    ref_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
