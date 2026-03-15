from __future__ import annotations

"""Motion command implementation for the current refactor stage.

This version supports the following staged features:

1. Multiple trajectories are concatenated into one global timeline.
2. Each environment tracks a single target frame, not a temporal window.
3. Resampling is guided by a failure-aware adaptive distribution over all valid
   global frames.
4. When stepping would cross into the next trajectory or exceed the dataset
   boundary, the environment is resampled from that adaptive distribution.

The goal of this rewrite is to recover a small, reliable baseline before adding
temporal buffers in later iterations.
"""

import os
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING
import math

import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def extract_part(path: str) -> str | None:
    """Extract an artifact-relative motion path.

    Args:
        path: Raw motion file path.

    Returns:
        The path relative to `artifacts/` when the input refers to an `.npz`
        file under that prefix. Otherwise returns `None`.
    """
    if path.startswith("artifacts/"):
        relative_path = path[len("artifacts/") :]
        if relative_path.endswith(".npz"):
            return relative_path
    return None


def get_run_name(motion_file: str) -> str:
    """Convert a motion file path to a compact run name.

    Args:
        motion_file: Raw motion file path.

    Returns:
        A filesystem-friendly run name without the `.npz` suffix.
    """
    if motion_file.startswith("artifacts/"):
        motion_file = motion_file[len("artifacts/") :]
    motion_file = motion_file.replace("/", "_")
    if motion_file.endswith(".npz"):
        motion_file = motion_file[:-4]
    return motion_file


class MotionLoader:
    """Load and organize a set of motion trajectories.

    The loader concatenates all trajectories along the time axis so the command
    term can index them efficiently with a single global frame index. Segment
    metadata is preserved to detect when a frame-by-frame rollout reaches the
    next trajectory boundary.

    Attributes:
        joint_pos: Concatenated joint positions with shape
            `[total_frames, num_joints]`.
        joint_vel: Concatenated joint velocities with shape
            `[total_frames, num_joints]`.
        body_pos_w: Concatenated selected body positions with shape
            `[total_frames, num_bodies, 3]`.
        body_quat_w: Concatenated selected body orientations with shape
            `[total_frames, num_bodies, 4]`.
        body_lin_vel_w: Concatenated selected body linear velocities.
        body_ang_vel_w: Concatenated selected body angular velocities.
        motion_indices: Tensor of shape `[num_motions, 2]` containing each
            trajectory's global `[start, end)` index range.
        new_data_flag: Boolean tensor over the global timeline. Frames that are
            the first frame of a new trajectory are marked `True`.
    """

    def __init__(
        self,
        motion_file_group: dict[str, list[str] | str],
        body_indexes: Sequence[int],
        device: str = "cpu",
    ) -> None:
        """Initialize the motion loader.

        Args:
            motion_file_group: Mapping from semantic group name to motion files.
            body_indexes: Body indices that should be retained from the raw
                motion files.
            device: Torch device on which tensors will be stored.
        """
        self.group_names: list[str] = []
        self.extracted_list: list[str] = []
        self.motion_lengths: list[int] = []
        self.num_motions = 0
        self.fps = None
        self._body_indexes = body_indexes

        joint_pos_list: list[torch.Tensor] = []
        joint_vel_list: list[torch.Tensor] = []
        body_pos_w_list: list[torch.Tensor] = []
        body_quat_w_list: list[torch.Tensor] = []
        body_lin_vel_w_list: list[torch.Tensor] = []
        body_ang_vel_w_list: list[torch.Tensor] = []
        motion_id_list: list[torch.Tensor] = []
        motion_group_list: list[torch.Tensor] = []

        motion_group_index = 0
        for group_name, paths in motion_file_group.items():
            normalized_paths = self._normalize_paths(paths)
            print(f"\nGroup: {group_name}")
            print(f"[INFO] Loading {len(normalized_paths)} motion files for training.")
            print(f"[INFO] load motion file: {normalized_paths}")

            extracted_list = [
                extract_part(path)
                for path in normalized_paths
                if extract_part(path) is not None
            ]

            for local_motion_id, motion_path in enumerate(normalized_paths):
                self._validate_motion_file(motion_path)
                data = np.load(motion_path)
                self._validate_fps(data)

                joint_pos_tensor = torch.tensor(
                    data["joint_pos"], dtype=torch.float32, device=device
                )
                num_frames = joint_pos_tensor.shape[0]

                joint_pos_list.append(joint_pos_tensor)
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
                    torch.full(
                        (num_frames, 1),
                        motion_group_index,
                        dtype=torch.float32,
                        device=device,
                    )
                )
                motion_id_list.append(
                    torch.full(
                        (num_frames, 1),
                        self.num_motions + local_motion_id,
                        dtype=torch.float32,
                        device=device,
                    )
                )
                self.motion_lengths.append(num_frames)

            self.extracted_list.extend(extracted_list)
            self.group_names.append(group_name)
            self.num_motions += len(normalized_paths)
            motion_group_index += 1

        assert self.num_motions > 0, "At least one motion file is required."

        self.joint_pos = torch.cat(joint_pos_list, dim=0)
        self.joint_vel = torch.cat(joint_vel_list, dim=0)
        self._body_pos_w = torch.cat(body_pos_w_list, dim=0)
        self._body_quat_w = torch.cat(body_quat_w_list, dim=0)
        self._body_lin_vel_w = torch.cat(body_lin_vel_w_list, dim=0)
        self._body_ang_vel_w = torch.cat(body_ang_vel_w_list, dim=0)
        self._motion_id = torch.cat(motion_id_list, dim=0)
        self._motion_group = torch.cat(motion_group_list, dim=0)

        self.body_pos_w = self._body_pos_w[:, self._body_indexes]
        self.body_quat_w = self._body_quat_w[:, self._body_indexes]
        self.body_lin_vel_w = self._body_lin_vel_w[:, self._body_indexes]
        self.body_ang_vel_w = self._body_ang_vel_w[:, self._body_indexes]

        self.time_step_total = self.joint_pos.shape[0]
        self.motion_indices = self._build_motion_indices(device)
        self.new_data_flag = self._build_new_data_flag(device)

    def _normalize_paths(self, paths: list[str] | str) -> list[str]:
        """Convert a path input to a normalized list."""
        if isinstance(paths, str):
            return [paths]
        return list(paths)

    def _validate_motion_file(self, motion_path: str) -> None:
        """Ensure the referenced motion file exists."""
        assert os.path.isfile(motion_path), f"Invalid file path: {motion_path}"

    def _validate_fps(self, data: np.lib.npyio.NpzFile) -> None:
        """Ensure all loaded motions share the same fps."""
        if self.fps is None:
            self.fps = data["fps"]
        else:
            assert self.fps == data["fps"], "All motion files must have the same fps."

    def _build_motion_indices(self, device: str) -> torch.Tensor:
        """Build `[start, end)` index ranges for each motion segment."""
        motion_indices = torch.zeros(
            self.num_motions, 2, dtype=torch.long, device=device
        )
        start = 0
        for motion_id, length in enumerate(self.motion_lengths):
            end = start + length
            motion_indices[motion_id] = torch.tensor(
                [start, end], dtype=torch.long, device=device
            )
            start = end
        return motion_indices

    def _build_new_data_flag(self, device: str) -> torch.Tensor:
        """Mark the first frame of every trajectory except the first one."""
        new_data_flag = torch.zeros(
            self.time_step_total, dtype=torch.bool, device=device
        )
        cumulative_length = 0
        for motion_id, length in enumerate(self.motion_lengths):
            if motion_id > 0:
                new_data_flag[cumulative_length] = True
            cumulative_length += length
        return new_data_flag


class MotionCommand(CommandTerm):
    """Reference motion command for multi-trajectory single-frame tracking.

    The current implementation keeps the command in a deliberately simple form:

    - one environment corresponds to one active target frame;
    - all trajectories are sampled from a shared global timeline;
    - resampling uses an adaptive distribution that increases probability mass
      around frames associated with failed episodes.

    The class still exposes the single-frame properties that existing
    observations, rewards, terminations, and events depend on.
    """

    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize the motion command term.

        Args:
            cfg: Motion command configuration.
            env: Owning RL environment.
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
        self._initialize_debug_caches()
        self._initialize_observation_caches()
        self._initialize_metrics()

        # Sample an initial frame for every environment before the first step.
        # self._resample_time_steps(torch.arange(self.num_envs, device=self.device))
        self._update_motion_data()
        self._update_state_data()

    def load_motion(self, motion_file: dict[str, list[str] | str]) -> None:
        """Load motion trajectories and initialize sampling state.

        Args:
            motion_file: Motion file mapping grouped by semantic source.
        """
        self.motion = MotionLoader(motion_file, self.body_indexes, device=self.device)
        self._motion_ends = self.motion.motion_indices[:, 1].contiguous()
        self.time_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.env_motion_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def _initialize_debug_caches(self) -> None:
        """Initialize caches related to visualization and optional extensions."""
        # Window-based observations are intentionally unsupported in stage one.
        self._motion_ref_pos_b_window = None
        self._motion_ref_ori_b_mat_window = None
        self._motion_body_pos_b_window = None
        self._motion_body_ori_b_mat_window = None
        self._joint_pos_delta_window = None
        self._previous_time_steps = None

    def _initialize_observation_caches(self) -> None:
        """Allocate all per-step caches used by observations and rewards."""
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

    def _initialize_metrics(self) -> None:
        """Create metrics required by current logging code.

        The adaptive-sampling metrics summarize how concentrated the resampling
        distribution is at the current step.
        """
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
        self.metrics["error_joint_pos"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_joint_vel"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["sampling_entropy"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["sampling_top1_prob"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["sampling_top1_bin"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.failed_frame_count = torch.zeros(
            self.motion.time_step_total, dtype=torch.float32, device=self.device
        )
        self._current_failed_frame_count = torch.zeros(
            self.motion.time_step_total, dtype=torch.float32, device=self.device
        )

    def _flatten_window(self, tensor: torch.Tensor | None) -> torch.Tensor:
        """Raise an explicit error for unsupported window observations.

        Args:
            tensor: Placeholder tensor from an unsupported window path.

        Raises:
            RuntimeError: Always, because stage one only supports single frames.
        """
        raise RuntimeError(
            "Temporal window observations are not enabled in the stage-one "
            "MotionCommand refactor."
        )

    @property
    def motion_id(self) -> torch.Tensor:
        """Return the current motion id for each environment."""
        return self.motion._motion_id[self.time_steps]

    @property
    def motion_group(self) -> torch.Tensor:
        """Return the current motion group id for each environment."""
        return self.motion._motion_group[self.time_steps]

    @property
    def command(self) -> torch.Tensor:
        """Return the legacy command tensor `[joint_pos, joint_vel]`."""
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        """Return motion joint positions at the current frame."""
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        """Return motion joint velocities at the current frame."""
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Return target body positions in world coordinates."""
        return self._body_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Return target body orientations in world coordinates."""
        return self._body_quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Return target body linear velocities in world coordinates."""
        return self._body_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Return target body angular velocities in world coordinates."""
        return self._body_ang_vel_w

    @property
    def ref_pos_w(self) -> torch.Tensor:
        """Return the target reference-body position in world coordinates."""
        return self._ref_pos_w

    @property
    def ref_quat_w(self) -> torch.Tensor:
        """Return the target reference-body orientation in world coordinates."""
        return self._ref_quat_w

    @property
    def ref_lin_vel_w(self) -> torch.Tensor:
        """Return the target reference-body linear velocity."""
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_ref_body_index]

    @property
    def ref_ang_vel_w(self) -> torch.Tensor:
        """Return the target reference-body angular velocity."""
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_ref_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        """Return the robot joint positions from the simulator."""
        return self._robot_joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        """Return the robot joint velocities from the simulator."""
        return self._robot_joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        """Return robot body positions in world coordinates."""
        return self._robot_body_pos_w

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        """Return robot body orientations in world coordinates."""
        return self._robot_body_quat_w

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        """Return robot body linear velocities in world coordinates."""
        return self._robot_body_lin_vel_w

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        """Return robot body angular velocities in world coordinates."""
        return self._robot_body_ang_vel_w

    @property
    def robot_ref_pos_w(self) -> torch.Tensor:
        """Return robot reference-body position in world coordinates."""
        return self._robot_ref_pos_w

    @property
    def robot_ref_quat_w(self) -> torch.Tensor:
        """Return robot reference-body orientation in world coordinates."""
        return self._robot_ref_quat_w

    @property
    def robot_ref_lin_vel_w(self) -> torch.Tensor:
        """Return robot reference-body linear velocity."""
        return self._robot_ref_lin_vel_w

    @property
    def robot_ref_ang_vel_w(self) -> torch.Tensor:
        """Return robot reference-body angular velocity."""
        return self._robot_ref_ang_vel_w

    def _sample_random_time_steps(self, count: int) -> torch.Tensor:
        """Sample global frame indices uniformly from the dataset.

        Args:
            count: Number of samples to draw.

        Returns:
            Tensor of shape `[count]` containing random global frame indices.
        """
        return torch.randint(
            low=0,
            high=self.motion.time_step_total,
            size=(count,),
            device=self.device,
            dtype=torch.long,
        )

    def _build_sampling_probabilities(self) -> torch.Tensor:
        """Construct the adaptive discrete resampling distribution.

        Returns:
            A probability vector over the concatenated global frame timeline.
        """
        sampling_probabilities = (
            self.failed_frame_count
            + self.cfg.adaptive_uniform_ratio / float(self.motion.time_step_total)
        )
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()
        return sampling_probabilities

    def _update_sampling_metrics(
        self, sampling_probabilities: torch.Tensor
    ) -> None:
        """Update metrics derived from the current sampling distribution.

        Args:
            sampling_probabilities: Adaptive sampling probabilities over global
                frame indices.
        """
        entropy = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        normalized_entropy = entropy / max(math.log(self.motion.time_step_total), 1e-12)
        top1_prob, top1_index = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = normalized_entropy
        self.metrics["sampling_top1_prob"][:] = top1_prob
        self.metrics["sampling_top1_bin"][:] = top1_index.float() / max(
            self.motion.time_step_total, 1
        )

    def _update_env_motion_ids(self, env_ids: Sequence[int]) -> None:
        """Refresh cached motion ids for the specified environments.

        Args:
            env_ids: Environment ids whose motion ids should be updated.
        """
        if len(env_ids) == 0:
            return
        sampled_time_steps = self.time_steps[env_ids]
        self.env_motion_ids[env_ids] = torch.bucketize(
            sampled_time_steps, self._motion_ends, right=True
        )

    def _resample_uniform_time_steps(self, env_ids: Sequence[int]) -> None:
        """Resample single frames uniformly for the specified environments.

        Args:
            env_ids: Environment ids that should receive a new frame.
        """
        if len(env_ids) == 0:
            return
        self.time_steps[env_ids] = self._sample_random_time_steps(len(env_ids))
        self._update_env_motion_ids(env_ids)

    def _accumulate_failed_frames(self, env_ids: Sequence[int]) -> None:
        """Accumulate failure statistics for the specified environments.

        Args:
            env_ids: Environment ids that are being resampled.
        """
        if len(env_ids) == 0:
            return
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if not torch.any(episode_failed):
            return

        # Use the frame that actually produced the failed transition when
        # available. During reset-driven resampling there may be no previous
        # step snapshot yet, so the current frame is the correct fallback.
        if self._previous_time_steps is None:
            failed_time_steps = self.time_steps[env_ids][episode_failed]
        else:
            failed_time_steps = self._previous_time_steps[env_ids][episode_failed]
        self._current_failed_frame_count.index_add_(
            0,
            failed_time_steps,
            torch.ones_like(failed_time_steps, dtype=torch.float32),
        )

    def _resample_time_steps(self, env_ids: Sequence[int]) -> None:
        """Resample single frames using the adaptive failure-guided distribution.

        Args:
            env_ids: Environment ids that should receive a new frame.
        """
        if len(env_ids) == 0:
            return
        self._accumulate_failed_frames(env_ids)
        sampling_probabilities = self._build_sampling_probabilities()
        sampled_time_steps = torch.multinomial(
            sampling_probabilities, len(env_ids), replacement=True
        )
        self.time_steps[env_ids] = sampled_time_steps
        self._update_env_motion_ids(env_ids)
        self._update_sampling_metrics(sampling_probabilities)

    def _update_metrics(self) -> None:
        """Update runtime metrics.

        Stage one focuses on reconstructing the single-frame sampling pathway,
        so the detailed metric computation remains intentionally disabled.
        """
        pass

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Resample motion frames and reset robot state around those frames.

        Args:
            env_ids: Environment ids that should be resampled.
        """
        if len(env_ids) == 0:
            return
        self._resample_time_steps(env_ids)
        # Refresh motion caches so the reset uses the newly sampled frame.
        self._update_motion_data()
        self._resample_reset_robot_state(env_ids)

    def _resample_reset_robot_state(self, env_ids: Sequence[int]) -> None:
        """Reset robot state around the currently sampled motion frames.

        Args:
            env_ids: Environment ids being reset.
        """
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        pose_range_list = [
            self.cfg.pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        pose_ranges = torch.tensor(pose_range_list, device=self.device)
        pose_noise = sample_uniform(
            pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_pos[env_ids] += pose_noise[:, 0:3]

        orientation_delta = quat_from_euler_xyz(
            pose_noise[:, 3], pose_noise[:, 4], pose_noise[:, 5]
        )
        root_ori[env_ids] = quat_mul(orientation_delta, root_ori[env_ids])

        velocity_range_list = [
            self.cfg.velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        velocity_ranges = torch.tensor(velocity_range_list, device=self.device)
        velocity_noise = sample_uniform(
            velocity_ranges[:, 0],
            velocity_ranges[:, 1],
            (len(env_ids), 6),
            device=self.device,
        )
        root_lin_vel[env_ids] += velocity_noise[:, :3]
        root_ang_vel[env_ids] += velocity_noise[:, 3:]

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

    def _update_command(self) -> None:
        """Advance the active frame and resample environments when needed."""
        # Keep a snapshot of the frame that produced the current transition so
        # failed episodes can attribute their sampling credit correctly.
        self._previous_time_steps = self.time_steps.clone()
        self.time_steps += 1
        env_ids = self._get_env_ids_to_resample()
        self._post_update_command()
        self._resample_command(env_ids)
        # Always refresh the caches so the frame actually advances every step.
        self._update_motion_data()
        self._update_state_data()
        self.failed_frame_count = (
            self.cfg.adaptive_alpha * self._current_failed_frame_count
            + (1 - self.cfg.adaptive_alpha) * self.failed_frame_count
        )
        self._current_failed_frame_count.zero_()

    def _update_motion_data(self) -> None:
        """Update motion tensors for the currently active single frame."""
        self._motion_body_pos_w_timestep = self.motion.body_pos_w[self.time_steps]
        self._motion_body_quat_w_timestep = self.motion.body_quat_w[self.time_steps]
        self._motion_body_lin_vel_w_timestep = self.motion.body_lin_vel_w[
            self.time_steps
        ]
        self._motion_body_ang_vel_w_timestep = self.motion.body_ang_vel_w[
            self.time_steps
        ]

        # Shift motion-space body positions into each environment's world origin.
        self._body_pos_w = (
            self._motion_body_pos_w_timestep + self._env.scene.env_origins[:, None, :]
        )
        self._body_quat_w = self._motion_body_quat_w_timestep
        self._body_lin_vel_w = self._motion_body_lin_vel_w_timestep
        self._body_ang_vel_w = self._motion_body_ang_vel_w_timestep

    def _get_env_ids_to_resample(self) -> torch.Tensor:
        """Find environments that have reached a trajectory boundary.

        Returns:
            Environment ids that should be resampled.
        """
        overflow_mask = self.time_steps >= self.motion.time_step_total
        valid_mask = ~overflow_mask
        cross_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if valid_mask.any():
            valid_env_ids = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
            cross_mask[valid_env_ids] = self.motion.new_data_flag[
                self.time_steps[valid_env_ids]
            ]
        total_mask = overflow_mask | cross_mask
        return torch.nonzero(total_mask, as_tuple=False).squeeze(-1)

    def _update_state_data(self) -> None:
        """Update observation caches derived from the simulator state."""
        ref_pos_w = (
            self._motion_body_pos_w_timestep[:, self.motion_ref_body_index]
            + self._env.scene.env_origins
        )
        ref_quat_w = self._motion_body_quat_w_timestep[:, self.motion_ref_body_index]

        robot_body_pos_w = self.robot.data.body_pos_w.clone()
        robot_body_quat_w = self.robot.data.body_quat_w.clone()
        robot_body_lin_vel_w = self.robot.data.body_lin_vel_w.clone()
        robot_body_ang_vel_w = self.robot.data.body_ang_vel_w.clone()
        robot_joint_pos = self.robot.data.joint_pos.clone()
        robot_joint_vel = self.robot.data.joint_vel.clone()

        robot_ref_pos_w = robot_body_pos_w[:, self.robot_ref_body_index]
        robot_ref_quat_w = robot_body_quat_w[:, self.robot_ref_body_index]
        selected_robot_body_pos_w = robot_body_pos_w[:, self.body_indexes]
        selected_robot_body_quat_w = robot_body_quat_w[:, self.body_indexes]
        selected_robot_body_lin_vel_w = robot_body_lin_vel_w[:, self.body_indexes]
        selected_robot_body_ang_vel_w = robot_body_ang_vel_w[:, self.body_indexes]
        robot_ref_lin_vel_w = robot_body_lin_vel_w[:, self.robot_ref_body_index]
        robot_ref_ang_vel_w = robot_body_ang_vel_w[:, self.robot_ref_body_index]

        self._ref_pos_w = ref_pos_w
        self._ref_quat_w = ref_quat_w
        self._robot_ref_pos_w = robot_ref_pos_w
        self._robot_ref_quat_w = robot_ref_quat_w
        self._robot_body_pos_w = selected_robot_body_pos_w
        self._robot_body_quat_w = selected_robot_body_quat_w
        self._robot_body_lin_vel_w = selected_robot_body_lin_vel_w
        self._robot_body_ang_vel_w = selected_robot_body_ang_vel_w
        self._robot_ref_lin_vel_w = robot_ref_lin_vel_w
        self._robot_ref_ang_vel_w = robot_ref_ang_vel_w
        self._robot_joint_pos = robot_joint_pos
        self._robot_joint_vel = robot_joint_vel
        self._robot_ref_ori_w_mat = matrix_from_quat(robot_ref_quat_w)

        num_bodies = len(self.cfg.body_names)
        ref_pos_repeat = ref_pos_w[:, None, :].expand(-1, num_bodies, -1)
        ref_quat_repeat = ref_quat_w[:, None, :].expand(-1, num_bodies, -1)
        robot_ref_pos_repeat = robot_ref_pos_w[:, None, :].expand(-1, num_bodies, -1)
        robot_ref_quat_repeat = robot_ref_quat_w[:, None, :].expand(
            -1, num_bodies, -1
        )

        # Align the motion target to the robot reference frame while preserving
        # only the height difference in translation and the yaw difference in rotation.
        delta_pos_w = ref_pos_repeat - robot_ref_pos_repeat
        delta_pos_w[..., :2] = 0.0
        delta_ori_w = yaw_quat(
            quat_mul(robot_ref_quat_repeat, quat_inv(ref_quat_repeat))
        )

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = (
            robot_ref_pos_repeat
            + delta_pos_w
            + quat_apply(delta_ori_w, self.body_pos_w - ref_pos_repeat)
        )

        robot_body_pos_b, robot_body_ori_b = subtract_frame_transforms(
            robot_ref_pos_repeat,
            robot_ref_quat_repeat,
            selected_robot_body_pos_w,
            selected_robot_body_quat_w,
        )
        self._robot_body_pos_b = robot_body_pos_b
        self._robot_body_ori_b_mat = matrix_from_quat(robot_body_ori_b)

        motion_ref_pos_b, motion_ref_ori_b = subtract_frame_transforms(
            robot_ref_pos_w,
            robot_ref_quat_w,
            ref_pos_w,
            ref_quat_w,
        )
        self._motion_ref_pos_b = motion_ref_pos_b
        self._motion_ref_ori_b_mat = matrix_from_quat(motion_ref_ori_b)

    def _post_update_command(self) -> None:
        """Hook for subclasses.

        The stage-one implementation does not need any extra processing here,
        but the hook is preserved to keep future refactors straightforward.
        """
        pass

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Enable or disable debug visualization markers.

        Args:
            debug_vis: Whether visualization markers should be shown.
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
                for body_name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(
                                prim_path="/Visuals/Command/current/" + body_name
                            )
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(
                                prim_path="/Visuals/Command/goal/" + body_name
                            )
                        )
                    )

            self.current_ref_visualizer.set_visibility(True)
            self.goal_ref_visualizer.set_visibility(True)
            for visualizer_id in range(len(self.cfg.body_names)):
                self.current_body_visualizers[visualizer_id].set_visibility(True)
                self.goal_body_visualizers[visualizer_id].set_visibility(True)
        else:
            if hasattr(self, "current_ref_visualizer"):
                self.current_ref_visualizer.set_visibility(False)
                self.goal_ref_visualizer.set_visibility(False)
                for visualizer_id in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[visualizer_id].set_visibility(False)
                    self.goal_body_visualizers[visualizer_id].set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        """Render debug markers for the current and target poses.

        Args:
            event: Visualization callback event.
        """
        if not self.robot.is_initialized:
            return

        self.current_ref_visualizer.visualize(
            self.robot_ref_pos_w, self.robot_ref_quat_w
        )
        self.goal_ref_visualizer.visualize(self.ref_pos_w, self.ref_quat_w)

        for body_id in range(len(self.cfg.body_names)):
            self.current_body_visualizers[body_id].visualize(
                self.robot_body_pos_w[:, body_id], self.robot_body_quat_w[:, body_id]
            )
            self.goal_body_visualizers[body_id].visualize(
                self.body_pos_relative_w[:, body_id],
                self.body_quat_relative_w[:, body_id],
            )


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for :class:`MotionCommand`.

    Only the single-frame pathway is active in the current stage. Temporal
    buffer configuration fields are kept for forward compatibility and
    documented as reserved.

    Attributes:
        asset_name: Name of the robot articulation in the scene.
        motion_file: Mapping from group name to motion file list.
        reference_body: Body name used as the motion/robot reference frame.
        body_names: Ordered body names tracked by the command term.
        pose_range: Root pose perturbation applied during reset.
        velocity_range: Root velocity perturbation applied during reset.
        joint_position_range: Joint position perturbation applied during reset.
        joint_velocity_range: Reserved for future use.
        adaptive_kernel_size: Reserved for future adaptive smoothing support.
        adaptive_lambda: Reserved for future adaptive smoothing support.
        adaptive_uniform_ratio: Uniform prior mixed into the adaptive frame
            distribution.
        adaptive_alpha: Exponential moving average factor for failure counts.
        history_frames: Reserved for future temporal-buffer support.
        future_frames: Reserved for future temporal-buffer support.
        profile_properties: Upstream profiling switch retained for compatibility.
        ref_visualizer_cfg: Visualization configuration for the reference body.
        body_visualizer_cfg: Visualization configuration for tracked bodies.
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

    # Reserved configuration fields for later refactor stages.
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    history_frames: int = 0
    future_frames: int = 0
    profile_properties: bool = True

    ref_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    ref_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
