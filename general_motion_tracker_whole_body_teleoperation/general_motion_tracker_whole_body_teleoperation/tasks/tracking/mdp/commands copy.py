from __future__ import annotations

"""Motion command implementation for the current refactor stage.

This version supports the following staged features:

1. Multiple trajectories are concatenated into one global timeline.
2. Each environment tracks a center frame together with a temporal window
   `[t - n, ..., t, ..., t + m]`.
3. Resampling is guided by a failure-aware adaptive distribution over global
   time bins, but only valid center frames can be sampled.
4. Single-frame outputs are derived directly from the center of the temporal
   window instead of being computed independently.
5. Optionally, each environment can be deterministically assigned to one motion
   and execute it sequentially from start to finish without resampling.

The goal of this rewrite is to recover a small, reliable baseline before adding
temporal buffers in later iterations.
"""

import os
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING
import math
from abc import ABC, abstractmethod

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
        history_frames: int,
        future_frames: int,
        device: str = "cpu",
    ) -> None:
        """Initialize the motion loader.

        Args:
            motion_file_group: Mapping from semantic group name to motion files.
            body_indexes: Body indices that should be retained from the raw
                motion files.
            history_frames: Number of historical frames in the temporal window.
            future_frames: Number of future frames in the temporal window.
            device: Torch device on which tensors will be stored.
        """
        self.group_names: list[str] = []
        self.extracted_list: list[str] = []
        self.motion_lengths: list[int] = []
        self.num_motions = 0
        self.fps = None
        self._body_indexes = body_indexes
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.window_size = history_frames + future_frames + 1

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
            # print(f"[INFO] load motion file: {normalized_paths}")

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
        self.window_offsets = torch.arange(
            -self.history_frames,
            self.future_frames + 1,
            dtype=torch.long,
            device=device,
        )
        self.valid_center_mask = self._build_valid_center_mask(device)
        self.valid_center_indices = torch.nonzero(
            self.valid_center_mask, as_tuple=False
        ).squeeze(-1)
        assert (
            self.valid_center_indices.numel() > 0
        ), "No valid center frames found for the configured window size."

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

    def _build_valid_center_mask(self, device: str) -> torch.Tensor:
        """Mark frame indices that can serve as valid window centers.

        A frame is valid when the full temporal window `[t - n, ..., t + m]`
        stays inside the same trajectory.

        Args:
            device: Device used for the output tensor.

        Returns:
            Boolean tensor over the concatenated global timeline.
        """
        valid_center_mask = torch.zeros(
            self.time_step_total, dtype=torch.bool, device=device
        )
        for motion_id in range(self.num_motions):
            start, end = self.motion_indices[motion_id]
            valid_start = start + self.history_frames
            valid_end = end - self.future_frames
            if valid_start < valid_end:
                valid_center_mask[valid_start:valid_end] = True
        return valid_center_mask


class AdaptiveSamplingModule(ABC):
    """Abstract interface for pluggable adaptive sampling strategies.

    Concrete implementations are responsible for tracking sampling statistics,
    building a probability distribution over bins, and updating any internal
    state after each environment step.
    """

    def __init__(self, command: MotionCommand) -> None:
        """Initialize the sampling module.

        Args:
            command: Owning motion command.
        """
        self.command = command

    @abstractmethod
    def on_resample_start(
        self, env_ids: Sequence[int], update_failure_statistics: bool
    ) -> None:
        """Update sampling statistics before a new batch of bins is sampled."""

    @abstractmethod
    def build_sampling_probabilities(self) -> torch.Tensor:
        """Return the current bin sampling probabilities."""

    @abstractmethod
    def on_resample_complete(
        self,
        env_ids: Sequence[int],
        sampled_bins: torch.Tensor,
        update_failure_statistics: bool,
    ) -> None:
        """Record any state that should persist after resampling."""

    @abstractmethod
    def on_step_end(self) -> None:
        """Finalize per-step temporary statistics."""


class LegacyBinAdaptiveSampling(AdaptiveSamplingModule):
    """Bin-based adaptive sampler that preserves the pre-SONIC behavior.

    This implementation:
    - accumulates failure counts in bin space;
    - smooths bin scores with an exponential convolution kernel;
    - mixes the smoothed scores with a uniform prior through additive blending;
    - updates the persistent bin statistics via EMA.
    """

    def __init__(self, command: MotionCommand) -> None:
        """Initialize the legacy sampler state.

        Args:
            command: Owning motion command.
        """
        super().__init__(command)
        self.bin_failed_count = torch.zeros(
            command.bin_count, dtype=torch.float32, device=command.device
        )
        self.current_bin_failed = torch.zeros(
            command.bin_count, dtype=torch.float32, device=command.device
        )
        self.kernel = torch.tensor(
            [command.cfg.adaptive_lambda**i for i in range(command.cfg.adaptive_kernel_size)],
            dtype=torch.float32,
            device=command.device,
        )
        self.kernel = self.kernel / self.kernel.sum()

    def on_resample_start(
        self, env_ids: Sequence[int], update_failure_statistics: bool
    ) -> None:
        """Accumulate failure counts for the bins that caused failed rollouts.

        Args:
            env_ids: Environment ids being resampled.
            update_failure_statistics: Whether runtime failure statistics should
                be updated for this resampling event.
        """
        if not update_failure_statistics or len(env_ids) == 0:
            return
        episode_failed = self.command._env.termination_manager.terminated[env_ids]
        if not torch.any(episode_failed):
            return

        previous_time_steps = (
            self.command.time_steps
            if self.command._previous_time_steps is None
            else self.command._previous_time_steps
        )
        failed_time_steps = previous_time_steps[env_ids][episode_failed]
        failed_bin_ids = torch.clamp(
            failed_time_steps // self.command.bin_frame_count,
            0,
            self.command.bin_count - 1,
        )
        self.current_bin_failed.index_add_(
            0,
            failed_bin_ids,
            torch.ones_like(failed_bin_ids, dtype=torch.float32),
        )

    def build_sampling_probabilities(self) -> torch.Tensor:
        """Construct the legacy adaptive probability distribution over bins.

        Returns:
            Probability vector over valid bins.
        """
        command = self.command
        sampling_probabilities = (
            self.bin_failed_count
            + command.cfg.adaptive_uniform_ratio / float(command.bin_count)
        )
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, command.cfg.adaptive_kernel_size - 1),
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(
            sampling_probabilities, self.kernel.view(1, 1, -1)
        ).view(-1)
        sampling_probabilities = (
            sampling_probabilities * command.valid_sampling_bin_mask.float()
        )
        if sampling_probabilities.sum() <= 0:
            sampling_probabilities = command.valid_sampling_bin_mask.float()
        return sampling_probabilities / sampling_probabilities.sum()

    def on_resample_complete(
        self,
        env_ids: Sequence[int],
        sampled_bins: torch.Tensor,
        update_failure_statistics: bool,
    ) -> None:
        """No-op hook kept for interface compatibility.

        Args:
            env_ids: Environment ids that were resampled.
            sampled_bins: Sampled bin ids.
            update_failure_statistics: Whether failure statistics were enabled.
        """
        del env_ids, sampled_bins, update_failure_statistics

    def on_step_end(self) -> None:
        """Apply EMA to the legacy failure counts and clear the step buffer."""
        self.bin_failed_count = (
            self.command.cfg.adaptive_alpha * self.current_bin_failed
            + (1 - self.command.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self.current_bin_failed.zero_()


class SonicBinAdaptiveSampling(AdaptiveSamplingModule):
    """Strict SONIC-style adaptive sampler.

    The sampler follows the paper logic:
    - bin the motion dataset uniformly in time;
    - record visit and failure counts for the starting bin of each sampled clip;
    - compute per-bin failure rates;
    - cap each failure rate by `beta * mean_failure_rate`;
    - normalize capped failure rates to obtain `p_hat`;
    - mix `p_hat` with a uniform distribution using `alpha`;
    - uniformly sample an initial valid center frame from the selected bin.
    """

    def __init__(self, command: MotionCommand) -> None:
        """Initialize SONIC statistics.

        Args:
            command: Owning motion command.
        """
        super().__init__(command)
        self.bin_visit_count = torch.zeros(
            command.bin_count, dtype=torch.float32, device=command.device
        )
        self.bin_fail_count = torch.zeros(
            command.bin_count, dtype=torch.float32, device=command.device
        )
        self.env_start_bin_ids = torch.zeros(
            command.num_envs, dtype=torch.long, device=command.device
        )

    def on_resample_start(
        self, env_ids: Sequence[int], update_failure_statistics: bool
    ) -> None:
        """Update visit/failure counts for the completed sampled segments.

        Args:
            env_ids: Environment ids being resampled.
            update_failure_statistics: Whether runtime failure statistics should
                be updated for this resampling event.
        """
        if not update_failure_statistics or len(env_ids) == 0:
            return
        start_bin_ids = self.env_start_bin_ids[env_ids]
        self.bin_visit_count.index_add_(
            0,
            start_bin_ids,
            torch.ones_like(start_bin_ids, dtype=torch.float32),
        )

        episode_failed = self.command._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            failed_start_bins = start_bin_ids[episode_failed]
            self.bin_fail_count.index_add_(
                0,
                failed_start_bins,
                torch.ones_like(failed_start_bins, dtype=torch.float32),
            )

    def build_sampling_probabilities(self) -> torch.Tensor:
        """Construct the SONIC sampling distribution over bins.

        Returns:
            Probability vector over valid bins.
        """
        command = self.command
        valid_mask = command.valid_sampling_bin_mask
        valid_bin_count = max(int(valid_mask.sum().item()), 1)

        failure_rate = torch.zeros(
            command.bin_count, dtype=torch.float32, device=command.device
        )
        visited_mask = self.bin_visit_count > 0
        failure_rate[visited_mask] = (
            self.bin_fail_count[visited_mask] / self.bin_visit_count[visited_mask]
        )
        failure_rate = failure_rate * valid_mask.float()

        valid_failure_rates = failure_rate[valid_mask]
        mean_failure_rate = (
            valid_failure_rates.mean()
            if valid_failure_rates.numel() > 0
            else torch.tensor(0.0, device=command.device)
        )
        capped_failure_rate = torch.minimum(
            failure_rate,
            command.cfg.sonic_failure_cap_beta * mean_failure_rate,
        )

        capped_sum = capped_failure_rate.sum()
        if capped_sum > 0:
            p_hat = capped_failure_rate / capped_sum
        else:
            p_hat = valid_mask.float() / float(valid_bin_count)

        uniform_distribution = valid_mask.float() / float(valid_bin_count)
        sampling_probabilities = (
            command.cfg.sonic_mix_alpha * p_hat
            + (1.0 - command.cfg.sonic_mix_alpha) * uniform_distribution
        )
        sampling_probabilities = sampling_probabilities * valid_mask.float()
        return sampling_probabilities / sampling_probabilities.sum()

    def on_resample_complete(
        self,
        env_ids: Sequence[int],
        sampled_bins: torch.Tensor,
        update_failure_statistics: bool,
    ) -> None:
        """Record the starting bins of the newly sampled clips.

        Args:
            env_ids: Environment ids that were resampled.
            sampled_bins: Sampled bin ids.
            update_failure_statistics: Whether failure statistics were enabled.
        """
        del update_failure_statistics
        if len(env_ids) == 0:
            return
        self.env_start_bin_ids[env_ids] = sampled_bins

    def on_step_end(self) -> None:
        """Finalize per-step SONIC sampling state.

        SONIC uses cumulative visit/failure counts, so no EMA update is needed.
        """
        return


class MotionCommand(CommandTerm):
    """Reference motion command for multi-trajectory temporal-window tracking.

    The current implementation keeps the command centered around a single
    center-frame index per environment:

    - one environment corresponds to one active center frame;
    - all trajectories are sampled from a shared global timeline;
    - resampling uses an adaptive bin distribution but only valid center frames
      can be chosen;
    - the temporal buffer `[t - n, ..., t, ..., t + m]` is the primary cached
      representation, and single-frame outputs are derived from its center.
    - an optional sequential-assignment mode binds each environment to a fixed
      motion and advances it monotonically without any resampling.

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
        self._initialize_sampling_metadata()
        self._initialize_assignment_metadata()

        # Sample or assign initial frames before the first step. This path must
        # not touch runtime managers such as the termination manager because
        # manager construction is still in progress.
        if self.cfg.sampling_mode == "assigned_sequential":
            self._initialize_assigned_motion_tracks()
        else:
            self._resample_time_steps(
                torch.arange(self.num_envs, device=self.device),
                update_failure_statistics=False,
            )
        self._update_motion_data()
        self._update_state_data()

    def load_motion(self, motion_file: dict[str, list[str] | str]) -> None:
        """Load motion trajectories and initialize sampling state.

        Args:
            motion_file: Motion file mapping grouped by semantic source.
        """
        self.motion = MotionLoader(
            motion_file,
            self.body_indexes,
            history_frames=self.cfg.history_frames,
            future_frames=self.cfg.future_frames,
            device=self.device,
        )
        self._motion_ends = self.motion.motion_indices[:, 1].contiguous()
        self.time_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.env_motion_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def _initialize_debug_caches(self) -> None:
        """Initialize caches related to visualization and optional extensions."""
        self.center_frame_index = self.cfg.history_frames
        self.window_size = self.motion.window_size
        self._window_time_steps = None
        self._motion_joint_pos_window = None
        self._motion_joint_vel_window = None
        self._motion_body_pos_w_window = None
        self._motion_body_quat_w_window = None
        self._motion_body_lin_vel_w_window = None
        self._motion_body_ang_vel_w_window = None
        self._body_pos_w_window = None
        self._motion_ref_pos_b_window = None
        self._motion_ref_ori_b_mat_window = None
        self._robot_body_pos_b_window = None
        self._robot_body_ori_b_mat_window = None
        self._motion_body_pos_b_window = None
        self._motion_body_ori_b_mat_window = None
        self._joint_pos_delta_window = None
        self._previous_time_steps = None
        self.assigned_motion_ids = None
        self.assigned_motion_starts = None
        self.assigned_motion_ends = None
        self.assigned_last_center_steps = None

    def _initialize_observation_caches(self) -> None:
        """Allocate all per-step caches used by observations and rewards."""
        self._motion_id = None
        self._motion_group = None
        self._command = None
        self._joint_pos = None
        self._joint_vel = None
        self._ref_pos_w = None
        self._ref_quat_w = None
        self._ref_lin_vel_w = None
        self._ref_ang_vel_w = None
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
        self._body_pos_w_center = None
        self._body_quat_w_center = None
        self._body_lin_vel_w_center = None
        self._body_ang_vel_w_center = None
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
        self.bin_frame_count = self._compute_bin_frame_count()
        self.bin_count = max(
            int(math.ceil(self.motion.time_step_total / float(self.bin_frame_count))), 1
        )

    def _compute_bin_frame_count(self) -> int:
        """Compute the number of motion frames contained in each sampling bin.

        The preferred path uses a bin duration expressed in seconds from the
        command configuration, which decouples the adaptive sampling bins from
        the simulation rate. A fallback to the legacy sim-rate-based behavior is
        kept for backward compatibility with older configs.

        Returns:
            Number of motion frames that belong to one adaptive-sampling bin.
        """
        if self.cfg.adaptive_bin_duration_s is not None:
            assert (
                self.cfg.adaptive_bin_duration_s > 0.0
            ), "adaptive_bin_duration_s must be positive."
            return max(
                int(round(float(self.motion.fps) * self.cfg.adaptive_bin_duration_s)),
                1,
            )

        # Legacy fallback: keep historical behavior when old configs do not set
        # an explicit bin duration yet.
        return max(
            int(1.0 / (self._env.cfg.decimation * self._env.cfg.sim.dt)),
            1,
        )
    def _initialize_sampling_metadata(self) -> None:
        """Build sampling metadata for valid center-frame bin sampling."""
        valid_center_indices = self.motion.valid_center_indices
        self.valid_center_bin_ids = torch.clamp(
            valid_center_indices // self.bin_frame_count,
            0,
            self.bin_count - 1,
        )
        self.valid_center_count_per_bin = torch.bincount(
            self.valid_center_bin_ids, minlength=self.bin_count
        )
        self.valid_sampling_bin_mask = self.valid_center_count_per_bin > 0

        max_valid_centers_per_bin = max(
            int(self.valid_center_count_per_bin.max().item()), 1
        )
        self.bin_valid_center_indices = torch.full(
            (self.bin_count, max_valid_centers_per_bin),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        for bin_id in range(self.bin_count):
            count = int(self.valid_center_count_per_bin[bin_id].item())
            if count == 0:
                continue
            bin_valid_centers = valid_center_indices[self.valid_center_bin_ids == bin_id]
            self.bin_valid_center_indices[bin_id, :count] = bin_valid_centers

        self.valid_center_lookup = torch.full(
            (self.motion.time_step_total,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.valid_center_lookup[self.motion.valid_center_indices] = torch.arange(
            self.motion.valid_center_indices.shape[0],
            dtype=torch.long,
            device=self.device,
        )
        self.adaptive_sampler = self._build_adaptive_sampler()

    def _initialize_assignment_metadata(self) -> None:
        """Initialize metadata for deterministic motion assignment mode."""
        self.assigned_motion_ids = None
        self.assigned_motion_starts = None
        self.assigned_motion_ends = None
        self.assigned_last_center_steps = None

    def _build_adaptive_sampler(self) -> AdaptiveSamplingModule:
        """Instantiate the configured adaptive sampling module.

        Returns:
            Adaptive sampling module selected by configuration.

        Raises:
            ValueError: If the configured sampler type is unknown.
        """
        if self.cfg.adaptive_sampler_type == "legacy_bin":
            return LegacyBinAdaptiveSampling(self)
        if self.cfg.adaptive_sampler_type == "sonic":
            return SonicBinAdaptiveSampling(self)
        raise ValueError(
            f"Unsupported adaptive sampler type: {self.cfg.adaptive_sampler_type}"
        )

    def _initialize_assigned_motion_tracks(self) -> None:
        """Assign one unique motion trajectory to each environment.

        This mode is intended for deterministic evaluation or dataset
        playback. Each environment starts from the first valid center frame of
        its assigned motion and then advances monotonically without resampling.

        Raises:
            AssertionError: If the number of environments does not match the
                number of motions while one-to-one assignment is requested.
        """
        assert (
            self.num_envs == self.motion.num_motions
        ), (
            "assigned_sequential sampling requires num_envs to equal the number "
            "of loaded motion trajectories."
        )
        self.assigned_motion_ids = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )
        assigned_motion_ranges = self.motion.motion_indices[self.assigned_motion_ids]
        self.assigned_motion_starts = assigned_motion_ranges[:, 0]
        self.assigned_motion_ends = assigned_motion_ranges[:, 1]
        self.assigned_last_center_steps = (
            self.assigned_motion_ends - self.cfg.future_frames - 1
        )
        self.time_steps = self.assigned_motion_starts + self.cfg.history_frames
        print("_initialize_assigned_motion_tracks: ",self.assigned_motion_starts,self.assigned_motion_ends,self.cfg.history_frames,self.time_steps)
        assert torch.all(
            self.time_steps <= self.assigned_last_center_steps
        ), (
            "assigned_sequential sampling requires every assigned motion to have "
            "at least history_frames + future_frames + 1 frames."
        )
        self.env_motion_ids = self.assigned_motion_ids.clone()

    def _advance_assigned_motion_tracks(self) -> None:
        """Advance deterministically assigned motions without resampling."""
        self._previous_time_steps = self.time_steps.clone()
        self.time_steps += 1
        print("_advance_assigned_motion_tracks: ",self.assigned_motion_starts,self.assigned_motion_ends,self.time_steps)
        if self.cfg.freeze_assigned_motion_at_end:
            self.time_steps = torch.minimum(
                self.time_steps, self.assigned_last_center_steps
            )
        else:
            overflow_mask = self.time_steps > self.assigned_last_center_steps
            self.time_steps[overflow_mask] = self.assigned_motion_starts[overflow_mask] + self.cfg.history_frames

    def _flatten_window(self, tensor: torch.Tensor | None) -> torch.Tensor:
        """Flatten a temporal-window tensor into `[num_envs, -1]`.

        Args:
            tensor: Window tensor whose leading dimension is `num_envs`.

        Returns:
            Flattened tensor preserving the fixed time order
            `[t - n, ..., t, ..., t + m]`.
        """
        if tensor is None:
            raise RuntimeError("Requested window cache has not been initialized.")
        return tensor.reshape(self.num_envs, -1)

    @property
    def motion_id(self) -> torch.Tensor:
        """Return the current motion id for each environment."""
        return self._motion_id

    @property
    def motion_group(self) -> torch.Tensor:
        """Return the current motion group id for each environment."""
        return self._motion_group

    @property
    def command(self) -> torch.Tensor:
        """Return the legacy command tensor `[joint_pos, joint_vel]`."""
        return self._command

    @property
    def joint_pos(self) -> torch.Tensor:
        """Return motion joint positions at the current center frame."""
        return self._joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        """Return motion joint velocities at the current center frame."""
        return self._joint_vel

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
        return self._ref_lin_vel_w

    @property
    def ref_ang_vel_w(self) -> torch.Tensor:
        """Return the target reference-body angular velocity."""
        return self._ref_ang_vel_w

    @property
    def joint_pos_window(self) -> torch.Tensor:
        """Return motion joint positions for the full temporal window."""
        return self._motion_joint_pos_window

    @property
    def joint_vel_window(self) -> torch.Tensor:
        """Return motion joint velocities for the full temporal window."""
        return self._motion_joint_vel_window

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

    def _update_sampling_metrics(
        self, sampling_probabilities: torch.Tensor
    ) -> None:
        """Update metrics derived from the current sampling distribution.

        Args:
            sampling_probabilities: Adaptive sampling probabilities over global
                time bins.
        """
        entropy = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        normalized_entropy = entropy / max(math.log(self.bin_count), 1e-12)
        top1_prob, top1_index = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = normalized_entropy
        self.metrics["sampling_top1_prob"][:] = top1_prob
        self.metrics["sampling_top1_bin"][:] = top1_index.float() / max(self.bin_count, 1)

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

    def _sample_time_steps_from_bins(self, sampled_bins: torch.Tensor) -> torch.Tensor:
        """Map sampled bins to valid center frames inside those bins.

        Args:
            sampled_bins: Bin indices sampled from the adaptive distribution.

        Returns:
            Valid center-frame indices sampled from the chosen bins.
        """
        # First sample a continuous position inside each bin, then snap it to
        # the nearest valid center frame that belongs to the same bin.
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
        sampled_time_steps = torch.empty_like(candidate_time_steps)

        for bin_id in torch.unique(sampled_bins).tolist():
            env_mask = sampled_bins == bin_id
            valid_count = int(self.valid_center_count_per_bin[bin_id].item())
            valid_centers = self.bin_valid_center_indices[bin_id, :valid_count]
            if valid_count == 1:
                sampled_time_steps[env_mask] = valid_centers[0]
                continue

            bin_candidates = candidate_time_steps[env_mask]
            right_indices = torch.bucketize(bin_candidates, valid_centers)
            right_indices = torch.clamp(right_indices, max=valid_count - 1)
            left_indices = torch.clamp(right_indices - 1, min=0)
            left_centers = valid_centers[left_indices]
            right_centers = valid_centers[right_indices]

            choose_right = torch.abs(right_centers - bin_candidates) < torch.abs(
                bin_candidates - left_centers
            )
            sampled_time_steps[env_mask] = torch.where(
                choose_right, right_centers, left_centers
            )

        return sampled_time_steps

    def _resample_time_steps(
        self,
        env_ids: Sequence[int],
        update_failure_statistics: bool = True,
    ) -> None:
        """Resample single frames using the adaptive bin-based distribution.

        Args:
            env_ids: Environment ids that should receive a new frame.
            update_failure_statistics: Whether failure counts should be updated
                before sampling. This must be disabled during cold start because
                the termination manager is not available yet.
        """
        if len(env_ids) == 0:
            return
        self.adaptive_sampler.on_resample_start(env_ids, update_failure_statistics)
        sampling_probabilities = self.adaptive_sampler.build_sampling_probabilities()
        sampled_bins = torch.multinomial(
            sampling_probabilities, len(env_ids), replacement=True
        )
        self.time_steps[env_ids] = self._sample_time_steps_from_bins(sampled_bins)
        self._update_env_motion_ids(env_ids)
        self.adaptive_sampler.on_resample_complete(
            env_ids, sampled_bins, update_failure_statistics
        )
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
        if self.cfg.sampling_mode == "assigned_sequential":
            # In assigned-sequential mode, resampling means restoring the
            # deterministic per-env motion assignment instead of drawing a new
            # motion/frame from the adaptive sampler. This branch is required
            # because Isaac Lab may call `_resample_command()` during reset or
            # initialization outside of `_update_command()`.
            env_ids_tensor = torch.as_tensor(
                env_ids, dtype=torch.long, device=self.device
            )
            self.time_steps[env_ids_tensor] = (
                self.assigned_motion_starts[env_ids_tensor] + self.cfg.history_frames
            )
            self.env_motion_ids[env_ids_tensor] = self.assigned_motion_ids[
                env_ids_tensor
            ]
        else:
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
        if self.cfg.sampling_mode == "assigned_sequential":
            self._advance_assigned_motion_tracks()
            self._update_motion_data()
            self._update_state_data()
            return

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
        self.adaptive_sampler.on_step_end()

    def _update_motion_data(self) -> None:
        """Update motion tensors for the active temporal window.

        The temporal window is the primary cache. The legacy single-frame
        caches are sliced from the center frame of this window.
        """
        self._window_time_steps = (
            self.time_steps[:, None] + self.motion.window_offsets[None, :]
        )
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

        self._motion_body_pos_w_timestep = self._motion_body_pos_w_window[
            :, self.center_frame_index
        ]
        self._motion_body_quat_w_timestep = self._motion_body_quat_w_window[
            :, self.center_frame_index
        ]
        self._motion_body_lin_vel_w_timestep = self._motion_body_lin_vel_w_window[
            :, self.center_frame_index
        ]
        self._motion_body_ang_vel_w_timestep = self._motion_body_ang_vel_w_window[
            :, self.center_frame_index
        ]
        self._joint_pos = self._motion_joint_pos_window[:, self.center_frame_index]
        self._joint_vel = self._motion_joint_vel_window[:, self.center_frame_index]
        self._motion_id = self.motion._motion_id[self.time_steps]
        self._motion_group = self.motion._motion_group[self.time_steps]
        self._command = torch.cat([self._joint_pos, self._joint_vel], dim=1)

        env_origins = self._env.scene.env_origins
        self._body_pos_w_window = (
            self._motion_body_pos_w_window + env_origins[:, None, None, :]
        )
        self._body_pos_w = self._body_pos_w_window[:, self.center_frame_index]
        self._body_quat_w = self._motion_body_quat_w_timestep
        self._body_lin_vel_w = self._motion_body_lin_vel_w_timestep
        self._body_ang_vel_w = self._motion_body_ang_vel_w_timestep
        self._ref_lin_vel_w = self._motion_body_lin_vel_w_timestep[
            :, self.motion_ref_body_index
        ]
        self._ref_ang_vel_w = self._motion_body_ang_vel_w_timestep[
            :, self.motion_ref_body_index
        ]

    def _get_env_ids_to_resample(self) -> torch.Tensor:
        """Find environments whose center frame is no longer window-valid.

        Returns:
            Environment ids that should be resampled.
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

    def _update_state_data(self) -> None:
        """Update observation caches derived from the simulator state.

        All window-relative targets are expressed relative to the current robot
        reference-body pose at the center frame.
        """
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

        # Express the robot's current tracked bodies in the robot reference
        # frame itself.
        #
        # Inputs:
        #   1. `robot_ref_pos_repeat`, `robot_ref_quat_repeat`
        #      The current robot reference-body pose in world frame, expanded
        #      to one copy per tracked body.
        #   2. `selected_robot_body_pos_w`, `selected_robot_body_quat_w`
        #      The current simulator-measured pose of each tracked robot body
        #      in world frame.
        #
        # Outputs:
        #   1. `robot_body_pos_b`
        #      Body positions represented in the robot reference frame.
        #   2. `robot_body_ori_b`
        #      Body orientations represented relative to the robot reference
        #      frame.
        #
        # Role in the command pipeline:
        #   This branch describes "what the robot is doing now" in a body-local
        #   coordinate system that is invariant to the robot's global position
        #   and heading. Downstream observations use it as the proprioceptive
        #   body-pose baseline that can be compared against motion targets.
        robot_body_pos_b, robot_body_ori_b = subtract_frame_transforms(
            robot_ref_pos_repeat,
            robot_ref_quat_repeat,
            selected_robot_body_pos_w,
            selected_robot_body_quat_w,
        )
        self._robot_body_pos_b = robot_body_pos_b
        self._robot_body_ori_b_mat = matrix_from_quat(robot_body_ori_b)

        # Express the motion reference body in the current robot reference
        # frame.
        #
        # Inputs:
        #   1. `robot_ref_pos_w`, `robot_ref_quat_w`
        #      The robot's current reference-body pose in world frame.
        #   2. `ref_pos_w`, `ref_quat_w`
        #      The sampled motion's reference-body pose at the current center
        #      frame in world frame.
        #
        # Outputs:
        #   1. `motion_ref_pos_b`
        #      The translation from the current robot reference body to the
        #      motion reference body, expressed in the robot reference frame.
        #   2. `motion_ref_ori_b`
        #      The relative orientation from the current robot reference body
        #      to the motion reference body.
        #
        # Role in the command pipeline:
        #   This branch describes "where the motion target is" relative to the
        #   robot's current pose. In other words, the previous block encodes the
        #   robot's present body configuration, while this block encodes the
        #   motion target anchor that the policy should track.
        motion_ref_pos_b, motion_ref_ori_b = subtract_frame_transforms(
            robot_ref_pos_w,
            robot_ref_quat_w,
            ref_pos_w,
            ref_quat_w,
        )
        self._motion_ref_pos_b = motion_ref_pos_b
        self._motion_ref_ori_b_mat = matrix_from_quat(motion_ref_ori_b)
        self._update_window_state_data(
            robot_ref_pos_w,
            robot_ref_quat_w,
            robot_joint_pos,
        )

    def _update_window_state_data(
        self,
        robot_ref_pos_w: torch.Tensor,
        robot_ref_quat_w: torch.Tensor,
        robot_joint_pos: torch.Tensor,
    ) -> None:
        """Update caches for temporal-window observations.

        Args:
            robot_ref_pos_w: Current robot reference-body positions.
            robot_ref_quat_w: Current robot reference-body orientations.
            robot_joint_pos: Current robot joint positions.
        """
        num_bodies = len(self.cfg.body_names)
        window_size = self.window_size

        motion_ref_pos_w_window = self._body_pos_w_window[
            :, :, self.motion_ref_body_index
        ]
        motion_ref_quat_w_window = self._motion_body_quat_w_window[
            :, :, self.motion_ref_body_index
        ]

        robot_ref_pos_w_window = robot_ref_pos_w[:, None, :].expand(-1, window_size, -1)
        robot_ref_quat_w_window = robot_ref_quat_w[:, None, :].expand(
            -1, window_size, -1
        )
        # Window version of `motion_ref_pos_b` / `motion_ref_ori_b`.
        #
        # Inputs:
        #   1. `robot_ref_pos_w_window`, `robot_ref_quat_w_window`
        #      The current robot reference-body pose, broadcast to every time
        #      slot in the command window. This means the whole window is
        #      always interpreted relative to the robot's current pose.
        #   2. `motion_ref_pos_w_window`, `motion_ref_quat_w_window`
        #      The sampled motion reference-body pose for each temporal offset
        #      in `[t-n, ..., t, ..., t+m]`.
        #
        # Outputs:
        #   1. `motion_ref_pos_b_window`
        #      For every window element, the motion reference-body translation
        #      expressed in the current robot reference frame.
        #   2. `motion_ref_ori_b_window`
        #      For every window element, the motion reference-body orientation
        #      relative to the current robot reference frame.
        #
        # Role in the command pipeline:
        #   This is the temporal extension of the single-frame target-anchor
        #   observation. It tells the policy where the motion reference body
        #   was, is, and will be, all described in one consistent coordinate
        #   frame tied to the robot's current state.
        motion_ref_pos_b_window, motion_ref_ori_b_window = subtract_frame_transforms(
            robot_ref_pos_w_window,
            robot_ref_quat_w_window,
            motion_ref_pos_w_window,
            motion_ref_quat_w_window,
        )
        self._motion_ref_pos_b_window = motion_ref_pos_b_window
        self._motion_ref_ori_b_mat_window = matrix_from_quat(motion_ref_ori_b_window)

        robot_ref_pos_w_body = robot_ref_pos_w[:, None, None, :].expand(
            -1, window_size, num_bodies, -1
        )
        robot_ref_quat_w_body = robot_ref_quat_w[:, None, None, :].expand(
            -1, window_size, num_bodies, -1
        )
        # Window version of `robot_body_pos_b` / `robot_body_ori_b`.
        #
        # Inputs:
        #   1. `robot_ref_pos_w_body`, `robot_ref_quat_w_body`
        #      The current robot reference-body pose, broadcast to every body
        #      and every time slot in the command window.
        #   2. `self._body_pos_w_window`, `self._motion_body_quat_w_window`
        #      The world-frame body pose sequence associated with the sampled
        #      command window, ordered as `[t-n, ..., t, ..., t+m]`.
        #
        # Outputs:
        #   1. `robot_body_pos_b_window`
        #      Body positions for each time slot, expressed in the current
        #      robot reference frame.
        #   2. `robot_body_ori_b_window`
        #      Body orientations for each time slot, expressed relative to the
        #      current robot reference frame.
        #
        # Role in the command pipeline:
        #   This is the temporal extension of the body-pose observation. It
        #   lets downstream window observations compare a full body-pose
        #   trajectory against the current robot-centered coordinate frame
        #   rather than only seeing the center frame.
        robot_body_pos_b_window, robot_body_ori_b_window = subtract_frame_transforms(
            robot_ref_pos_w_body,
            robot_ref_quat_w_body,
            self._body_pos_w_window,
            self._motion_body_quat_w_window,
        )
        self._robot_body_pos_b_window = robot_body_pos_b_window
        self._robot_body_ori_b_mat_window = matrix_from_quat(robot_body_ori_b_window)

        # Preserve the legacy cache names so existing downstream code continues
        # to work while the window-observation API is migrated to the aligned
        # `robot_body_*_window` naming scheme.
        self._motion_body_pos_b_window = self._robot_body_pos_b_window
        self._motion_body_ori_b_mat_window = self._robot_body_ori_b_mat_window

        robot_joint_pos_window = robot_joint_pos[:, None, :].expand(-1, window_size, -1)
        self._joint_pos_delta_window = (
            self._motion_joint_pos_window - robot_joint_pos_window
        )

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

    The current stage supports both single-frame outputs and temporal-window
    caches. Single-frame outputs are always derived from the center of the
    configured temporal window.

    Attributes:
        asset_name: Name of the robot articulation in the scene.
        motion_file: Mapping from group name to motion file list.
        reference_body: Body name used as the motion/robot reference frame.
        body_names: Ordered body names tracked by the command term.
        pose_range: Root pose perturbation applied during reset.
        velocity_range: Root velocity perturbation applied during reset.
        joint_position_range: Joint position perturbation applied during reset.
        joint_velocity_range: Reserved for future use.
        sampling_mode: High-level motion playback mode. Use `assigned_sequential`
            to bind one environment to one motion and run it from start to end.
        freeze_assigned_motion_at_end: Whether assigned-sequential mode should
            stop at the final valid center frame instead of looping.
        adaptive_sampler_type: Name of the adaptive sampling module to use.
        adaptive_bin_duration_s: Duration of each adaptive-sampling bin in
            seconds. When left unset, the implementation falls back to the
            legacy sim-rate-based bin size for backward compatibility.
        adaptive_kernel_size: Size of the smoothing kernel used in bin space.
        adaptive_lambda: Exponential decay factor for the bin-smoothing kernel.
        adaptive_uniform_ratio: Uniform prior mixed into the adaptive bin
            distribution in the legacy sampler.
        adaptive_alpha: Exponential moving average factor for bin failure counts.
        sonic_mix_alpha: SONIC mixture factor for combining hard-bin sampling
            and uniform coverage.
        sonic_failure_cap_beta: SONIC cap coefficient for failure-rate clipping.
        history_frames: Number of frames before the center frame in the window.
        future_frames: Number of frames after the center frame in the window.
        profile_properties: Upstream profiling switch retained for compatibility.
        ref_visualizer_cfg: Visualization configuration for the reference body.
        body_visualizer_cfg: Visualization configuration for tracked bodies.
    """

    class_type: type = MotionCommand

    asset_name: str = MISSING
    motion_file: dict[str, list[str] | str] = MISSING
    reference_body: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    joint_velocity_range: tuple[float, float] = (-0.52, 0.52)

    sampling_mode: str = "adaptive"  # "adaptive" | "assigned_sequential"
    freeze_assigned_motion_at_end: bool = True
    adaptive_sampler_type: str = "sonic" # "sonic" | "legacy_bin"
    adaptive_bin_duration_s: float | None = None
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    sonic_mix_alpha: float = 0.1
    sonic_failure_cap_beta: float = 200.0
    history_frames: int = 0
    future_frames: int = 4
    profile_properties: bool = True

    ref_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    ref_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
