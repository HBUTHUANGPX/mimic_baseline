
import numpy as np
import torch
from abc import ABC, abstractmethod
from collections.abc import Sequence
# from general_motion_tracker_whole_body_teleoperation.tasks.tracking.mdp.commands import MotionCommand
from isaaclab.managers import CommandTerm, CommandTermCfg

class AdaptiveSamplingModule(ABC):
    """Abstract interface for pluggable adaptive sampling strategies.

    Concrete implementations are responsible for tracking sampling statistics,
    building a probability distribution over bins, and updating any internal
    state after each environment step.
    """

    def __init__(self, command: CommandTerm) -> None:
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

    def __init__(self, command: CommandTerm) -> None:
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

    def __init__(self, command: CommandTerm) -> None:
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
