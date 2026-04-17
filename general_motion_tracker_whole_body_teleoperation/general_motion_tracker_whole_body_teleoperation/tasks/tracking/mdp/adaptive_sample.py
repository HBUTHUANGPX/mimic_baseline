from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from general_motion_tracker_whole_body_teleoperation.tasks.tracking.mdp.commands import (
        MotionCommand,
    )


class AdaptiveSamplingModule(ABC):
    """Abstract interface for pluggable time-bin sampling strategies."""

    def __init__(
        self, command: CommandTerm, cfg: AdaptiveSamplingModuleCfg
    ) -> None:
        self.command = command
        self.cfg = cfg

    @abstractmethod
    def on_resample_start(
        self, env_ids: Sequence[int], update_failure_statistics: bool
    ) -> None:
        """Update any statistics needed before drawing the next bins."""

    @abstractmethod
    def build_sampling_probabilities(self) -> torch.Tensor:
        """Build the current sampling distribution over valid bins."""

    @abstractmethod
    def on_resample_complete(
        self,
        env_ids: Sequence[int],
        sampled_bins: torch.Tensor,
        update_failure_statistics: bool,
    ) -> None:
        """Persist state for the newly sampled bins if needed."""

    @abstractmethod
    def on_step_end(self) -> None:
        """Finalize any per-step temporary statistics."""


class LegacyBinAdaptiveSampling(AdaptiveSamplingModule):
    """Legacy bin-based sampler matching the original mdp behavior."""

    def __init__(
        self, command: CommandTerm, cfg: LegacyBinAdaptiveSamplingCfg
    ) -> None:
        super().__init__(command, cfg)
        self.bin_failed_count = torch.zeros(
            command.bin_count, dtype=torch.float32, device=command.device
        )
        self.current_bin_failed = torch.zeros(
            command.bin_count, dtype=torch.float32, device=command.device
        )
        self.kernel = torch.tensor(
            [cfg.adaptive_lambda**i for i in range(cfg.adaptive_kernel_size)],
            dtype=torch.float32,
            device=command.device,
        )
        self.kernel = self.kernel / self.kernel.sum()

    def on_resample_start(
        self, env_ids: Sequence[int], update_failure_statistics: bool
    ) -> None:
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
        command = self.command
        sampling_probabilities = (
            self.bin_failed_count
            + self.cfg.adaptive_uniform_ratio / float(command.bin_count)
        )
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),
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
        del env_ids, sampled_bins, update_failure_statistics

    def on_step_end(self) -> None:
        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self.current_bin_failed
            + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self.current_bin_failed.zero_()


class SonicBinAdaptiveSampling(AdaptiveSamplingModule):
    """SONIC-style time-bin sampler."""

    def __init__(
        self, command: CommandTerm, cfg: SonicBinAdaptiveSamplingCfg
    ) -> None:
        super().__init__(command, cfg)
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
            self.cfg.failure_cap_beta * mean_failure_rate,
        )

        capped_sum = capped_failure_rate.sum()
        if capped_sum > 0:
            p_hat = capped_failure_rate / capped_sum
        else:
            p_hat = valid_mask.float() / float(valid_bin_count)

        uniform_distribution = valid_mask.float() / float(valid_bin_count)
        sampling_probabilities = (
            self.cfg.mix_alpha * p_hat
            + (1.0 - self.cfg.mix_alpha) * uniform_distribution
        )
        sampling_probabilities = sampling_probabilities * valid_mask.float()
        return sampling_probabilities / sampling_probabilities.sum()

    def on_resample_complete(
        self,
        env_ids: Sequence[int],
        sampled_bins: torch.Tensor,
        update_failure_statistics: bool,
    ) -> None:
        del update_failure_statistics
        if len(env_ids) == 0:
            return
        self.env_start_bin_ids[env_ids] = sampled_bins

    def on_step_end(self) -> None:
        return


@configclass
class AdaptiveSamplingModuleCfg:
    """Base configuration for time-bin adaptive samplers."""

    class_type: type[AdaptiveSamplingModule] = MISSING


@configclass
class LegacyBinAdaptiveSamplingCfg(AdaptiveSamplingModuleCfg):
    """Configuration for the legacy bin-based sampler."""

    class_type: type[LegacyBinAdaptiveSampling] = LegacyBinAdaptiveSampling
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001


@configclass
class SonicBinAdaptiveSamplingCfg(AdaptiveSamplingModuleCfg):
    """Configuration for the SONIC time-bin sampler."""

    class_type: type[SonicBinAdaptiveSampling] = SonicBinAdaptiveSampling
    mix_alpha: float = 0.1
    failure_cap_beta: float = 200.0
