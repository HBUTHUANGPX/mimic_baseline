from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager

from .probabilistic_termination import (
    apply_probabilistic_termination_gate,
    probability_from_expected_recovery_horizon,
    resolve_probabilistic_term_names,
)


class ProbabilisticTerminationManager(TerminationManager):
    """Termination manager that gates selected failure terms with a Bernoulli draw."""

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.expected_recovery_horizon = int(
            getattr(env.cfg, "probabilistic_termination_expected_recovery_horizon", 200)
        )
        self.probabilistic_probability = probability_from_expected_recovery_horizon(
            self.expected_recovery_horizon
        )
        self._raw_probabilistic_terminated_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        time_out_term_names = {
            name for name, term_cfg in zip(self._term_names, self._term_cfgs) if term_cfg.time_out
        }
        configured_term_names = getattr(env.cfg, "probabilistic_termination_term_names", None)
        self.probabilistic_term_names = resolve_probabilistic_term_names(
            term_names=self._term_names,
            time_out_term_names=time_out_term_names,
            configured_term_names=configured_term_names,
        )
        self._probabilistic_term_name_set = set(self.probabilistic_term_names)

        missing_terms = sorted(self._probabilistic_term_name_set.difference(self._term_names))
        if missing_terms:
            raise ValueError(
                "Probabilistic termination terms are not active in the termination manager: "
                f"{missing_terms}. Active terms: {self._term_names}."
            )

    @property
    def raw_probabilistic_terminated(self) -> torch.Tensor:
        """Raw selected bad-tracking signal before the Bernoulli termination gate."""

        return self._raw_probabilistic_terminated_buf

    def compute(self) -> torch.Tensor:
        """Computes dones while probabilistically gating selected failure terms."""

        for i, term_cfg in enumerate(self._term_cfgs):
            self._term_dones[:, i] = term_cfg.func(self._env, **term_cfg.params)

        time_out_term_names = {
            name for name, term_cfg in zip(self._term_names, self._term_cfgs) if term_cfg.time_out
        }
        self._truncated_buf, self._terminated_buf, self._raw_probabilistic_terminated_buf = (
            apply_probabilistic_termination_gate(
                term_values=self._term_dones,
                term_names=self._term_names,
                time_out_term_names=time_out_term_names,
                probabilistic_term_names=self._probabilistic_term_name_set,
                probability=self.probabilistic_probability,
            )
        )

        rows = self._term_dones.any(dim=1).nonzero(as_tuple=True)[0]
        if rows.numel() > 0:
            self._last_episode_dones[rows] = self._term_dones[rows]

        return self._truncated_buf | self._terminated_buf

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        terms = list(super().get_active_iterable_terms(env_idx))
        terms.append(
            (
                "raw_probabilistic_terminated",
                [self._raw_probabilistic_terminated_buf[env_idx].float().cpu().item()],
            )
        )
        return terms


class ProbabilisticTrackingRLEnv(ManagerBasedRLEnv):
    """Manager-based RL env that uses :class:`ProbabilisticTerminationManager`."""

    def load_managers(self):
        # Keep IsaacLab's manager initialization order; only the termination manager changes.
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        ManagerBasedEnv.load_managers(self)

        self.termination_manager = ProbabilisticTerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)

        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)

        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        self._configure_gym_env_spaces()

        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
