from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch


def probability_from_expected_recovery_horizon(expected_recovery_horizon: int) -> float:
    """Converts an expected recovery horizon into the Bernoulli termination probability."""

    if expected_recovery_horizon <= 0:
        raise ValueError(f"expected_recovery_horizon must be positive, got {expected_recovery_horizon}.")
    return 1.0 / float(expected_recovery_horizon)


def resolve_probabilistic_term_names(
    *,
    term_names: Sequence[str],
    time_out_term_names: Iterable[str],
    configured_term_names: Iterable[str] | None,
) -> tuple[str, ...]:
    """Resolves which termination terms should use probabilistic termination."""

    time_out_term_names = set(time_out_term_names)
    if configured_term_names is None:
        return tuple(name for name in term_names if name not in time_out_term_names)
    return tuple(configured_term_names)


def apply_probabilistic_termination_gate(
    *,
    term_values: torch.Tensor,
    term_names: Sequence[str],
    time_out_term_names: Iterable[str],
    probabilistic_term_names: Iterable[str],
    probability: float,
    random_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Applies a Bernoulli gate to selected termination terms.

    Timeout terms and non-selected termination terms remain deterministic. The
    returned raw probabilistic buffer is useful for diagnostics and logging.
    """

    if term_values.ndim != 2:
        raise ValueError(f"term_values must have shape (num_envs, num_terms), got {tuple(term_values.shape)}.")
    if len(term_names) != term_values.shape[1]:
        raise ValueError(
            f"term_names length ({len(term_names)}) must match term_values second dimension ({term_values.shape[1]})."
        )
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}.")

    num_envs = term_values.shape[0]
    device = term_values.device
    time_out_term_names = set(time_out_term_names)
    probabilistic_term_names = set(probabilistic_term_names)

    time_out_mask = torch.tensor(
        [name in time_out_term_names for name in term_names],
        dtype=torch.bool,
        device=device,
    )
    probabilistic_mask = torch.tensor(
        [name in probabilistic_term_names and name not in time_out_term_names for name in term_names],
        dtype=torch.bool,
        device=device,
    )
    deterministic_mask = ~(time_out_mask | probabilistic_mask)

    false_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
    truncated = term_values[:, time_out_mask].any(dim=1) if time_out_mask.any() else false_buf.clone()
    raw_probabilistic = (
        term_values[:, probabilistic_mask].any(dim=1) if probabilistic_mask.any() else false_buf.clone()
    )
    deterministic_terminated = (
        term_values[:, deterministic_mask].any(dim=1) if deterministic_mask.any() else false_buf.clone()
    )

    if probability <= 0.0:
        probabilistic_terminated = false_buf
    elif probability >= 1.0:
        probabilistic_terminated = raw_probabilistic
    else:
        if random_values is None:
            random_values = torch.rand(num_envs, device=device)
        else:
            random_values = random_values.to(device=device)
            if random_values.shape != (num_envs,):
                raise ValueError(
                    f"random_values must have shape ({num_envs},), got {tuple(random_values.shape)}."
                )
        probabilistic_terminated = raw_probabilistic & (random_values < probability)

    return truncated, deterministic_terminated | probabilistic_terminated, raw_probabilistic
