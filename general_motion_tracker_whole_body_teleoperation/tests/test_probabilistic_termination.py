from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "general_motion_tracker_whole_body_teleoperation"
    / "tasks"
    / "tracking"
    / "probabilistic_termination.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "probabilistic_termination_under_test",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
probabilistic_termination = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault(
    "probabilistic_termination_under_test",
    probabilistic_termination,
)
_SPEC.loader.exec_module(probabilistic_termination)

apply_probabilistic_termination_gate = (
    probabilistic_termination.apply_probabilistic_termination_gate
)
probability_from_expected_recovery_horizon = (
    probabilistic_termination.probability_from_expected_recovery_horizon
)
resolve_probabilistic_term_names = (
    probabilistic_termination.resolve_probabilistic_term_names
)


def test_expected_recovery_horizon_maps_to_original_pterm():
    probability = probability_from_expected_recovery_horizon(200)

    assert probability == 0.005


def test_default_probabilistic_terms_are_all_non_timeout_terms():
    term_names = ("time_out", "ref_pos", "ref_ori", "fallen")

    probabilistic_terms = resolve_probabilistic_term_names(
        term_names=term_names,
        time_out_term_names={"time_out"},
        configured_term_names=None,
    )

    assert probabilistic_terms == ("ref_pos", "ref_ori", "fallen")


def test_probabilistic_gate_preserves_timeout_and_deterministic_terminations():
    term_names = ("time_out", "ref_pos", "ref_ori", "fallen")
    term_values = torch.tensor(
        [
            [True, False, False, False],
            [False, True, False, False],
            [False, False, False, True],
        ],
        dtype=torch.bool,
    )

    truncated, terminated, raw_probabilistic = apply_probabilistic_termination_gate(
        term_values=term_values,
        term_names=term_names,
        time_out_term_names={"time_out"},
        probabilistic_term_names={"ref_pos", "ref_ori"},
        probability=0.0,
        random_values=torch.zeros(3),
    )

    assert truncated.tolist() == [True, False, False]
    assert terminated.tolist() == [False, False, True]
    assert raw_probabilistic.tolist() == [False, True, False]


def test_probabilistic_gate_applies_bernoulli_to_selected_terms_only():
    term_names = ("time_out", "ref_pos", "ref_ori", "fallen")
    term_values = torch.tensor(
        [
            [False, True, False, False],
            [False, False, True, False],
            [False, True, True, False],
            [False, False, False, True],
        ],
        dtype=torch.bool,
    )

    truncated, terminated, raw_probabilistic = apply_probabilistic_termination_gate(
        term_values=term_values,
        term_names=term_names,
        time_out_term_names={"time_out"},
        probabilistic_term_names={"ref_pos", "ref_ori"},
        probability=0.5,
        random_values=torch.tensor([0.49, 0.50, 0.1, 0.99]),
    )

    assert truncated.tolist() == [False, False, False, False]
    assert terminated.tolist() == [True, False, True, True]
    assert raw_probabilistic.tolist() == [True, True, True, False]
