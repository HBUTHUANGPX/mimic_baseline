from pathlib import Path

import numpy as np
import pytest

from awesome_deploy.inference.protocol_loader import load_protocol_from_file
from awesome_deploy.inference.transform_registry import TRANSFORM_REGISTRY


def test_load_protocol_from_file_supports_multiple_observation_inputs(tmp_path: Path):
    protocol_path = tmp_path / "policy.protocol.yaml"
    protocol_path.write_text(
        "\n".join(
            [
                "version: 1",
                "inputs:",
                "  actor_obs:",
                "    source_kind: state",
                "    source_key: actor_obs",
                "    transform: as_batch_vector",
                "  actor_fsq_obs:",
                "    source_kind: state",
                "    source_key: actor_fsq_obs",
                "    transform: as_batch_vector",
                "outputs:",
                "  actions:",
                "    target_kind: primary",
                "    target_key: q1_action",
                "    transform: flatten_float32",
                "buffer_initializers:",
                "  action:",
                "    init_kind: zeros_from_output",
                "    tensor_name: actions",
                "    axis: 1",
                "per_step_buffer_updates:",
                "  action:",
                "    source_kind: result",
                "    source_key: q1_action",
            ]
        ),
        encoding="utf-8",
    )

    protocol = load_protocol_from_file(protocol_path, TRANSFORM_REGISTRY)

    assert set(protocol.input_bindings) == {"actor_obs", "actor_fsq_obs"}
    assert protocol.output_bindings["actions"].target_key == "q1_action"
    transformed = protocol.input_bindings["actor_fsq_obs"].transform(
        np.zeros(704, dtype=np.float32)
    )
    assert transformed.shape == (1, 704)


def test_load_protocol_from_file_rejects_unknown_transform(tmp_path: Path):
    protocol_path = tmp_path / "bad.protocol.yaml"
    protocol_path.write_text(
        "\n".join(
            [
                "version: 1",
                "inputs:",
                "  obs:",
                "    source_kind: state",
                "    source_key: policy_obs",
                "    transform: missing_transform",
                "outputs:",
                "  actions:",
                "    target_kind: primary",
                "    target_key: action",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing_transform"):
        load_protocol_from_file(protocol_path, TRANSFORM_REGISTRY)


def test_checked_in_q1_protocol_matches_expected_bindings():
    protocol_path = Path(
        "awesome_deploy/policy/q1/"
        "2026-03-20_15-37-52_xsens_all_fsq_s/policy.protocol.yaml"
    )

    protocol = load_protocol_from_file(protocol_path, TRANSFORM_REGISTRY)

    assert set(protocol.input_bindings) == {"actor_obs", "actor_fsq_obs"}
    assert protocol.output_bindings["actions"].target_key == "policy_action"
    assert protocol.buffer_initializers["action"].tensor_name == "actions"
    assert protocol.per_step_buffer_updates["action"].source_key == "policy_action"
