"""Loads declarative model protocol files into runtime protocol objects."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import yaml

from awesome_deploy.inference.protocol import (
    BufferInitializer,
    InputBinding,
    ModelProtocol,
    OutputBinding,
)


def load_protocol_from_file(
    protocol_path: str | Path,
    transform_registry: dict[str, Callable],
) -> ModelProtocol:
    """Loads a YAML protocol file and converts it into ``ModelProtocol``.

    Args:
        protocol_path: Filesystem path to ``policy.protocol.yaml``.
        transform_registry: Mapping from transform names declared in the YAML
            file to executable callables.

    Returns:
        Parsed ``ModelProtocol`` instance.

    Raises:
        ValueError: If the protocol file is malformed or references unsupported
            fields or transforms.
    """
    protocol_path = Path(protocol_path)
    data = _load_yaml(protocol_path)
    _validate_top_level(data, protocol_path)
    return ModelProtocol(
        input_bindings=_parse_input_bindings(
            data.get("inputs", {}),
            transform_registry,
            protocol_path,
        ),
        output_bindings=_parse_output_bindings(
            data.get("outputs", {}),
            transform_registry,
            protocol_path,
        ),
        buffer_initializers=_parse_buffer_initializers(
            data.get("buffer_initializers", {}),
            protocol_path,
        ),
        per_step_buffer_updates=_parse_input_bindings(
            data.get("per_step_buffer_updates", {}),
            transform_registry,
            protocol_path,
        ),
    )


def _load_yaml(protocol_path: Path) -> dict:
    """Reads YAML data from disk."""
    with protocol_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid protocol '{protocol_path}': top-level YAML must be a mapping.")
    return data


def _validate_top_level(data: dict, protocol_path: Path) -> None:
    """Validates top-level protocol structure."""
    if data.get("version") != 1:
        raise ValueError(
            f"Invalid protocol '{protocol_path}': only version 1 is supported."
        )
    for key in ("inputs", "outputs"):
        if key not in data:
            raise ValueError(f"Invalid protocol '{protocol_path}': missing top-level key '{key}'.")
        if not isinstance(data[key], dict):
            raise ValueError(
                f"Invalid protocol '{protocol_path}': top-level key '{key}' must be a mapping."
            )


def _parse_input_bindings(
    binding_map: dict,
    transform_registry: dict[str, Callable],
    protocol_path: Path,
) -> dict[str, InputBinding]:
    """Parses input bindings or per-step buffer updates."""
    parsed = {}
    for name, raw_binding in binding_map.items():
        if not isinstance(raw_binding, dict):
            raise ValueError(
                f"Invalid protocol '{protocol_path}': binding '{name}' must be a mapping."
            )
        source_kind = raw_binding.get("source_kind")
        if source_kind not in {"state", "buffer", "result", "constant"}:
            raise ValueError(
                f"Invalid protocol '{protocol_path}': binding '{name}' has unsupported "
                f"source_kind '{source_kind}'."
            )
        source_key = raw_binding.get("source_key")
        value = raw_binding.get("value")
        if source_kind == "constant":
            if "value" not in raw_binding:
                raise ValueError(
                    f"Invalid protocol '{protocol_path}': constant binding '{name}' requires 'value'."
                )
        elif source_key is None:
            raise ValueError(
                f"Invalid protocol '{protocol_path}': binding '{name}' requires 'source_key'."
            )
        parsed[name] = InputBinding(
            source_kind=source_kind,
            source_key=source_key,
            value=value,
            transform=_resolve_transform(raw_binding.get("transform"), transform_registry, protocol_path),
        )
    return parsed


def _parse_output_bindings(
    binding_map: dict,
    transform_registry: dict[str, Callable],
    protocol_path: Path,
) -> dict[str, OutputBinding]:
    """Parses output bindings and validates the primary output count."""
    parsed = {}
    primary_count = 0
    for name, raw_binding in binding_map.items():
        if not isinstance(raw_binding, dict):
            raise ValueError(
                f"Invalid protocol '{protocol_path}': output binding '{name}' must be a mapping."
            )
        target_kind = raw_binding.get("target_kind")
        if target_kind not in {"primary", "output"}:
            raise ValueError(
                f"Invalid protocol '{protocol_path}': output binding '{name}' has unsupported "
                f"target_kind '{target_kind}'."
            )
        if target_kind == "primary":
            primary_count += 1
        target_key = raw_binding.get("target_key")
        if not target_key:
            raise ValueError(
                f"Invalid protocol '{protocol_path}': output binding '{name}' requires 'target_key'."
            )
        parsed[name] = OutputBinding(
            target_kind=target_kind,
            target_key=target_key,
            transform=_resolve_transform(raw_binding.get("transform"), transform_registry, protocol_path),
        )
    if primary_count > 1:
        raise ValueError(
            f"Invalid protocol '{protocol_path}': only one output may be marked as primary."
        )
    return parsed


def _parse_buffer_initializers(
    initializer_map: dict,
    protocol_path: Path,
) -> dict[str, BufferInitializer]:
    """Parses buffer initializer declarations."""
    parsed = {}
    for name, raw_initializer in initializer_map.items():
        if not isinstance(raw_initializer, dict):
            raise ValueError(
                f"Invalid protocol '{protocol_path}': buffer initializer '{name}' must be a mapping."
            )
        init_kind = raw_initializer.get("init_kind")
        if init_kind not in {"constant", "zeros_from_output"}:
            raise ValueError(
                f"Invalid protocol '{protocol_path}': buffer initializer '{name}' has unsupported "
                f"init_kind '{init_kind}'."
            )
        if init_kind == "constant" and "value" not in raw_initializer:
            raise ValueError(
                f"Invalid protocol '{protocol_path}': constant initializer '{name}' requires 'value'."
            )
        if init_kind == "zeros_from_output" and not raw_initializer.get("tensor_name"):
            raise ValueError(
                f"Invalid protocol '{protocol_path}': zeros_from_output initializer '{name}' requires "
                f"'tensor_name'."
            )
        parsed[name] = BufferInitializer(
            init_kind=init_kind,
            value=raw_initializer.get("value"),
            tensor_name=raw_initializer.get("tensor_name"),
            axis=int(raw_initializer.get("axis", 0)),
        )
    return parsed


def _resolve_transform(
    transform_name: str | None,
    transform_registry: dict[str, Callable],
    protocol_path: Path,
):
    """Resolves an optional transform name through the provided registry."""
    if transform_name is None:
        return None
    if transform_name not in transform_registry:
        raise ValueError(
            f"Invalid protocol '{protocol_path}': transform '{transform_name}' is not registered."
        )
    return transform_registry[transform_name]
