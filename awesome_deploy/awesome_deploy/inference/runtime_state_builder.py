"""Protocol-aware runtime state construction for simulator-side inference."""

from __future__ import annotations

from typing import Any

import numpy as np

from awesome_deploy.inference.protocol import ModelProtocol
from awesome_deploy.inference.types import ModelSignature, RuntimeState


class RuntimeStateBuilder:
    """Builds ``RuntimeState`` values required by a loaded model protocol.

    The builder decouples simulator-side resource collection from
    ``infer.py``. The current implementation intentionally preserves the
    existing single observation source and fans it out to every protocol state
    input. This allows multi-input models to run through the common inference
    path before robot-specific observation producers are fully separated.

    Args:
        protocol: Declarative protocol for the active model.
        signature: Backend-reported input signature used for shape-aware
            fallback observation adaptation.
    """

    def __init__(self, protocol: ModelProtocol, signature: ModelSignature) -> None:
        """Stores protocol metadata needed to build runtime resources."""
        self.protocol = protocol
        self.signature = signature

    def build(self, infer_runner: Any) -> RuntimeState:
        """Builds one runtime state snapshot for a simulator step.

        Args:
            infer_runner: Active ``infere`` instance or compatible object that
                exposes ``update_obs()``, ``time_step``, ``cmd``, and optional
                ``motion``.

        Returns:
            RuntimeState populated with semantic values referenced by protocol
            state bindings and legacy shared resources.
        """
        values = {
            "time_step": infer_runner.time_step,
            "command": infer_runner.cmd,
            "motion": getattr(infer_runner, "motion", None),
        }

        for input_name, binding in self.protocol.input_bindings.items():
            if binding.source_kind != "state" or binding.source_key is None:
                continue
            if binding.source_key in values:
                continue
            values[binding.source_key] = self._build_state_input(
                infer_runner=infer_runner,
                source_key=binding.source_key,
                input_name=input_name,
            )
        return RuntimeState(values=values)

    def get_primary_observation_dim(self) -> int:
        """Returns a compatibility observation dimension for legacy fields.

        The deployment wrapper still exposes ``obs_num`` and ``single_obs``
        even though protocol-driven models may now consume multiple observation
        tensors. This method chooses one representative state input dimension
        so existing code can continue to allocate those arrays.

        Returns:
            Feature dimension of the first state-driven input tensor.

        Raises:
            RuntimeError: If no static batch-first state input exists.
        """
        for input_name, binding in self.protocol.input_bindings.items():
            if binding.source_kind != "state":
                continue
            tensor_spec = self.signature.inputs.get(input_name)
            if tensor_spec is None or len(tensor_spec.shape) < 2:
                continue
            feature_dim = tensor_spec.shape[1]
            if feature_dim is None:
                raise RuntimeError(
                    f"State input '{input_name}' must have a static feature dimension."
                )
            return int(feature_dim)
        raise RuntimeError("Model protocol does not define a compatible state input.")

    def _build_state_input(
        self,
        infer_runner: Any,
        source_key: str,
        input_name: str,
    ) -> np.ndarray:
        """Builds one state-fed model input from the configured obs groups."""
        group_name = self._resolve_group_name(infer_runner, source_key)
        group_obs = np.asarray(
            infer_runner.compute_obs_group(group_name),
            dtype=np.float32,
        ).reshape(-1)
        return self._fit_to_input_dim(group_obs, input_name)

    def _resolve_group_name(self, infer_runner: Any, source_key: str) -> str:
        """Resolves which observation group should feed one protocol state key."""
        input_group_map = getattr(getattr(infer_runner, "obs_cfg", None), "input_group_map", {})
        if source_key in input_group_map:
            return input_group_map[source_key]
        if source_key == "policy_obs":
            return "policy"
        return source_key

    def _fit_to_input_dim(self, obs: np.ndarray, input_name: str) -> np.ndarray:
        """Fits one observation vector to the target input feature width."""
        tensor_spec = self.signature.inputs.get(input_name)
        if tensor_spec is None or len(tensor_spec.shape) < 2:
            raise RuntimeError(
                f"State input '{input_name}' must have a batch-first tensor shape."
            )
        feature_dim = tensor_spec.shape[1]
        if feature_dim is None:
            raise RuntimeError(
                f"State input '{input_name}' must have a static feature dimension."
            )

        target_width = int(feature_dim)
        adapted = np.zeros(target_width, dtype=np.float32)
        copy_width = min(target_width, obs.shape[0])
        adapted[:copy_width] = obs[:copy_width]
        return adapted
