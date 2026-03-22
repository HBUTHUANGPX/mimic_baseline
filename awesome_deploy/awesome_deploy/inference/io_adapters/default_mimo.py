"""Default name-driven MIMO adapter for the current policy export format."""

from __future__ import annotations

import numpy as np

from awesome_deploy.inference.buffers import BufferManager
from awesome_deploy.inference.types import (
    InferenceContext,
    InferenceResult,
    ModelSignature,
)


class DefaultMimoAdapter:
    """Maps the current ``obs`` + ``time_step`` policy protocol to tensors.

    This adapter preserves the existing deployment contract while removing ONNX
    specifics from ``infere``. It assumes:

    - The model exposes an ``obs`` input.
    - The model exposes a ``time_step`` input.
    - The main action output is named ``actions``.

    Additional outputs are passed through untouched in the returned
    ``InferenceResult.outputs`` mapping.
    """

    def initialize(self, signature: ModelSignature, buffers: BufferManager) -> None:
        """Initializes per-episode buffers from static model metadata.

        Args:
            signature: Name-based model signature exposed by the backend.
            buffers: Mutable buffer store owned by the inference engine.
        """
        action_spec = signature.outputs.get("actions")
        if action_spec is not None and len(action_spec.shape) > 1:
            action_dim = action_spec.shape[1]
            if action_dim is not None:
                action_dim = int(action_dim)
                # Pre-allocate action history so downstream code can assume the
                # buffers exist from the first rollout step onward.
                buffers.set("action_dim", action_dim)
                zero_action = np.zeros(action_dim, dtype=np.float32)
                buffers.set("action", zero_action.copy())
                buffers.set("prev_action", zero_action.copy())
                buffers.set("prev_prev_action", zero_action.copy())
        buffers.set("time_step", 1)

    def build_inputs(
        self,
        context: InferenceContext,
        buffers: BufferManager,
    ) -> dict[str, np.ndarray]:
        """Builds backend tensors for one policy step.

        Args:
            context: Simulator-side semantic state for this step.
            buffers: Current persistent inference buffers.

        Returns:
            Backend-ready numpy arrays keyed by input tensor name.
        """
        del buffers
        return {
            "obs": np.asarray(context.obs, dtype=np.float32).reshape(1, -1),
            "time_step": np.asarray([[context.time_step]], dtype=np.float32),
        }

    def parse_outputs(
        self,
        raw_outputs: dict[str, np.ndarray],
        buffers: BufferManager,
    ) -> InferenceResult:
        """Parses raw backend outputs into semantic results and state updates.

        Args:
            raw_outputs: Tensor outputs keyed by model output name.
            buffers: Persistent buffer store holding previous action history.

        Returns:
            ``InferenceResult`` containing passthrough outputs, the primary
            action vector, and the next-step buffer updates.
        """
        action = raw_outputs.get("actions")
        if action is not None:
            action = np.asarray(action, dtype=np.float32).reshape(-1)

        return InferenceResult(
            outputs={name: np.asarray(value) for name, value in raw_outputs.items()},
            primary_action=action,
            state_updates={
                "prev_prev_action": buffers.get("prev_action"),
                "prev_action": buffers.get("action"),
                "action": action,
                "time_step": int(buffers.get("time_step", 1)) + 1,
            },
        )
