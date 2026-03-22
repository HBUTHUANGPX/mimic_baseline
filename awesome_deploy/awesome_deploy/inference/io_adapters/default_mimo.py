from __future__ import annotations

import numpy as np

from awesome_deploy.inference.buffers import BufferManager
from awesome_deploy.inference.types import InferenceContext, InferenceResult, ModelSignature


class DefaultMimoAdapter:
    def initialize(self, signature: ModelSignature, buffers: BufferManager) -> None:
        action_spec = signature.outputs.get("actions")
        if action_spec is not None and len(action_spec.shape) > 1:
            action_dim = action_spec.shape[1]
            if action_dim is not None:
                action_dim = int(action_dim)
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
        return {
            "obs": np.asarray(context.obs, dtype=np.float32).reshape(1, -1),
            "time_step": np.asarray([[context.time_step]], dtype=np.float32),
        }

    def parse_outputs(
        self,
        raw_outputs: dict[str, np.ndarray],
        buffers: BufferManager,
    ) -> InferenceResult:
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
