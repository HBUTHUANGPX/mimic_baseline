"""Factory for selecting the deployment motion reference provider."""

from __future__ import annotations

from collections.abc import Sequence

from awesome_deploy.utils.motion_loader import MotionLoader
from awesome_deploy.utils.motion_source import MotionSource
from awesome_deploy.utils.realtime_motion_source import build_realtime_motion_source


def build_motion_source(
    cfg,
    body_indexes: Sequence[int],
    device: str = "cpu",
    body_names: Sequence[str] | None = None,
) -> MotionSource:
    """Builds the configured motion source for deployment.

    Offline mode preserves the legacy ``MotionLoader`` implementation so the
    existing simulator and observation code can continue to consume numpy
    arrays unchanged. Realtime mode is reserved for the next phase.
    """

    if cfg.motion_source == "offline":
        return MotionLoader(cfg.motion_source_uri, body_indexes, device)
    if cfg.motion_source == "realtime":
        if body_names is None:
            raise ValueError("Realtime motion source requires body_names.")
        return build_realtime_motion_source(cfg, list(body_names))
    raise ValueError(f"Unsupported motion_source: {cfg.motion_source}")
