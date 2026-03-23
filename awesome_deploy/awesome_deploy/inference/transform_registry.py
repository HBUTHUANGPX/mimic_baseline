"""Named transforms used by externally-declared model protocols."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


Transform = Callable[[Any], Any]


def as_batch_vector(value):
    """Converts a 1-D feature vector into a batch-first float32 tensor."""
    return np.asarray(value, dtype=np.float32).reshape(1, -1)


def as_batch_scalar(value):
    """Converts a scalar into a ``(1, 1)`` float32 tensor."""
    return np.asarray([[value]], dtype=np.float32)


def flatten_float32(value):
    """Flattens a tensor into a 1-D float32 vector."""
    return np.asarray(value, dtype=np.float32).reshape(-1)


def increment_int(value):
    """Increments a scalar-like value and converts it to ``int``."""
    return int(value) + 1


TRANSFORM_REGISTRY: dict[str, Transform] = {
    "as_batch_scalar": as_batch_scalar,
    "as_batch_vector": as_batch_vector,
    "flatten_float32": flatten_float32,
    "increment_int": increment_int,
}
