"""Shared interfaces for deployment motion reference providers."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class MotionSource(Protocol):
    """Array-oriented motion reference interface consumed by deployment code."""

    joint_order_space: str
    body_order_space: str
    fps: np.ndarray
    time_step_total: int
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray
