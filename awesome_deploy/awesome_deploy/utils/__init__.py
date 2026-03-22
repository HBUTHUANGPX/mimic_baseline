"""Utility exports shared by deployment scripts and runtime modules."""

from .video_recorder import VideoRecorder

__all__ = ["VideoRecorder"]

import os

# Expose the utility package directory for code that needs to resolve bundled
# helper assets relative to the installed package.
AWESOME_UTILS_DIR = os.path.abspath(os.path.dirname(__file__))
