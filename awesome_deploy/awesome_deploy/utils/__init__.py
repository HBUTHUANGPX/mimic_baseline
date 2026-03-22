from .video_recorder import VideoRecorder

__all__ = ["VideoRecorder"]

import os

# Conveniences to other module directories via relative paths
AWESOME_UTILS_DIR = os.path.abspath(os.path.dirname(__file__))
