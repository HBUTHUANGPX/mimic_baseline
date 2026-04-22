"""面向 FSQ/iFSQ 动作重构的可复用训练包。"""

from .pipeline import MotionRuntimeBundle, build_autoencoder, build_motion_runtime, resolve_motion_files

__all__ = [
    "MotionRuntimeBundle",
    "__version__",
    "build_autoencoder",
    "build_motion_runtime",
    "resolve_motion_files",
]

__version__ = "0.1.0"
