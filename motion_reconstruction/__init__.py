"""面向 FSQ/iFSQ 动作重构的可复用训练包。"""

from .evaluation import ReconstructionResult, reconstruct_motion
from .inference import InferenceSourceBundle, build_hdf5_human_source, build_inference_source, build_raw_source
from .pipeline import MotionRuntimeBundle, build_autoencoder, build_motion_runtime, resolve_motion_files
from .visualization import visualize_hdf5_human_npz, visualize_reconstruction_from_source

__all__ = [
    "InferenceSourceBundle",
    "MotionRuntimeBundle",
    "ReconstructionResult",
    "__version__",
    "build_autoencoder",
    "build_hdf5_human_source",
    "build_inference_source",
    "build_motion_runtime",
    "build_raw_source",
    "reconstruct_motion",
    "resolve_motion_files",
    "visualize_hdf5_human_npz",
    "visualize_reconstruction_from_source",
]

__version__ = "0.1.0"
