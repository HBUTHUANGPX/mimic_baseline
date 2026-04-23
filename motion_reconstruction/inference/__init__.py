"""推理阶段的数据来源适配。"""

from .sources import InferenceSourceBundle, build_hdf5_human_source, build_inference_source, build_raw_source

__all__ = [
    "InferenceSourceBundle",
    "build_hdf5_human_source",
    "build_inference_source",
    "build_raw_source",
]
