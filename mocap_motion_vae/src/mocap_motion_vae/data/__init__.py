"""mocap_motion_vae 的数据加载入口。

该模块用于统一导出核心数据结构与解析入口，便于上层调用：
- MotionBank / MotionView: 全局时间轴拼接与批量索引
- MotionWindowDataset: 窗口化采样的数据集
- SMPLXClipParser / build_amass_smplx_bank: AMASS/SMPL-X 解析入口
"""

from .bank import MotionBank, MotionView, MotionSample, FeatureSpec
from .dataset import MotionWindowDataset, WindowIndex
from .amass_smplx import (
    SMPLXClip,
    SMPLXFieldSpec,
    SMPLXClipParser,
    build_amass_smplx_bank,
    discover_amass_smplx_files,
)

__all__ = [
    "MotionBank",
    "MotionView",
    "MotionSample",
    "FeatureSpec",
    "WindowIndex",
    "MotionWindowDataset",
    "SMPLXClip",
    "SMPLXFieldSpec",
    "SMPLXClipParser",
    "build_amass_smplx_bank",
    "discover_amass_smplx_files",
]
