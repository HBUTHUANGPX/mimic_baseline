"""最小完整示例：从 SMPL-X npz 构建 MotionBank 并采样窗口。

调用链：
1) 解析命令行参数（AMASS 路径 / SMPL-X 模型路径 / window / stride）
2) 准备 npz 文件列表（可选生成临时数据）
3) build_amass_smplx_bank 解析并构建 MotionBank
4) MotionWindowDataset 进行窗口化采样
5) 打印样本 shapes 作为验证

注意：脚本需要可用的 SMPL-X 模型路径（--smplx-model-path 或环境变量 SMPLX_MODEL_PATH）。
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np

# 允许直接运行脚本时自动加入本地 src 路径（无需手动设置 PYTHONPATH）
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mocap_motion_vae.data import (
    FeatureSpec,
    MotionWindowDataset,
    SMPLXFieldSpec,
    build_amass_smplx_bank,
    discover_amass_smplx_files,
)


def _write_npz(path: Path, frames: int, fps: int) -> None:
    """写入一个最小 SMPL-X npz 文件。

    Args:
        path: npz 输出路径。
        frames: 帧数。
        fps: 帧率。
    """
    np.savez(
        path,
        fps=np.array(fps),
        pose_body=np.zeros((frames, 63), dtype=np.float32),
        root_orient=np.zeros((frames, 3), dtype=np.float32),
        trans=np.zeros((frames, 3), dtype=np.float32),
        betas=np.zeros((16,), dtype=np.float32),
        gender=np.array("neutral"),
    )


def _prepare_files(amass_root: str | None) -> List[str]:
    """准备 npz 文件列表。

    Args:
        amass_root: AMASS 根目录或单个 npz 文件路径，可为空。

    Returns:
        npz 文件路径列表。
    """
    if amass_root:
        path = Path(amass_root)
        if path.is_file():
            return [str(path)]
        files = discover_amass_smplx_files(path)
        if len(files) == 0:
            raise FileNotFoundError(f"未在目录中找到 npz 文件: {amass_root}")
        return files

    temp_dir = Path(tempfile.mkdtemp(prefix="amass_minimal_"))
    _write_npz(temp_dir / "clip_0.npz", frames=120, fps=60)
    _write_npz(temp_dir / "clip_1.npz", frames=90, fps=60)
    return [str(temp_dir / "clip_0.npz"), str(temp_dir / "clip_1.npz")]


def main() -> None:
    """执行最小流程：构建 MotionBank 并采样窗口。"""
    parser = argparse.ArgumentParser(description="最小完整示例：SMPL-X 数据加载")
    parser.add_argument(
        "--amass-root",
        type=str,
        default=None,
        help="AMASS/SMPL-X 数据根目录或单个 npz 文件路径（不提供则生成临时数据）",
    )
    parser.add_argument(
        "--smplx-model-path",
        type=str,
        default=None,
        help="SMPL-X 模型目录路径（可选，提供后会计算 joints/vertices/full_pose）",
    )
    parser.add_argument("--window", type=int, default=60, help="窗口长度（帧）")
    parser.add_argument("--stride", type=int, default=30, help="窗口步长（帧）")
    args = parser.parse_args()

    files = _prepare_files(args.amass_root)

    smplx_model_path = args.smplx_model_path or os.getenv("SMPLX_MODEL_PATH")
    if smplx_model_path is None:
        raise RuntimeError(
            "未找到 SMPL-X 模型路径，请使用 --smplx-model-path 或设置 SMPLX_MODEL_PATH。"
        )

    field_spec = SMPLXFieldSpec()
    bank = build_amass_smplx_bank(files, field_spec, smplx_model_path=smplx_model_path)

    feature_spec = FeatureSpec(
        inputs=("pose_body", "root_orient", "trans", "joints"),
        targets=("pose_body",),
        static=("betas",),
    )
    dataset = MotionWindowDataset(
        bank, window=args.window, stride=args.stride, feature_spec=feature_spec
    )

    sample = dataset[0]
    print("num_clips:", bank.num_clips)
    print("num_frames:", bank.num_frames)
    print("inputs:", sample.inputs.shape)
    print("targets:", sample.targets.shape if sample.targets is not None else None)
    print("static betas:", sample.static["betas"].shape)


if __name__ == "__main__":
    main()
