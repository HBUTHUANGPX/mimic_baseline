"""SMPL-X 解析与构建 MotionBank 的单元测试。"""

import os

import numpy as np
import pytest

from mocap_motion_vae.data.amass_smplx import (
    SMPLXClipParser,
    SMPLXFieldSpec,
    build_amass_smplx_bank,
    discover_amass_smplx_files,
)


def _write_npz(path, **kwargs):
    """写入一个临时 npz 文件。

    Args:
        path: 目标路径。
        **kwargs: 写入字段。
    """
    np.savez(path, **kwargs)


def test_parser_basic_fields(tmp_path):
    """测试基础字段解析（pose_body/root_orient/trans）。"""
    if os.getenv("SMPLX_MODEL_PATH") is None:
        pytest.skip("SMPLX_MODEL_PATH 未设置，跳过 SMPL-X 解析测试。")
    path = tmp_path / "clip_basic.npz"
    _write_npz(
        path,
        fps=np.array(60),
        pose_body=np.zeros((10, 63), dtype=np.float32),
        root_orient=np.zeros((10, 3), dtype=np.float32),
        trans=np.zeros((10, 3), dtype=np.float32),
        betas=np.zeros((16,), dtype=np.float32),
        gender=np.array("male"),
    )

    parser = SMPLXClipParser()
    clip = parser.parse(str(path))

    assert clip.fps == 60.0
    assert clip.pose_body.shape == (10, 63)
    assert clip.root_orient.shape == (10, 3)
    assert clip.trans.shape == (10, 3)
    assert clip.betas.shape[0] == 16
    assert clip.gender == "male"


def test_parser_fallback_fields(tmp_path):
    """测试 poses/transl 等兼容字段解析。"""
    if os.getenv("SMPLX_MODEL_PATH") is None:
        pytest.skip("SMPLX_MODEL_PATH 未设置，跳过 SMPL-X 解析测试。")
    path = tmp_path / "clip_fallback.npz"
    poses = np.zeros((8, 66), dtype=np.float32)
    transl = np.zeros((8, 3), dtype=np.float32)
    _write_npz(
        path,
        mocap_framerate=np.array(30),
        poses=poses,
        transl=transl,
        betas=np.zeros((10,), dtype=np.float32),
        gender=np.array(b"female"),
    )

    parser = SMPLXClipParser()
    clip = parser.parse(str(path))

    assert clip.fps == 30.0
    assert clip.pose_body.shape == (8, 63)
    assert clip.root_orient.shape == (8, 3)
    assert clip.trans.shape == (8, 3)
    assert clip.gender == "female"


def test_build_bank_and_discover(tmp_path):
    """测试文件发现与 MotionBank 构建。"""
    if os.getenv("SMPLX_MODEL_PATH") is None:
        pytest.skip("SMPLX_MODEL_PATH 未设置，跳过 SMPL-X 解析测试。")
    root = tmp_path / "amass"
    root.mkdir()
    (root / "A").mkdir()
    (root / "B").mkdir()

    for i in range(3):
        _write_npz(
            root / "A" / f"clip_{i}.npz",
            fps=np.array(120),
            pose_body=np.zeros((5, 63), dtype=np.float32),
            root_orient=np.zeros((5, 3), dtype=np.float32),
            trans=np.zeros((5, 3), dtype=np.float32),
            betas=np.zeros((16,), dtype=np.float32),
            gender=np.array("neutral"),
        )
    _write_npz(
        root / "B" / "clip_3.npz",
        fps=np.array(120),
        pose_body=np.zeros((7, 63), dtype=np.float32),
        root_orient=np.zeros((7, 3), dtype=np.float32),
        trans=np.zeros((7, 3), dtype=np.float32),
        betas=np.zeros((16,), dtype=np.float32),
        gender=np.array("neutral"),
    )

    files = discover_amass_smplx_files(root)
    assert len(files) == 4

    spec = SMPLXFieldSpec()
    bank = build_amass_smplx_bank(files, spec)
    assert bank.num_clips == 4
    assert bank.num_frames == 5 * 3 + 7
