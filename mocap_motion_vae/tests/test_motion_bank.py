"""MotionBank 与 MotionView 的单元测试。"""

import torch

from mocap_motion_vae.data.bank import ClipData, MotionBank, MotionView


def _make_clip(name: str, length: int) -> ClipData:
    """构造一个测试用的 ClipData。

    Args:
        name: 片段名称。
        length: 帧数。

    Returns:
        ClipData 实例。
    """
    frames = {
        "pose_body": torch.zeros((length, 63)),
        "root_orient": torch.zeros((length, 3)),
        "trans": torch.zeros((length, 3)),
    }
    static = {
        "betas": torch.zeros((16,)),
    }
    return ClipData(name=name, fps=60.0, frames=frames, static=static, meta={})


def test_motion_bank_indices():
    """测试 clip 索引与新片段标记。"""
    clips = [_make_clip("c0", 4), _make_clip("c1", 6)]
    bank = MotionBank.from_clips(clips)

    assert bank.num_clips == 2
    assert bank.num_frames == 10
    assert bank.clip_indices.tolist() == [[0, 4], [4, 10]]
    assert bank.new_clip_flag.sum().item() == 1
    assert bank.new_clip_flag[4].item() is True


def test_motion_view_indexing():
    """测试 MotionView 的索引与拼接逻辑。"""
    clips = [_make_clip("c0", 5)]
    bank = MotionBank.from_clips(clips)
    time_steps = torch.tensor([0, 2, 4], dtype=torch.long)
    view = MotionView(bank, time_steps)

    pose = view.field("pose_body")
    assert pose.shape == (3, 63)

    betas = view.static("betas")
    assert betas.shape == (3, 16)

    concat = view.concat("pose_body", "root_orient")
    assert concat.shape == (3, 66)
