"""窗口数据集的单元测试。"""

import torch

from mocap_motion_vae.data.bank import ClipData, FeatureSpec, MotionBank
from mocap_motion_vae.data.dataset import MotionWindowDataset


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


def test_window_dataset_shapes():
    """测试窗口采样的输出形状。"""
    clips = [_make_clip("c0", 8), _make_clip("c1", 10)]
    bank = MotionBank.from_clips(clips)
    spec = FeatureSpec(
        inputs=("pose_body", "root_orient"),
        targets=("pose_body",),
        static=("betas",),
    )

    dataset = MotionWindowDataset(bank, window=4, stride=2, feature_spec=spec)
    assert len(dataset) > 0

    sample = dataset[0]
    assert isinstance(sample.inputs, torch.Tensor)
    assert sample.inputs.shape == (4, 66)
    assert sample.targets is not None
    assert sample.targets.shape == (4, 63)
    assert sample.static["betas"].shape == (4, 16)
