from pathlib import Path

import torch

from motion_reconstruction.models.components import DualFSQAutoEncoder
from motion_reconstruction.models.quantizers import FSQQuantizer
from motion_reconstruction.training.checkpoint import save_checkpoint
from motion_reconstruction.training.normalization import WindowFeatureNormalizer


def test_window_normalizer_fits_frame_stats_and_repeats_for_windows():
    frames = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    normalizer = WindowFeatureNormalizer.from_frame_features(frames, window_size=3)

    assert normalizer.mean.shape == (6,)
    assert torch.allclose(normalizer.mean, torch.tensor([3.0, 6.0] * 3))
    window = torch.tensor([[1.0, 2.0, 3.0, 6.0, 5.0, 10.0]])
    restored = normalizer.inverse(normalizer(window))
    assert torch.allclose(restored, window, atol=1e-5)


def test_checkpoint_contains_model_optimizer_normalizers_and_schema(tmp_path: Path):
    model = DualFSQAutoEncoder(
        robot_input_dim=6,
        human_input_dim=6,
        latent_dim=2,
        robot_encoder_hidden_dims=[8],
        human_encoder_hidden_dims=[8],
        decoder_hidden_dims=[8],
        quantizer=FSQQuantizer(levels=[5, 5]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    normalizers = {
        "robot": WindowFeatureNormalizer(torch.zeros(6), torch.ones(6)),
        "human": WindowFeatureNormalizer(torch.zeros(6), torch.ones(6)),
    }

    path = save_checkpoint(
        output_dir=tmp_path,
        name="latest.pt",
        model=model,
        optimizer=optimizer,
        epoch=2,
        global_step=11,
        config={"train": {"epochs": 3}},
        normalizers=normalizers,
        feature_schema={"robot_joint_names": ["hip"]},
        quantizer_config={"type": "fsq", "levels": [5, 5]},
    )

    payload = torch.load(path, map_location="cpu")
    assert payload["epoch"] == 2
    assert payload["global_step"] == 11
    assert "model" in payload
    assert "optimizer" in payload
    assert payload["normalizers"]["robot"]["mean"].shape == (6,)
    assert payload["feature_schema"]["robot_joint_names"] == ["hip"]
