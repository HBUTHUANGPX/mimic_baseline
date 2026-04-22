import torch

from motion_reconstruction.config.schema import ModelConfig, MotionReconstructionConfig, QuantizerConfig
from motion_reconstruction.pipeline import build_autoencoder


def test_build_autoencoder_returns_model_and_stable_quantizer_config():
    config = MotionReconstructionConfig(
        model=ModelConfig(
            latent_dim=3,
            robot_encoder_hidden_dims=[8],
            human_encoder_hidden_dims=[8],
            decoder_hidden_dims=[8],
            quantizer=QuantizerConfig(type="fsq", levels=5),
        )
    )

    model, quantizer_config = build_autoencoder(config, robot_input_dim=6, human_input_dim=9)
    output = model(torch.zeros(2, 6), torch.zeros(2, 9))

    assert quantizer_config["levels"] == [5, 5, 5]
    assert output.recon_from_robot.shape == (2, 6)
    assert output.recon_from_human.shape == (2, 6)
