import torch

from motion_reconstruction.models.components import DualFSQAutoEncoder
from motion_reconstruction.models.quantizers import FSQQuantizer, IFSQQuantizer
from motion_reconstruction.training.losses import DualReconstructionLoss


def test_fsq_quantizer_returns_level_indices_and_uses_ste():
    z = torch.tensor([[-0.2, 0.0, 0.8]], requires_grad=True)
    quantizer = FSQQuantizer(levels=[5, 5, 5])

    out = quantizer(z)
    out.z_q.sum().backward()

    assert out.z_q.shape == z.shape
    assert out.level_indices.shape == z.shape
    assert out.level_indices.min() >= 0
    assert out.level_indices.max() < 5
    assert z.grad is not None
    assert torch.all(torch.isfinite(z.grad))
    assert torch.all(z.grad.abs() > 0)


def test_ifsq_quantizer_uses_simple_bound_defaults():
    z = torch.linspace(-10, 10, steps=4).view(1, 4)
    quantizer = IFSQQuantizer(levels=17)

    out = quantizer(z)

    assert out.z_q.shape == z.shape
    assert out.level_indices.shape == z.shape
    assert torch.all(out.z_q <= 1.0)
    assert torch.all(out.z_q >= -1.0)


def test_dual_autoencoder_shares_quantizer_and_backpropagates_losses():
    model = DualFSQAutoEncoder(
        robot_input_dim=15,
        human_input_dim=18,
        latent_dim=4,
        robot_encoder_hidden_dims=[16],
        human_encoder_hidden_dims=[16],
        decoder_hidden_dims=[16],
        quantizer=FSQQuantizer(levels=[5, 5, 5, 5]),
    )
    robot_window = torch.randn(3, 15)
    human_window = torch.randn(3, 18)

    out = model(robot_window, human_window)
    loss_out = DualReconstructionLoss()(out, robot_window)
    loss_out.total.backward()

    assert model.shared_quantizer is model.shared_quantizer
    assert out.recon_from_robot.shape == robot_window.shape
    assert out.recon_from_human.shape == robot_window.shape
    assert out.q_robot.shape == (3, 4)
    assert out.q_human.shape == (3, 4)
    assert out.q_cycle.shape == (3, 4)
    assert set(loss_out.terms) == {
        "robot_recon",
        "human_recon",
        "latent_align",
        "cycle_latent",
    }
    assert model.robot_encoder.net[0].weight.grad is not None
    assert model.human_encoder.net[0].weight.grad is not None
