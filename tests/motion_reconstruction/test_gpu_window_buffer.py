import torch

from motion_reconstruction.data.gpu_buffer import MotionWindowBuffer


def test_window_buffer_keeps_full_valid_pool_and_final_small_batch():
    robot = torch.arange(10, dtype=torch.float32).view(10, 1)
    human = torch.arange(100, 110, dtype=torch.float32).view(10, 1)
    buffer = MotionWindowBuffer(
        robot_features=robot,
        human_features=human,
        motion_lengths=torch.tensor([5, 5]),
        history=1,
        future=1,
        device="cpu",
    )

    assert buffer.valid_center_indices.tolist() == [1, 2, 3, 6, 7, 8]
    seen = []
    for batch in buffer.iter_epoch_batches(batch_size=4, generator=torch.Generator().manual_seed(0)):
        assert batch.robot_window.shape[1:] == (3, 1)
        assert batch.human_window.shape[1:] == (3, 1)
        seen.extend(batch.center_indices.tolist())
        for window_indices in batch.window_indices.tolist():
            assert window_indices in (
                [0, 1, 2],
                [1, 2, 3],
                [2, 3, 4],
                [5, 6, 7],
                [6, 7, 8],
                [7, 8, 9],
            )

    assert sorted(seen) == [1, 2, 3, 6, 7, 8]
