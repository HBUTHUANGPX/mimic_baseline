from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bench_motion_command_sampling import (
    MAX_MULTINOMIAL_CATEGORIES,
    MotionCommandSamplingBench,
)


def test_sampled_time_steps_stay_within_valid_range():
    bench = MotionCommandSamplingBench(
        num_envs=32,
        num_motions=4,
        mean_length_s=3,
        fps=50,
        future_frames=4,
        device="cpu",
        seed=7,
    )

    sampled_bins = bench.sample_bins_for_test()
    sampled_time_steps = bench.sample_time_steps_from_bins(sampled_bins)
    motion_ids = bench.motion_ids_from_time_steps(sampled_time_steps)

    valid_starts = bench.motion_starts[motion_ids] + bench.history_frames
    valid_last = bench.motion_ends[motion_ids] - bench.future_frames - 1

    assert sampled_time_steps.shape == sampled_bins.shape
    assert (sampled_time_steps >= valid_starts).all()
    assert (sampled_time_steps <= valid_last).all()


def test_rejects_bin_count_above_multinomial_limit():
    with pytest.raises(ValueError, match="multinomial"):
        MotionCommandSamplingBench(
            num_envs=16,
            num_motions=10000,
            mean_length_s=2000,
            fps=50,
            future_frames=4,
            device="cpu",
        )

    assert MAX_MULTINOMIAL_CATEGORIES == 2**24
