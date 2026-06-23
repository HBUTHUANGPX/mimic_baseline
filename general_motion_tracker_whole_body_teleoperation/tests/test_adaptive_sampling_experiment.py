from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch

_EXPERIMENT_PATH = (
    Path(__file__).resolve().parents[1]
    / "general_motion_tracker_whole_body_teleoperation"
    / "utils"
    / "adaptive_sampling_experiment.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "adaptive_sampling_experiment_under_test",
    _EXPERIMENT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
adaptive_sampling_experiment = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault(
    "adaptive_sampling_experiment_under_test",
    adaptive_sampling_experiment,
)
_SPEC.loader.exec_module(adaptive_sampling_experiment)

DifficultyDataset = adaptive_sampling_experiment.DifficultyDataset
DifficultyDatasetConfig = adaptive_sampling_experiment.DifficultyDatasetConfig
LearningState = adaptive_sampling_experiment.LearningState
SimulationConfig = adaptive_sampling_experiment.SimulationConfig
TrainingSimulator = adaptive_sampling_experiment.TrainingSimulator
UniformSamplerAdapter = adaptive_sampling_experiment.UniformSamplerAdapter
LegacySamplerAdapter = adaptive_sampling_experiment.LegacySamplerAdapter
SonicSamplerAdapter = adaptive_sampling_experiment.SonicSamplerAdapter
StratifiedLegacySamplerAdapter = (
    adaptive_sampling_experiment.StratifiedLegacySamplerAdapter
)


def test_difficulty_dataset_matches_quadratic_ratios_and_window_constraint():
    config = DifficultyDatasetConfig(total_frames=3850, max_window_span=2)

    dataset = DifficultyDataset.generate(config)

    counts = torch.bincount(dataset.difficulty, minlength=11)[1:]
    expected_counts = torch.tensor([(11 - d) ** 2 * 10 for d in range(1, 11)])
    assert torch.equal(counts.cpu(), expected_counts)

    windows = dataset.difficulty.unfold(0, 10, 1).to(torch.int16)
    window_spans = windows.max(dim=1).values - windows.min(dim=1).values
    assert int(window_spans.max().item()) <= 2


def test_learning_state_requires_each_frame_to_reach_its_own_threshold():
    dataset = DifficultyDataset(
        difficulty=torch.tensor([1, 2, 10], dtype=torch.uint8),
    )
    state = LearningState(dataset)

    frame_ids = torch.tensor([0] * 20 + [1] * 79 + [2] * 1999)
    failures = state.record_samples(frame_ids)

    assert failures.all()
    assert state.learned_mask.tolist() == [True, False, False]

    state.record_samples(torch.tensor([1, 2]))

    assert state.learned_mask.tolist() == [True, True, True]


def test_uniform_simulator_learns_all_tiny_frames_with_independent_counts():
    dataset = DifficultyDataset(
        difficulty=torch.tensor([1, 1], dtype=torch.uint8),
    )
    sampler = UniformSamplerAdapter(num_envs=2, dataset=dataset, seed=7)
    simulator = TrainingSimulator(
        dataset=dataset,
        sampler=sampler,
        config=SimulationConfig(num_envs=2, max_iterations=60),
    )

    result = simulator.run()

    assert result.learned_frames == 2
    assert result.total_frames == 2
    assert result.iterations <= 60


def test_sonic_sampler_adapter_runs_one_iteration_against_real_sampler_logic():
    dataset = DifficultyDataset(
        difficulty=torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.uint8),
    )
    sampler = SonicSamplerAdapter(
        num_envs=4,
        dataset=dataset,
        bin_frame_count=2,
        seed=3,
        device="cpu",
    )
    simulator = TrainingSimulator(
        dataset=dataset,
        sampler=sampler,
        config=SimulationConfig(num_envs=4, max_iterations=1),
    )

    result = simulator.run()

    assert result.iterations == 1
    assert result.samples == 4


def test_legacy_sampler_keeps_original_uniform_offset_behavior():
    dataset = DifficultyDataset(
        difficulty=torch.ones(4, dtype=torch.uint8),
    )
    sampler = LegacySamplerAdapter(
        num_envs=1,
        dataset=dataset,
        bin_frame_count=1,
        seed=3,
        device="cpu",
    )
    legacy_sampler = sampler.sampler
    legacy_sampler.bin_failed_count[:] = torch.tensor([0.0, 1.0, 0.0, 0.0])

    probabilities = legacy_sampler.build_sampling_probabilities()

    expected_probabilities = torch.tensor(
        [0.025 / 1.1, 1.025 / 1.1, 0.025 / 1.1, 0.025 / 1.1]
    )
    assert torch.allclose(probabilities.cpu(), expected_probabilities, atol=1e-6)


def test_stratified_legacy_sampler_uses_fixed_failure_uniform_mixture():
    dataset = DifficultyDataset(
        difficulty=torch.ones(4, dtype=torch.uint8),
    )
    sampler = StratifiedLegacySamplerAdapter(
        num_envs=1,
        dataset=dataset,
        bin_frame_count=1,
        seed=3,
        device="cpu",
    )
    legacy_sampler = sampler.sampler
    legacy_sampler.bin_failed_count[:] = torch.tensor([0.0, 1.0, 0.0, 0.0])

    probabilities = legacy_sampler.build_sampling_probabilities()

    expected_probabilities = torch.tensor([0.05, 0.85, 0.05, 0.05])
    assert torch.allclose(probabilities.cpu(), expected_probabilities, atol=1e-6)


def test_legacy_sampler_runs_on_cuda_without_cudnn_initialization_failure():
    if not torch.cuda.is_available():
        return

    dataset = DifficultyDataset.generate(
        DifficultyDatasetConfig(total_frames=30_000_000, device="cuda")
    )
    sampler = LegacySamplerAdapter(
        num_envs=4096,
        dataset=dataset,
        bin_frame_count=50,
        seed=3,
        device="cuda",
    )
    simulator = TrainingSimulator(
        dataset=dataset,
        sampler=sampler,
        config=SimulationConfig(num_envs=4096, max_iterations=1),
    )

    result = simulator.run()

    assert result.iterations == 1
    assert result.samples == 4096
