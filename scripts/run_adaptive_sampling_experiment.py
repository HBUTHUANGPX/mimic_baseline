from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import time

import torch


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_experiment_module():
    module_path = (
        _REPO_ROOT
        / "general_motion_tracker_whole_body_teleoperation"
        / "general_motion_tracker_whole_body_teleoperation"
        / "utils"
        / "adaptive_sampling_experiment.py"
    )
    spec = importlib.util.spec_from_file_location(
        "adaptive_sampling_experiment_cli",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("adaptive_sampling_experiment_cli", module)
    spec.loader.exec_module(module)
    return module


_EXPERIMENT_MODULE = _load_experiment_module()
DEFAULT_FPS = _EXPERIMENT_MODULE.DEFAULT_FPS
DEFAULT_HOURS = _EXPERIMENT_MODULE.DEFAULT_HOURS
DEFAULT_NUM_ENVS = _EXPERIMENT_MODULE.DEFAULT_NUM_ENVS
DifficultyDataset = _EXPERIMENT_MODULE.DifficultyDataset
DifficultyDatasetConfig = _EXPERIMENT_MODULE.DifficultyDatasetConfig
LegacySamplerAdapter = _EXPERIMENT_MODULE.LegacySamplerAdapter
SimulationConfig = _EXPERIMENT_MODULE.SimulationConfig
SonicSamplerAdapter = _EXPERIMENT_MODULE.SonicSamplerAdapter
StratifiedLegacySamplerAdapter = _EXPERIMENT_MODULE.StratifiedLegacySamplerAdapter
TrainingSimulator = _EXPERIMENT_MODULE.TrainingSimulator
UniformSamplerAdapter = _EXPERIMENT_MODULE.UniformSamplerAdapter


def main() -> None:
    args = _parse_args()
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    total_frames = args.total_frames or args.fps * 60 * 60 * args.hours
    dataset = DifficultyDataset.generate(
        DifficultyDatasetConfig(
            total_frames=total_frames,
            max_window_span=args.max_window_span,
            device=args.dataset_device,
        )
    )
    simulation_config = SimulationConfig(
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
        target_learned_ratio=args.target_learned_ratio,
    )

    for sampler_name in args.samplers:
        sampler = _build_sampler(
            sampler_name=sampler_name,
            dataset=dataset,
            num_envs=args.num_envs,
            bin_frame_count=args.bin_frame_count,
            seed=args.seed,
            device=args.sampler_device,
        )
        start_time = time.perf_counter()
        result = TrainingSimulator(
            dataset=dataset,
            sampler=sampler,
            config=simulation_config,
        ).run()
        elapsed = time.perf_counter() - start_time
        _print_result(result, elapsed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate per-frame learning efficiency for adaptive motion samplers."
        )
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--hours", type=int, default=DEFAULT_HOURS)
    parser.add_argument(
        "--total-frames",
        type=int,
        default=None,
        help="Overrides --fps * 60 * 60 * --hours when provided.",
    )
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--max-iterations", type=int, default=1_000)
    parser.add_argument("--target-learned-ratio", type=float, default=1.0)
    parser.add_argument("--max-window-span", type=int, default=2)
    parser.add_argument("--bin-frame-count", type=int, default=DEFAULT_FPS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset-device", default="cpu")
    parser.add_argument("--sampler-device", default="cpu")
    parser.add_argument(
        "--disable-cudnn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable cuDNN for CUDA runs; this avoids conv1d init issues in legacy.",
    )
    parser.add_argument(
        "--samplers",
        nargs="+",
        choices=("uniform", "legacy", "stratified_legacy", "sonic"),
        default=("uniform", "legacy", "sonic"),
    )
    return parser.parse_args()


def _build_sampler(
    sampler_name: str,
    dataset: DifficultyDataset,
    num_envs: int,
    bin_frame_count: int,
    seed: int,
    device: str,
):
    if sampler_name == "uniform":
        return UniformSamplerAdapter(num_envs=num_envs, dataset=dataset, seed=seed)
    if sampler_name == "legacy":
        return LegacySamplerAdapter(
            num_envs=num_envs,
            dataset=dataset,
            bin_frame_count=bin_frame_count,
            seed=seed,
            device=device,
        )
    if sampler_name == "stratified_legacy":
        return StratifiedLegacySamplerAdapter(
            num_envs=num_envs,
            dataset=dataset,
            bin_frame_count=bin_frame_count,
            seed=seed,
            device=device,
        )
    if sampler_name == "sonic":
        return SonicSamplerAdapter(
            num_envs=num_envs,
            dataset=dataset,
            bin_frame_count=bin_frame_count,
            seed=seed,
            device=device,
        )
    raise ValueError(f"Unknown sampler: {sampler_name}")


def _print_result(result, elapsed: float) -> None:
    print(f"\n[{result.sampler_name}]")
    print(f"iterations: {result.iterations}")
    print(f"samples: {result.samples}")
    print(f"learned_frames: {result.learned_frames}/{result.total_frames}")
    print(f"learned_ratio: {result.learned_ratio:.6f}")
    print(f"mean_samples_per_frame: {result.mean_samples_per_frame:.3f}")
    print(f"elapsed_seconds: {elapsed:.2f}")
    print("learned_by_difficulty:")
    for difficulty in range(1, 11):
        learned = result.learned_frames_by_difficulty[difficulty]
        total = result.total_frames_by_difficulty[difficulty]
        print(f"  d={difficulty}: {learned}/{total}")


if __name__ == "__main__":
    main()
