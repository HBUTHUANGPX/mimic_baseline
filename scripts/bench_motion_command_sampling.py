from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

MAX_MULTINOMIAL_CATEGORIES = 2**24
DEFAULT_NUM_ENVS = (4096, 4096 * 2, 4096 * 4)
DEFAULT_NUM_MOTIONS = (100, 1000, 5000, 10000)
DEFAULT_MEAN_LENGTH_SECONDS = (3, 10, 300)


def device_from_arg(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def load_reference_artifact(path: str | Path) -> dict[str, int]:
    data = np.load(path)
    return {
        "fps": int(data["fps"].reshape(-1)[0]),
        "joint_dim": int(data["joint_pos"].shape[1]),
        "body_count": int(data["body_pos_w"].shape[1]),
    }


class Timer:
    def __init__(self, device: torch.device):
        self.device = device

    def stamp(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()


@dataclass
class BenchmarkResult:
    num_envs: int
    num_motions: int
    mean_length_s: int
    fps: int
    device: str
    status: str
    reason: str
    total_frames: int
    bin_count: int
    on_resample_start_ms: float
    build_sampling_probabilities_ms: float
    multinomial_ms: float
    sample_time_steps_ms: float
    update_env_motion_ids_ms: float
    on_resample_complete_ms: float
    update_sampling_metrics_ms: float
    total_ms: float


class MotionCommandSamplingBench:
    """Compact benchmark for MotionCommand's adaptive resampling hot path.

    This mirrors the runtime path in `_resample_time_steps()` for the default
    Q1 configuration: SONIC bin sampling, `history_frames=0`,
    `future_frames=4`, and a 1-second bin width derived from
    `decimation=1, sim.dt=0.02`, which matches the artifact FPS of 50.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        num_motions: int,
        mean_length_s: int,
        fps: int,
        future_frames: int,
        history_frames: int = 0,
        device: str | torch.device = "auto",
        seed: int = 0,
        sonic_mix_alpha: float = 0.1,
        sonic_failure_cap_beta: float = 200.0,
        terminated_prob: float = 0.2,
    ) -> None:
        self.device = (
            device_from_arg(device) if isinstance(device, str) else torch.device(device)
        )
        self.num_envs = int(num_envs)
        self.num_motions = int(num_motions)
        self.mean_length_s = int(mean_length_s)
        self.fps = int(fps)
        self.future_frames = int(future_frames)
        self.history_frames = int(history_frames)
        self.bin_frame_count = int(fps)
        self.sonic_mix_alpha = float(sonic_mix_alpha)
        self.sonic_failure_cap_beta = float(sonic_failure_cap_beta)
        self.terminated_prob = float(terminated_prob)

        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)

        self.frames_per_motion = int(round(self.mean_length_s * self.fps))
        min_required_frames = self.history_frames + self.future_frames + 1
        if self.frames_per_motion < min_required_frames:
            raise ValueError(
                f"Motion length {self.frames_per_motion} is shorter than the valid window "
                f"requirement {min_required_frames}."
            )

        self.motion_lengths = torch.full(
            (self.num_motions,),
            self.frames_per_motion,
            dtype=torch.long,
            device=self.device,
        )
        self.motion_ends = torch.cumsum(self.motion_lengths, dim=0)
        self.motion_starts = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=self.device),
                self.motion_ends[:-1],
            ],
            dim=0,
        )
        self.total_frames = int(self.motion_ends[-1].item())
        self.bin_count = int(
            (self.total_frames + self.bin_frame_count - 1) // self.bin_frame_count
        )
        if self.bin_count > MAX_MULTINOMIAL_CATEGORIES:
            raise ValueError(
                f"bin_count={self.bin_count} exceeds torch.multinomial limit "
                f"{MAX_MULTINOMIAL_CATEGORIES}"
            )

        self.motion_bin_ends = (
            (self.motion_ends + self.bin_frame_count - 1) // self.bin_frame_count
        ).contiguous()
        self.valid_sampling_bin_mask = torch.ones(
            self.bin_count, dtype=torch.bool, device=self.device
        )

        initial_motion_ids = torch.randint(
            0,
            self.num_motions,
            (self.num_envs,),
            dtype=torch.long,
            device=self.device,
        )
        local_valid_steps = torch.randint(
            self.history_frames,
            self.frames_per_motion - self.future_frames,
            (self.num_envs,),
            dtype=torch.long,
            device=self.device,
        )
        self.time_steps = self.motion_starts[initial_motion_ids] + local_valid_steps
        self.env_motion_ids = initial_motion_ids.clone()
        self.env_start_bin_ids = self.time_steps // self.bin_frame_count

        self.bin_visit_count = torch.zeros(
            self.bin_count, dtype=torch.float32, device=self.device
        )
        self.bin_fail_count = torch.zeros(
            self.bin_count, dtype=torch.float32, device=self.device
        )
        self.terminated = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.metrics = {
            "sampling_entropy": torch.zeros(1, dtype=torch.float32, device=self.device),
            "sampling_top1_prob": torch.zeros(1, dtype=torch.float32, device=self.device),
            "sampling_top1_bin": torch.zeros(1, dtype=torch.float32, device=self.device),
        }

    def sample_bins_for_test(self) -> torch.Tensor:
        return torch.arange(self.num_envs, device=self.device) % self.bin_count

    def motion_ids_from_time_steps(self, time_steps: torch.Tensor) -> torch.Tensor:
        return torch.bucketize(time_steps, self.motion_ends, right=True)

    def _motion_ids_from_bins(self, sampled_bins: torch.Tensor) -> torch.Tensor:
        return torch.bucketize(sampled_bins, self.motion_bin_ends, right=True)

    def sample_time_steps_from_bins(self, sampled_bins: torch.Tensor) -> torch.Tensor:
        candidate_time_steps = (
            sampled_bins * self.bin_frame_count
            + torch.randint(
                0,
                self.bin_frame_count,
                sampled_bins.shape,
                dtype=torch.long,
                device=self.device,
            )
        )
        candidate_time_steps = torch.clamp(
            candidate_time_steps, 0, self.total_frames - 1
        )

        motion_ids = self._motion_ids_from_bins(sampled_bins)
        bin_starts = sampled_bins * self.bin_frame_count
        bin_ends = torch.clamp(bin_starts + self.bin_frame_count, max=self.total_frames)
        valid_starts = torch.maximum(
            bin_starts, self.motion_starts[motion_ids] + self.history_frames
        )
        valid_ends = torch.minimum(
            bin_ends, self.motion_ends[motion_ids] - self.future_frames
        )
        if torch.any(valid_starts >= valid_ends):
            raise RuntimeError("Encountered a sampling bin without valid centers.")

        return torch.minimum(
            torch.maximum(candidate_time_steps, valid_starts), valid_ends - 1
        )

    def on_resample_start(
        self, env_ids: torch.Tensor, update_failure_statistics: bool = True
    ) -> None:
        if env_ids.numel() == 0 or not update_failure_statistics:
            return
        start_bin_ids = self.env_start_bin_ids[env_ids]
        self.bin_visit_count.index_add_(
            0, start_bin_ids, torch.ones_like(start_bin_ids, dtype=torch.float32)
        )
        failed = self.terminated[env_ids]
        if torch.any(failed):
            failed_start_bins = start_bin_ids[failed]
            self.bin_fail_count.index_add_(
                0,
                failed_start_bins,
                torch.ones_like(failed_start_bins, dtype=torch.float32),
            )

    def build_sampling_probabilities(self) -> torch.Tensor:
        valid_mask = self.valid_sampling_bin_mask
        valid_bin_count = max(int(valid_mask.sum().item()), 1)

        failure_rate = torch.zeros(
            self.bin_count, dtype=torch.float32, device=self.device
        )
        visited_mask = self.bin_visit_count > 0
        failure_rate[visited_mask] = (
            self.bin_fail_count[visited_mask] / self.bin_visit_count[visited_mask]
        )
        failure_rate = failure_rate * valid_mask.float()

        valid_failure_rates = failure_rate[valid_mask]
        mean_failure_rate = (
            valid_failure_rates.mean()
            if valid_failure_rates.numel() > 0
            else torch.tensor(0.0, dtype=torch.float32, device=self.device)
        )
        capped_failure_rate = torch.minimum(
            failure_rate, self.sonic_failure_cap_beta * mean_failure_rate
        )

        capped_sum = capped_failure_rate.sum()
        if capped_sum > 0:
            p_hat = capped_failure_rate / capped_sum
        else:
            p_hat = valid_mask.float() / float(valid_bin_count)

        uniform_distribution = valid_mask.float() / float(valid_bin_count)
        sampling_probabilities = (
            self.sonic_mix_alpha * p_hat
            + (1.0 - self.sonic_mix_alpha) * uniform_distribution
        )
        sampling_probabilities = sampling_probabilities * valid_mask.float()
        return sampling_probabilities / sampling_probabilities.sum()

    def on_resample_complete(self, env_ids: torch.Tensor, sampled_bins: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.env_start_bin_ids[env_ids] = sampled_bins

    def update_sampling_metrics(self, sampling_probabilities: torch.Tensor) -> None:
        entropy = -(
            sampling_probabilities * (sampling_probabilities + 1e-12).log()
        ).sum()
        normalized_entropy = entropy / max(math.log(self.bin_count), 1e-12)
        top1_prob, top1_index = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][0] = normalized_entropy
        self.metrics["sampling_top1_prob"][0] = top1_prob
        self.metrics["sampling_top1_bin"][0] = top1_index.float() / max(
            self.bin_count, 1
        )

    def benchmark_iteration(self, timer: Timer) -> dict[str, float]:
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.terminated = (
            torch.rand(self.num_envs, device=self.device) < self.terminated_prob
        )

        t0 = timer.stamp()
        self.on_resample_start(env_ids)
        t1 = timer.stamp()

        sampling_probabilities = self.build_sampling_probabilities()
        t2 = timer.stamp()

        sampled_bins = torch.multinomial(
            sampling_probabilities, self.num_envs, replacement=True
        )
        t3 = timer.stamp()

        sampled_time_steps = self.sample_time_steps_from_bins(sampled_bins)
        self.time_steps[env_ids] = sampled_time_steps
        t4 = timer.stamp()

        self.env_motion_ids[env_ids] = self.motion_ids_from_time_steps(sampled_time_steps)
        t5 = timer.stamp()

        self.on_resample_complete(env_ids, sampled_bins)
        t6 = timer.stamp()

        self.update_sampling_metrics(sampling_probabilities)
        t7 = timer.stamp()

        return {
            "on_resample_start_ms": (t1 - t0) * 1000.0,
            "build_sampling_probabilities_ms": (t2 - t1) * 1000.0,
            "multinomial_ms": (t3 - t2) * 1000.0,
            "sample_time_steps_ms": (t4 - t3) * 1000.0,
            "update_env_motion_ids_ms": (t5 - t4) * 1000.0,
            "on_resample_complete_ms": (t6 - t5) * 1000.0,
            "update_sampling_metrics_ms": (t7 - t6) * 1000.0,
            "total_ms": (t7 - t0) * 1000.0,
        }


def mean_timings(samples: list[dict[str, float]]) -> dict[str, float]:
    keys = samples[0].keys()
    return {key: sum(sample[key] for sample in samples) / len(samples) for key in keys}


def run_single_benchmark(
    *,
    num_envs: int,
    num_motions: int,
    mean_length_s: int,
    fps: int,
    future_frames: int,
    device: torch.device,
    warmup: int,
    iters: int,
    seed: int,
) -> BenchmarkResult:
    try:
        bench = MotionCommandSamplingBench(
            num_envs=num_envs,
            num_motions=num_motions,
            mean_length_s=mean_length_s,
            fps=fps,
            future_frames=future_frames,
            device=device,
            seed=seed,
        )
    except ValueError as exc:
        return BenchmarkResult(
            num_envs=num_envs,
            num_motions=num_motions,
            mean_length_s=mean_length_s,
            fps=fps,
            device=str(device),
            status="unsupported",
            reason=str(exc),
            total_frames=num_motions * mean_length_s * fps,
            bin_count=math.ceil((num_motions * mean_length_s * fps) / fps),
            on_resample_start_ms=float("nan"),
            build_sampling_probabilities_ms=float("nan"),
            multinomial_ms=float("nan"),
            sample_time_steps_ms=float("nan"),
            update_env_motion_ids_ms=float("nan"),
            on_resample_complete_ms=float("nan"),
            update_sampling_metrics_ms=float("nan"),
            total_ms=float("nan"),
        )

    timer = Timer(bench.device)
    for _ in range(warmup):
        bench.benchmark_iteration(timer)

    samples = [bench.benchmark_iteration(timer) for _ in range(iters)]
    averages = mean_timings(samples)
    return BenchmarkResult(
        num_envs=num_envs,
        num_motions=num_motions,
        mean_length_s=mean_length_s,
        fps=fps,
        device=str(device),
        status="ok",
        reason="",
        total_frames=bench.total_frames,
        bin_count=bench.bin_count,
        on_resample_start_ms=averages["on_resample_start_ms"],
        build_sampling_probabilities_ms=averages["build_sampling_probabilities_ms"],
        multinomial_ms=averages["multinomial_ms"],
        sample_time_steps_ms=averages["sample_time_steps_ms"],
        update_env_motion_ids_ms=averages["update_env_motion_ids_ms"],
        on_resample_complete_ms=averages["on_resample_complete_ms"],
        update_sampling_metrics_ms=averages["update_sampling_metrics_ms"],
        total_ms=averages["total_ms"],
    )


def write_csv(path: Path, results: list[BenchmarkResult]) -> None:
    fieldnames = list(BenchmarkResult.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def write_markdown(
    path: Path,
    results: list[BenchmarkResult],
    reference_artifact: Path,
    artifact_info: dict[str, int],
) -> None:
    ok_results = [result for result in results if result.status == "ok"]
    unsupported_results = [result for result in results if result.status != "ok"]
    slowest = sorted(ok_results, key=lambda item: item.total_ms, reverse=True)[:5]

    lines = [
        "# MotionCommand Sampling Benchmark",
        "",
        f"- Reference artifact: `{reference_artifact}`",
        f"- Artifact fps: `{artifact_info['fps']}`",
        f"- Artifact joint dim: `{artifact_info['joint_dim']}`",
        f"- Artifact body count: `{artifact_info['body_count']}`",
        f"- Sweep num_envs: `{list(DEFAULT_NUM_ENVS)}`",
        f"- Sweep num_motions: `{list(DEFAULT_NUM_MOTIONS)}`",
        f"- Sweep mean_length_s: `{list(DEFAULT_MEAN_LENGTH_SECONDS)}`",
        "",
        "## Slowest Supported Configurations",
        "",
    ]

    if slowest:
        lines.append(
            "| num_envs | num_motions | mean_length_s | bin_count | build_probs_ms | multinomial_ms | total_ms |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for item in slowest:
            lines.append(
                f"| {item.num_envs} | {item.num_motions} | {item.mean_length_s} | "
                f"{item.bin_count} | {item.build_sampling_probabilities_ms:.3f} | "
                f"{item.multinomial_ms:.3f} | {item.total_ms:.3f} |"
            )
    else:
        lines.append("No supported configurations completed.")

    lines.extend(["", "## Unsupported Configurations", ""])
    if unsupported_results:
        lines.append("| num_envs | num_motions | mean_length_s | reason |")
        lines.append("| --- | --- | --- | --- |")
        for item in unsupported_results:
            lines.append(
                f"| {item.num_envs} | {item.num_motions} | {item.mean_length_s} | {item.reason} |"
            )
    else:
        lines.append("None.")

    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MotionCommand sampling.")
    parser.add_argument(
        "--reference-artifact",
        type=Path,
        default=Path("artifacts/Q1/100STYLE/Aeroplane/Aeroplane_BR.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bench_motion_command_sampling"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--future-frames", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_info = load_reference_artifact(args.reference_artifact)
    device = device_from_arg(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[BenchmarkResult] = []
    for num_envs in DEFAULT_NUM_ENVS:
        for num_motions in DEFAULT_NUM_MOTIONS:
            for mean_length_s in DEFAULT_MEAN_LENGTH_SECONDS:
                print(
                    f"[bench] envs={num_envs} motions={num_motions} "
                    f"mean_length_s={mean_length_s}"
                )
                result = run_single_benchmark(
                    num_envs=num_envs,
                    num_motions=num_motions,
                    mean_length_s=mean_length_s,
                    fps=artifact_info["fps"],
                    future_frames=args.future_frames,
                    device=device,
                    warmup=args.warmup,
                    iters=args.iters,
                    seed=args.seed,
                )
                results.append(result)

    csv_path = args.output_dir / "motion_command_sampling_benchmark.csv"
    md_path = args.output_dir / "motion_command_sampling_benchmark.md"
    write_csv(csv_path, results)
    write_markdown(md_path, results, args.reference_artifact, artifact_info)
    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {md_path}")


if __name__ == "__main__":
    main()
