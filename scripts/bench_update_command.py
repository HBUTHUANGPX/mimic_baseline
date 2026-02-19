import argparse
import time
from dataclasses import dataclass

import torch


def _device_from_arg(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


@dataclass
class TerminationManager:
    terminated: torch.Tensor
    time_outs: torch.Tensor


class FakeEnv:
    def __init__(self, num_envs: int, device: torch.device):
        self.termination_manager = TerminationManager(
            terminated=torch.zeros(num_envs, dtype=torch.bool, device=device),
            time_outs=torch.zeros(num_envs, dtype=torch.bool, device=device),
        )


class FakeMotion:
    def __init__(
        self,
        num_motions: int,
        frames_per_motion: int,
        length_jitter: float,
        device: torch.device,
    ):
        self.num_motions = num_motions
        # Randomize motion lengths around frames_per_motion.
        # Keep lengths >= 1 to avoid empty segments.
        if length_jitter <= 0.0:
            lengths = torch.full((num_motions,), frames_per_motion, device=device, dtype=torch.long)
        else:
            low = max(1, int(frames_per_motion * (1.0 - length_jitter)))
            high = max(low + 1, int(frames_per_motion * (1.0 + length_jitter)) + 1)
            lengths = torch.randint(low=low, high=high, size=(num_motions,), device=device)

        self.motion_lengths = lengths
        self.time_step_total = int(lengths.sum().item())

        # motion_indices: [start, end)
        ends = torch.cumsum(lengths, dim=0)
        starts = torch.cat([torch.zeros(1, device=device, dtype=torch.long), ends[:-1]], dim=0)
        self.motion_indices = torch.stack([starts, ends], dim=1)

        # new_data_flag: True at the first frame of each motion except the first
        self.new_data_flag = torch.zeros(self.time_step_total, dtype=torch.bool, device=device)
        self.new_data_flag[starts[1:]] = True

        self.extracted_list = [f"m{i:03d}" for i in range(num_motions)]

        self.motion_distribution = torch.full(
            (1, num_motions), 1.0 / num_motions, dtype=torch.float32, device=device
        )


class BenchModule:
    def __init__(
        self,
        num_envs: int,
        num_motions: int,
        frames_per_motion: int,
        length_jitter: float,
        bin_size: int,
        device: torch.device,
        terminated_prob: float,
        timeout_ratio: float,
        seed: int,
    ):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        self.device = device
        self.num_envs = num_envs
        self.motion = FakeMotion(num_motions, frames_per_motion, length_jitter, device)
        self._env = FakeEnv(num_envs, device)

        self.time_steps = torch.randint(
            low=0,
            high=self.motion.time_step_total,
            size=(num_envs,),
            device=device,
            dtype=torch.long,
        )

        self._bin_size = max(1, int(bin_size))
        self._bin_count = int((self.motion.time_step_total + self._bin_size - 1) // self._bin_size)
        self._bin_failed = torch.zeros(self._bin_count, dtype=torch.float32, device=device)
        self._current_bin_failed = torch.zeros(self._bin_count, dtype=torch.float32, device=device)

        self.counts = torch.zeros(num_motions, dtype=torch.float32, device=device)
        self.metrics = {name: torch.zeros(num_envs, dtype=torch.float32, device=device)
                        for name in self.motion.extracted_list}

        self.terminated_prob = float(terminated_prob)
        self.timeout_ratio = float(timeout_ratio)
        # Cache for vectorized mapping
        self._motion_ends = self.motion.motion_indices[:, 1].contiguous()

    def _update_termination(self):
        # Random termination mask
        terminated = torch.rand(self.num_envs, device=self.device) < self.terminated_prob
        time_outs = terminated & (torch.rand(self.num_envs, device=self.device) < self.timeout_ratio)
        self._env.termination_manager.terminated = terminated
        self._env.termination_manager.time_outs = time_outs

    def _distribution_loop(self):
        for i in range(self.motion.num_motions):
            start, end = self.motion.motion_indices[i]
            mask = (self.time_steps >= start) & (self.time_steps < end)
            self.metrics[self.motion.extracted_list[i]] = mask.clone().float()
            self.counts[i] = mask.sum().float()
        self.motion.motion_distribution = (self.counts / self.num_envs).unsqueeze(0)

    def _distribution_vectorized(self, update_metrics: bool):
        # Compute motion id per env, then bincount.
        # Clamp to valid range to avoid overflow after increment.
        ts = torch.clamp(self.time_steps, 0, self.motion.time_step_total - 1)
        # For variable lengths, use bucketize on end indices.
        # Use right=True so that ts == end goes to next motion (intervals are [start, end))
        motion_ids = torch.bucketize(ts, self._motion_ends, right=True)
        counts = torch.bincount(motion_ids, minlength=self.motion.num_motions).float()
        self.counts.copy_(counts)
        self.motion.motion_distribution = (self.counts / self.num_envs).unsqueeze(0)
        if update_metrics:
            # Update per-motion metrics as 0/1 mask
            # Equivalent to loop: metrics[name] = mask.float()
            for i in range(self.motion.num_motions):
                self.metrics[self.motion.extracted_list[i]] = (motion_ids == i).float()

    def _resample_command_trunc(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        epsilon = 1e-6
        uniform_bin_prob = 1.0 / float(self._bin_count)
        min_bin_prob = 0.1 * uniform_bin_prob

        if self._bin_failed.sum() <= epsilon:
            bin_probs = torch.full((self._bin_count,), uniform_bin_prob, device=self.device)
        else:
            bin_probs = self._bin_failed / self._bin_failed.sum()
            bin_probs = torch.clamp(bin_probs, min=min_bin_prob)
            bin_probs = bin_probs / bin_probs.sum()

        sampled_bins = torch.multinomial(bin_probs, env_ids.numel(), replacement=True)
        local_steps = (
            (sampled_bins + torch.rand((env_ids.numel(),), device=self.device))
            * float(self._bin_size)
        ).long()
        self.time_steps[env_ids] = torch.clamp(
            local_steps, 0, self.motion.time_step_total - 1
        )

    def update_until_resample(self, alpha: float) -> dict:
        # Start of _update_command
        self.time_steps += 1

        overflow_mask = self.time_steps >= self.motion.time_step_total
        valid_mask = ~overflow_mask
        cross_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if valid_mask.any():
            valid_ids = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
            cross_flags = self.motion.new_data_flag[self.time_steps[valid_ids]]
            cross_mask[valid_ids] = cross_flags

        total_mask = overflow_mask | cross_mask
        env_ids = torch.nonzero(total_mask, as_tuple=False).squeeze(-1)

        # Termination stats
        terminated = self._env.termination_manager.terminated
        non_timeout = terminated & (~self._env.termination_manager.time_outs)
        if torch.any(non_timeout):
            time_steps_clamped = torch.clamp(
                self.time_steps, 0, self.motion.time_step_total - 1
            )
            term_ids = torch.nonzero(non_timeout, as_tuple=False).squeeze(-1)
            term_steps = time_steps_clamped[term_ids]
            bin_ids = torch.clamp(
                term_steps // self._bin_size, 0, self._bin_count - 1
            )
            ones = torch.ones_like(bin_ids, dtype=torch.float32, device=self.device)
            self._current_bin_failed.index_put_((bin_ids,), ones, accumulate=True)

        # Dynamic distribution
        for i in range(self.motion.num_motions):
            start, end = self.motion.motion_indices[i]
            mask = (self.time_steps >= start) & (self.time_steps < end)
            self.metrics[self.motion.extracted_list[i]] = mask.clone().float()
            self.counts[i] = mask.sum().float()
        self.motion.motion_distribution = (self.counts / self.num_envs).unsqueeze(0)

        # EMA update
        self._bin_failed = alpha * self._current_bin_failed + (1.0 - alpha) * self._bin_failed
        self._current_bin_failed.zero_()

        # Truncated resample
        self._resample_command_trunc(env_ids)

        return {
            "env_ids": env_ids.numel(),
            "overflow": overflow_mask.sum().item(),
            "cross": cross_mask.sum().item(),
        }


class Timer:
    def __init__(self, device: torch.device):
        self.device = device

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def stamp(self) -> float:
        self._sync()
        return time.perf_counter()


def main():
    parser = argparse.ArgumentParser(description="Benchmark update_command -> resample_command (truncated)")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--num_motions", type=int, default=200)
    parser.add_argument("--frames_per_motion", type=int, default=3000)
    parser.add_argument(
        "--length_jitter",
        type=float,
        default=0.05,
        help="Randomize motion lengths by ±ratio around frames_per_motion (e.g., 0.2)",
    )
    parser.add_argument("--bin_size", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--terminated_prob", type=float, default=0.02)
    parser.add_argument("--timeout_ratio", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument(
        "--vectorized_metrics",
        action="store_true",
        help="Update per-motion metrics in vectorized mode (extra cost)",
    )
    parser.add_argument(
        "--vectorized",
        action="store_true",
        help="Use vectorized distribution counting (bincount) instead of per-motion loop",
    )
    args = parser.parse_args()

    device = _device_from_arg(args.device)
    torch.set_grad_enabled(False)

    bench = BenchModule(
        num_envs=args.num_envs,
        num_motions=args.num_motions,
        frames_per_motion=args.frames_per_motion,
        length_jitter=args.length_jitter,
        bin_size=args.bin_size,
        device=device,
        terminated_prob=args.terminated_prob,
        timeout_ratio=args.timeout_ratio,
        seed=args.seed,
    )
    timer = Timer(device)

    # Warmup
    for _ in range(args.warmup):
        bench._update_termination()
        if args.vectorized:
            bench.update_until_resample(args.alpha)
            bench._distribution_vectorized(update_metrics=args.vectorized_metrics)
        else:
            bench.update_until_resample(args.alpha)

    # Timed
    totals = {
        "mask": 0.0,
        "termination": 0.0,
        "distribution": 0.0,
        "ema": 0.0,
        "resample": 0.0,
        "total": 0.0,
    }
    counts = []

    for _ in range(args.iters):
        bench._update_termination()

        t0 = timer.stamp()
        bench.time_steps += 1

        overflow_mask = bench.time_steps >= bench.motion.time_step_total
        valid_mask = ~overflow_mask
        cross_mask = torch.zeros(bench.num_envs, dtype=torch.bool, device=device)
        if valid_mask.any():
            valid_ids = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
            cross_flags = bench.motion.new_data_flag[bench.time_steps[valid_ids]]
            cross_mask[valid_ids] = cross_flags
        total_mask = overflow_mask | cross_mask
        env_ids = torch.nonzero(total_mask, as_tuple=False).squeeze(-1)
        t1 = timer.stamp()

        terminated = bench._env.termination_manager.terminated
        non_timeout = terminated & (~bench._env.termination_manager.time_outs)
        if torch.any(non_timeout):
            time_steps_clamped = torch.clamp(
                bench.time_steps, 0, bench.motion.time_step_total - 1
            )
            term_ids = torch.nonzero(non_timeout, as_tuple=False).squeeze(-1)
            term_steps = time_steps_clamped[term_ids]
            bin_ids = torch.clamp(
                term_steps // bench._bin_size, 0, bench._bin_count - 1
            )
            ones = torch.ones_like(bin_ids, dtype=torch.float32, device=device)
            bench._current_bin_failed.index_put_((bin_ids,), ones, accumulate=True)
        t2 = timer.stamp()

        if args.vectorized:
            bench._distribution_vectorized(update_metrics=args.vectorized_metrics)
        else:
            bench._distribution_loop()
        t3 = timer.stamp()

        bench._bin_failed = args.alpha * bench._current_bin_failed + (1.0 - args.alpha) * bench._bin_failed
        bench._current_bin_failed.zero_()
        t4 = timer.stamp()

        # truncated resample
        if env_ids.numel() > 0:
            epsilon = 1e-6
            uniform_bin_prob = 1.0 / float(bench._bin_count)
            min_bin_prob = 0.1 * uniform_bin_prob

            if bench._bin_failed.sum() <= epsilon:
                bin_probs = torch.full((bench._bin_count,), uniform_bin_prob, device=device)
            else:
                bin_probs = bench._bin_failed / bench._bin_failed.sum()
                bin_probs = torch.clamp(bin_probs, min=min_bin_prob)
                bin_probs = bin_probs / bin_probs.sum()

            sampled_bins = torch.multinomial(bin_probs, env_ids.numel(), replacement=True)
            local_steps = (
                (sampled_bins + torch.rand((env_ids.numel(),), device=device))
                * float(bench._bin_size)
            ).long()
            bench.time_steps[env_ids] = torch.clamp(
                local_steps, 0, bench.motion.time_step_total - 1
            )
        t5 = timer.stamp()

        totals["mask"] += t1 - t0
        totals["termination"] += t2 - t1
        totals["distribution"] += t3 - t2
        totals["ema"] += t4 - t3
        totals["resample"] += t5 - t4
        totals["total"] += t5 - t0

        counts.append(env_ids.numel())

    def ms(x: float) -> float:
        return x * 1000.0 / float(args.iters)

    avg_resample = sum(counts) / float(len(counts))

    print("Benchmark config:")
    print(f"  device: {device}")
    print(f"  num_envs: {args.num_envs}")
    print(f"  num_motions: {args.num_motions}")
    print(f"  frames_per_motion: {args.frames_per_motion}")
    print(f"  length_jitter: {args.length_jitter}")
    print(f"  time_step_total: {bench.motion.time_step_total}")
    print(f"  bin_size: {bench._bin_size}")
    print(f"  bin_count: {bench._bin_count}")
    print(f"  terminated_prob: {args.terminated_prob}")
    print(f"  timeout_ratio: {args.timeout_ratio}")
    print(f"  alpha: {args.alpha}")
    print(f"  distribution_mode: {'vectorized' if args.vectorized else 'loop'}")
    if args.vectorized:
        print(f"  vectorized_metrics: {args.vectorized_metrics}")
    print()
    print("Average per-iter time (ms):")
    print(f"  mask + env_ids: {ms(totals['mask']):.4f}")
    print(f"  termination stats: {ms(totals['termination']):.4f}")
    print(f"  distribution loop: {ms(totals['distribution']):.4f}")
    print(f"  EMA update: {ms(totals['ema']):.4f}")
    print(f"  resample (trunc): {ms(totals['resample']):.4f}")
    print(f"  total: {ms(totals['total']):.4f}")
    print()
    print(f"Avg resample env_ids per iter: {avg_resample:.2f}")


if __name__ == "__main__":
    main()
