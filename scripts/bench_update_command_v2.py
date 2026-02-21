import argparse
import time
import torch


def _device_from_arg(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


class Bench:
    def __init__(self, num_envs, num_motions, frames_per_motion, length_jitter, bin_size, device, seed):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        self.device = device
        self.num_envs = num_envs

        # motion lengths
        if length_jitter <= 0.0:
            lengths = torch.full((num_motions,), frames_per_motion, device=device, dtype=torch.long)
        else:
            low = max(1, int(frames_per_motion * (1.0 - length_jitter)))
            high = max(low + 1, int(frames_per_motion * (1.0 + length_jitter)) + 1)
            lengths = torch.randint(low=low, high=high, size=(num_motions,), device=device)

        self.motion_lengths = lengths
        self.time_step_total = int(lengths.sum().item())

        ends = torch.cumsum(lengths, dim=0)
        starts = torch.cat([torch.zeros(1, device=device, dtype=torch.long), ends[:-1]], dim=0)
        self.motion_indices = torch.stack([starts, ends], dim=1)

        self.new_data_flag = torch.zeros(self.time_step_total, dtype=torch.bool, device=device)
        self.new_data_flag[starts[1:]] = True

        self.time_steps = torch.randint(
            low=0,
            high=self.time_step_total,
            size=(num_envs,),
            device=device,
            dtype=torch.long,
        )

        self._bin_size = max(1, int(bin_size))
        self._bin_count = int((self.time_step_total + self._bin_size - 1) // self._bin_size)
        self._bin_failed = torch.zeros(self._bin_count, dtype=torch.float32, device=device)
        self._current_bin_failed = torch.zeros(self._bin_count, dtype=torch.float32, device=device)

        self.terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def update_termination(self, terminated_prob):
        self.terminated = torch.rand(self.num_envs, device=self.device) < terminated_prob

    def _get_env_ids_to_resample(self, timer):
        t0 = timer.stamp()
        overflow_mask = self.time_steps >= self.time_step_total
        valid_mask = ~overflow_mask
        t1 = timer.stamp()

        cross_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if valid_mask.any():
            valid_ids = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
            cross_flags = self.new_data_flag[self.time_steps[valid_ids]]
            cross_mask[valid_ids] = cross_flags
        t2 = timer.stamp()

        total_mask = overflow_mask | cross_mask | self.terminated
        env_ids = torch.nonzero(total_mask, as_tuple=False).squeeze(-1)
        t3 = timer.stamp()

        return env_ids, {
            "get_overflow": t1 - t0,
            "get_cross": t2 - t1,
            "get_nonzero": t3 - t2,
        }

    def _post_update_command(self, alpha: float, timer, use_bincount: bool):
        t0 = timer.stamp()
        non_timeout = self.terminated
        t1 = timer.stamp()

        if torch.any(non_timeout):
            time_steps_clamped = torch.clamp(self.time_steps, 0, self.time_step_total - 1)
            if use_bincount:
                term_steps = time_steps_clamped[non_timeout]
                bin_ids = torch.clamp(term_steps // self._bin_size, 0, self._bin_count - 1)
                inc = torch.bincount(bin_ids, minlength=self._bin_count).float()
                self._current_bin_failed.add_(inc)
            else:
                term_ids = torch.nonzero(non_timeout, as_tuple=False).squeeze(-1)
                term_steps = time_steps_clamped[term_ids]
                bin_ids = torch.clamp(term_steps // self._bin_size, 0, self._bin_count - 1)
                ones = torch.ones_like(bin_ids, dtype=torch.float32, device=self.device)
                self._current_bin_failed.index_put_((bin_ids,), ones, accumulate=True)
        t2 = timer.stamp()

        self._bin_failed = alpha * self._current_bin_failed + (1.0 - alpha) * self._bin_failed
        self._current_bin_failed.zero_()
        t3 = timer.stamp()

        return {
            "post_mask": t1 - t0,
            "post_bins": t2 - t1,
            "post_ema": t3 - t2,
        }

    def _resample_adaptive_sampling(self, env_ids, timer):
        t0 = timer.stamp()
        # ============ resample_probs ==================
        if len(env_ids) == 0:
            t1 = timer.stamp()
            return {
                "resample_empty": t1 - t0,
                "resample_probs": 0.0,
                "resample_sample": 0.0,
                "resample_rand": 0.0,
                "resample_local_steps": 0.0,
                "resample_assign": 0.0,
            }
        epsilon = 1e-6
        uniform_bin_prob = 1.0 / float(self._bin_count)
        min_bin_prob = 0.1 * uniform_bin_prob

        if self._bin_failed.sum() <= epsilon:
            bin_probs = torch.full((self._bin_count,), uniform_bin_prob, device=self.device)
        else:
            bin_probs = self._bin_failed / self._bin_failed.sum()
            bin_probs = torch.clamp(bin_probs, min=min_bin_prob)
            bin_probs = bin_probs / bin_probs.sum()
        t1 = timer.stamp()
        # ============ resample_sample ==================
        sampled_bins = torch.multinomial(bin_probs, len(env_ids), replacement=True)
        t2 = timer.stamp()
        # ============ resample_rand ====================
        rand_u = torch.rand((len(env_ids),), device=self.device)
        t3 = timer.stamp()
        # ============ resample_local_steps =============
        local_steps = ((sampled_bins + rand_u) * float(self._bin_size)).long()
        t4 = timer.stamp()
        # ============ resample_assign ==================
        self.time_steps[env_ids] = torch.clamp(local_steps, 0, self.time_step_total - 1)
        t5 = timer.stamp()

        return {
            "resample_empty": 0.0,
            "resample_probs": t1 - t0,
            "resample_sample": t2 - t1,
            "resample_rand": t3 - t2,
            "resample_local_steps": t4 - t3,
            "resample_assign": t5 - t4,
        }


class Timer:
    def __init__(self, device):
        self.device = device

    def stamp(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()


def main():
    p = argparse.ArgumentParser(description="Benchmark _get_env_ids_to_resample / _post_update_command / _resample_adaptive_sampling")
    p.add_argument("--num_envs", type=int, default=4096*4)
    p.add_argument("--num_motions", type=int, default=800)
    p.add_argument("--frames_per_motion", type=int, default=3000)
    p.add_argument("--length_jitter", type=float, default=0.05)
    p.add_argument("--bin_size", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--terminated_prob", type=float, default=0.98)
    p.add_argument("--alpha", type=float, default=0.001)
    p.add_argument("--use_bincount", action="store_true", help="Use torch.bincount for bin_accum")
    args = p.parse_args()

    device = _device_from_arg(args.device)
    torch.set_grad_enabled(False)

    bench = Bench(
        num_envs=args.num_envs,
        num_motions=args.num_motions,
        frames_per_motion=args.frames_per_motion,
        length_jitter=args.length_jitter,
        bin_size=args.bin_size,
        device=device,
        seed=args.seed,
    )
    timer = Timer(device)

    for _ in range(args.warmup):
        bench.update_termination(args.terminated_prob)
        bench.time_steps += 1
        env_ids, _ = bench._get_env_ids_to_resample(timer)
        bench._post_update_command(args.alpha, timer, args.use_bincount)
        bench._resample_adaptive_sampling(env_ids, timer)

    totals = {
        "get_env_ids": 0.0,
        "post_update": 0.0,
        "resample": 0.0,
        "total": 0.0,
        "get_overflow": 0.0,
        "get_cross": 0.0,
        "get_nonzero": 0.0,
        "post_mask": 0.0,
        "post_bins": 0.0,
        "post_ema": 0.0,
        "resample_empty": 0.0,
        "resample_probs": 0.0,
        "resample_sample": 0.0,
        "resample_rand": 0.0,
        "resample_local_steps": 0.0,
        "resample_assign": 0.0,
    }
    resample_counts = []

    for _ in range(args.iters):
        bench.update_termination(args.terminated_prob)
        bench.time_steps += 1

        t0 = timer.stamp()
        env_ids, t_get = bench._get_env_ids_to_resample(timer)
        t1 = timer.stamp()

        t_post = bench._post_update_command(args.alpha, timer, args.use_bincount)
        t2 = timer.stamp()

        t_resample = bench._resample_adaptive_sampling(env_ids, timer)
        t3 = timer.stamp()

        totals["get_env_ids"] += t1 - t0
        totals["post_update"] += t2 - t1
        totals["resample"] += t3 - t2
        totals["total"] += t3 - t0
        for k, v in t_get.items():
            totals[k] += v
        for k, v in t_post.items():
            totals[k] += v
        for k, v in t_resample.items():
            totals[k] += v
        resample_counts.append(int(env_ids.numel()))

    def ms(x):
        return x * 1000.0 / float(args.iters)

    print("Benchmark config:")
    print(f"  device: {device}")
    print(f"  num_envs: {args.num_envs}")
    print(f"  num_motions: {args.num_motions}")
    print(f"  frames_per_motion: {args.frames_per_motion}")
    print(f"  length_jitter: {args.length_jitter}")
    print(f"  time_step_total: {bench.time_step_total}")
    print(f"  bin_size: {bench._bin_size}")
    print(f"  bin_count: {bench._bin_count}")
    print(f"  terminated_prob: {args.terminated_prob}")
    print(f"  alpha: {args.alpha}")
    print(f"  use_bincount: {args.use_bincount}")
    print()
    print("Average per-iter time (ms):")
    print(f"  _get_env_ids_to_resample: {ms(totals['get_env_ids']):.4f}")
    print(f"    overflow+valid: {ms(totals['get_overflow']):.4f}")
    print(f"    cross_mask: {ms(totals['get_cross']):.4f}")
    print(f"    nonzero: {ms(totals['get_nonzero']):.4f}")
    print(f"  _post_update_command: {ms(totals['post_update']):.4f}")
    print(f"    mask: {ms(totals['post_mask']):.4f}")
    print(f"    bin_accum: {ms(totals['post_bins']):.4f}")
    print(f"    EMA: {ms(totals['post_ema']):.4f}")
    print(f"  _resample_adaptive_sampling: {ms(totals['resample']):.4f}")
    print(f"    empty: {ms(totals['resample_empty']):.4f}")
    print(f"    probs: {ms(totals['resample_probs']):.4f}")
    print(f"    sample: {ms(totals['resample_sample']):.4f}")
    print(f"    rand: {ms(totals['resample_rand']):.4f}")
    print(f"    local_steps: {ms(totals['resample_local_steps']):.4f}")
    print(f"    assign: {ms(totals['resample_assign']):.4f}")
    print(f"  total: {ms(totals['total']):.4f}")
    print()
    print(f"Avg resample env_ids per iter: {sum(resample_counts)/len(resample_counts):.2f}")
    print(f"Avg resample percent env_ids per iter: {(sum(resample_counts)/len(resample_counts))/args.num_envs:.5f}")


if __name__ == "__main__":
    main()
