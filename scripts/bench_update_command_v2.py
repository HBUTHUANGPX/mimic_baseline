import argparse
import time
import torch


def _device_from_arg(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


class Bench:
    def __init__(
        self,
        num_envs,
        num_motions,
        frames_per_motion,
        length_jitter,
        bin_size,
        device,
        seed,
        adaptive_kernel_size=1,
        adaptive_lambda=0.8,
    ):
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
        self._motion_ends = self.motion_indices[:, 1].contiguous()

        self.new_data_flag = torch.zeros(self.time_step_total, dtype=torch.bool, device=device)
        self.new_data_flag[starts[1:]] = True

        # motion distribution (cmd3)
        self.extracted_list = [f"m{i:03d}" for i in range(num_motions)]
        self.counts = torch.zeros(num_motions, dtype=torch.float32, device=device)
        total_length = float(self.motion_lengths.sum().item())
        self.target_dist = (self.motion_lengths.float() / total_length).unsqueeze(0)
        self.motion_distribution = torch.full(
            (1, num_motions), 1.0 / num_motions, dtype=torch.float32, device=device
        )

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

        # commands_2 style buffers
        self.bin_count2 = int(self.time_step_total // self._bin_size) + 1
        self.bin_failed_count2 = torch.zeros(self.bin_count2, dtype=torch.float32, device=device)
        self._current_bin_failed2 = torch.zeros(self.bin_count2, dtype=torch.float32, device=device)
        self.kernel2 = torch.tensor(
            [adaptive_lambda**i for i in range(adaptive_kernel_size)],
            device=self.device,
        )
        self.kernel2 = self.kernel2 / self.kernel2.sum()
        self._metric_entropy = torch.zeros(1, device=self.device)
        self._metric_top1_prob = torch.zeros(1, device=self.device)
        self._metric_top1_bin = torch.zeros(1, device=self.device)

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

    # ===== commands_3 style =====
    def _cmd3_get_env_ids(self, timer):
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

        total_mask = overflow_mask | cross_mask| self.terminated
        env_ids = torch.nonzero(total_mask, as_tuple=False).squeeze(-1)
        t3 = timer.stamp()

        return env_ids, {
            "cmd3_overflow": t1 - t0,
            "cmd3_cross": t2 - t1,
            "cmd3_nonzero": t3 - t2,
        }

    def _cmd3_update_distribution(self, timer):
        t0 = timer.stamp()
        # for i in range(self.motion_lengths.numel()):
        #     start = self.motion_indices[i, 0]
        #     end = self.motion_indices[i, 1]
        #     mask = (self.time_steps >= start) & (self.time_steps < end)
        #     self.counts[i] = mask.sum().float()
        # self.motion_distribution = (self.counts / self.num_envs).unsqueeze(0)
        ts = torch.clamp(self.time_steps, 0, self.time_step_total - 1)
        # Intervals are [start, end); right=True ensures ts==end maps to next motion
        motion_ids = torch.bucketize(ts, self._motion_ends, right=True)
        counts = torch.bincount(motion_ids, minlength=self.motion_lengths.numel()).float()
        self.counts.copy_(counts)
        self.motion_distribution = (self.counts / self.num_envs).unsqueeze(0)
        
        t1 = timer.stamp()
        return {"cmd3_dist": t1 - t0}

    def _cmd3_resample(self, env_ids, timer):
        t0 = timer.stamp()
        if len(env_ids) == 0:
            t1 = timer.stamp()
            return {"cmd3_empty": t1 - t0, "cmd3_probs": 0.0, "cmd3_sample": 0.0, "cmd3_assign": 0.0}
        epsilon = 1e-6
        current_dist = self.motion_distribution.squeeze(0)
        target_dist = self.target_dist.squeeze(0)
        weights = target_dist / (current_dist + epsilon)
        probs = weights / weights.sum()
        t1 = timer.stamp()

        motion_ids = torch.multinomial(probs, len(env_ids), replacement=True)
        t2 = timer.stamp()

        selected_starts = self.motion_indices[motion_ids, 0]
        selected_lengths = self.motion_lengths[motion_ids]
        local_steps = (torch.rand((len(env_ids),), device=self.device) * (selected_lengths - 1)).long()
        self.time_steps[env_ids] = selected_starts + local_steps
        t3 = timer.stamp()

        return {
            "cmd3_empty": 0.0,
            "cmd3_probs": t1 - t0,
            "cmd3_sample": t2 - t1,
            "cmd3_assign": t3 - t2,
        }

    # ===== commands_2 style =====
    def _cmd2_get_env_ids(self, timer):
        t0 = timer.stamp()
        env_ids = torch.nonzero(self.time_steps >= self.time_step_total | self.terminated,as_tuple=False).squeeze(-1)
        t1 = timer.stamp()
        return env_ids, {"get_where": t1 - t0}

    def _cmd2_adaptive_sampling(self, env_ids, timer, adaptive_uniform_ratio, adaptive_kernel_size):
        t0 = timer.stamp()
        if len(env_ids) == 0:
            t1 = timer.stamp()
            return {
                "cmd2_mask": t1 - t0,
                "cmd2_fail_bins": 0.0,
                "cmd2_probs": 0.0,
                "cmd2_sample": 0.0,
                "cmd2_assign": 0.0,
                "cmd2_metrics": 0.0,
            }
        episode_failed = self.terminated[env_ids]
        t1 = timer.stamp()

        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count2) // max(self.time_step_total, 1),
                0,
                self.bin_count2 - 1,
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed2[:] = torch.bincount(
                fail_bins, minlength=self.bin_count2
            )
        t2 = timer.stamp()

        sampling_probabilities = (
            self.bin_failed_count2
            + adaptive_uniform_ratio / float(self.bin_count2)
        )
        if adaptive_kernel_size > 1:
            sampling_probabilities = torch.nn.functional.pad(
                sampling_probabilities.unsqueeze(0).unsqueeze(0),
                (0, adaptive_kernel_size - 1),
                mode="replicate",
            )
            sampling_probabilities = torch.nn.functional.conv1d(
                sampling_probabilities, self.kernel2.view(1, 1, -1)
            ).view(-1)
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()
        t3 = timer.stamp()

        sampled_bins = torch.multinomial(
            sampling_probabilities, len(env_ids), replacement=True
        )
        t4 = timer.stamp()

        self.time_steps[env_ids] = (
            (
                sampled_bins
                + torch.rand((len(env_ids),), device=self.device)
            )
            / self.bin_count2
            * (self.time_step_total - 1)
        ).long()
        t5 = timer.stamp()

        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / torch.tensor(float(self.bin_count2), device=self.device).log()
        pmax, imax = sampling_probabilities.max(dim=0)
        self._metric_entropy[:] = H_norm
        self._metric_top1_prob[:] = pmax
        self._metric_top1_bin[:] = imax.float() / self.bin_count2
        t6 = timer.stamp()

        return {
            "cmd2_mask": t1 - t0,
            "cmd2_fail_bins": t2 - t1,
            "cmd2_probs": t3 - t2,
            "cmd2_sample": t4 - t3,
            "cmd2_assign": t5 - t4,
            "cmd2_metrics": t6 - t5,
        }

    def _cmd2_post_update(self, adaptive_alpha, timer):
        t0 = timer.stamp()
        self.bin_failed_count2 = (
            adaptive_alpha * self._current_bin_failed2
            + (1.0 - adaptive_alpha) * self.bin_failed_count2
        )
        self._current_bin_failed2.zero_()
        t1 = timer.stamp()
        return {"cmd2_ema": t1 - t0}


class Timer:
    def __init__(self, device):
        self.device = device

    def stamp(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()


def main():
    p = argparse.ArgumentParser(description="Benchmark cmd1 vs cmd2 vs cmd3 sampling paths")
    p.add_argument("--mode", type=str, default="cmd1", choices=["cmd1", "cmd2", "cmd3", "both"])
    p.add_argument("--num_envs", type=int, default=4096 * 4 * 8)
    # p.add_argument("--num_envs", type=int, default=4096 * 4 * 1)
    # p.add_argument("--num_envs", type=int, default=4096 * 1 * 1)
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
    p.add_argument("--use_bincount", action="store_true", help="cmd1: use torch.bincount for bin_accum")
    p.add_argument("--adaptive_kernel_size", type=int, default=1)
    p.add_argument("--adaptive_lambda", type=float, default=0.8)
    p.add_argument("--adaptive_uniform_ratio", type=float, default=0.1)
    p.add_argument("--adaptive_alpha", type=float, default=0.001)
    args = p.parse_args()

    device = _device_from_arg(args.device)
    torch.set_grad_enabled(False)

    def ms(x):
        return x * 1000.0 / float(args.iters)

    def run_cmd1():
        bench = Bench(
            num_envs=args.num_envs,
            num_motions=args.num_motions,
            frames_per_motion=args.frames_per_motion,
            length_jitter=args.length_jitter,
            bin_size=args.bin_size,
            device=device,
            seed=args.seed,
            adaptive_kernel_size=args.adaptive_kernel_size,
            adaptive_lambda=args.adaptive_lambda,
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

        print("Mode: cmd1")
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
        print()

    def run_cmd2():
        bench = Bench(
            num_envs=args.num_envs,
            num_motions=args.num_motions,
            frames_per_motion=args.frames_per_motion,
            length_jitter=args.length_jitter,
            bin_size=args.bin_size,
            device=device,
            seed=args.seed,
            adaptive_kernel_size=args.adaptive_kernel_size,
            adaptive_lambda=args.adaptive_lambda,
        )
        timer = Timer(device)

        for _ in range(args.warmup):
            bench.update_termination(args.terminated_prob)
            bench.time_steps += 1
            env_ids, _ = bench._cmd2_get_env_ids(timer)
            bench._cmd2_adaptive_sampling(env_ids, timer, args.adaptive_uniform_ratio, args.adaptive_kernel_size)
            bench._cmd2_post_update(args.adaptive_alpha, timer)

        totals = {
            "get_env_ids": 0.0,
            "adaptive": 0.0,
            "post_update": 0.0,
            "total": 0.0,
            "get_where": 0.0,
            "cmd2_mask": 0.0,
            "cmd2_fail_bins": 0.0,
            "cmd2_probs": 0.0,
            "cmd2_sample": 0.0,
            "cmd2_assign": 0.0,
            "cmd2_metrics": 0.0,
            "cmd2_ema": 0.0,
        }
        resample_counts = []

        for _ in range(args.iters):
            bench.update_termination(args.terminated_prob)
            bench.time_steps += 1

            t0 = timer.stamp()
            env_ids, t_get = bench._cmd2_get_env_ids(timer)
            t1 = timer.stamp()

            t_adapt = bench._cmd2_adaptive_sampling(env_ids, timer, args.adaptive_uniform_ratio, args.adaptive_kernel_size)
            t2 = timer.stamp()

            t_post = bench._cmd2_post_update(args.adaptive_alpha, timer)
            t3 = timer.stamp()

            totals["get_env_ids"] += t1 - t0
            totals["adaptive"] += t2 - t1
            totals["post_update"] += t3 - t2
            totals["total"] += t3 - t0
            for k, v in t_get.items():
                totals[k] += v
            for k, v in t_adapt.items():
                totals[k] += v
            for k, v in t_post.items():
                totals[k] += v
            resample_counts.append(int(env_ids.numel()))

        print("Mode: cmd2")
        print("Benchmark config:")
        print(f"  device: {device}")
        print(f"  num_envs: {args.num_envs}")
        print(f"  num_motions: {args.num_motions}")
        print(f"  frames_per_motion: {args.frames_per_motion}")
        print(f"  length_jitter: {args.length_jitter}")
        print(f"  time_step_total: {bench.time_step_total}")
        print(f"  bin_size: {bench._bin_size}")
        print(f"  bin_count: {bench.bin_count2}")
        print(f"  terminated_prob: {args.terminated_prob}")
        print(f"  adaptive_kernel_size: {args.adaptive_kernel_size}")
        print(f"  adaptive_lambda: {args.adaptive_lambda}")
        print(f"  adaptive_uniform_ratio: {args.adaptive_uniform_ratio}")
        print(f"  adaptive_alpha: {args.adaptive_alpha}")
        print()
        print("Average per-iter time (ms):")
        print(f"  _get_env_ids: {ms(totals['get_env_ids']):.4f}")
        print(f"    where: {ms(totals['get_where']):.4f}")
        print(f"  _adaptive_sampling: {ms(totals['adaptive']):.4f}")
        print(f"    mask: {ms(totals['cmd2_mask']):.4f}")
        print(f"    fail_bins: {ms(totals['cmd2_fail_bins']):.4f}")
        print(f"    probs: {ms(totals['cmd2_probs']):.4f}")
        print(f"    sample: {ms(totals['cmd2_sample']):.4f}")
        print(f"    assign: {ms(totals['cmd2_assign']):.4f}")
        print(f"    metrics: {ms(totals['cmd2_metrics']):.4f}")
        print(f"  _post_update: {ms(totals['post_update']):.4f}")
        print(f"    EMA: {ms(totals['cmd2_ema']):.4f}")
        print(f"  total: {ms(totals['total']):.4f}")
        print()
        print(f"Avg resample env_ids per iter: {sum(resample_counts)/len(resample_counts):.2f}")
        print(f"Avg resample percent env_ids per iter: {(sum(resample_counts)/len(resample_counts))/args.num_envs:.5f}")
        print()

    def run_cmd3():
        bench = Bench(
            num_envs=args.num_envs,
            num_motions=args.num_motions,
            frames_per_motion=args.frames_per_motion,
            length_jitter=args.length_jitter,
            bin_size=args.bin_size,
            device=device,
            seed=args.seed,
            adaptive_kernel_size=args.adaptive_kernel_size,
            adaptive_lambda=args.adaptive_lambda,
        )
        timer = Timer(device)

        for _ in range(args.warmup):
            bench.update_termination(args.terminated_prob)
            bench.time_steps += 1
            env_ids, _ = bench._cmd3_get_env_ids(timer)
            bench._cmd3_update_distribution(timer)
            bench._cmd3_resample(env_ids, timer)

        totals = {
            "get_env_ids": 0.0,
            "dist": 0.0,
            "resample": 0.0,
            "total": 0.0,
            "cmd3_overflow": 0.0,
            "cmd3_cross": 0.0,
            "cmd3_nonzero": 0.0,
            "cmd3_dist": 0.0,
            "cmd3_empty": 0.0,
            "cmd3_probs": 0.0,
            "cmd3_sample": 0.0,
            "cmd3_assign": 0.0,
        }
        resample_counts = []

        for _ in range(args.iters):
            bench.update_termination(args.terminated_prob)
            bench.time_steps += 1

            t0 = timer.stamp()
            env_ids, t_get = bench._cmd3_get_env_ids(timer)
            t1 = timer.stamp()

            t_dist = bench._cmd3_update_distribution(timer)
            t2 = timer.stamp()

            t_resample = bench._cmd3_resample(env_ids, timer)
            t3 = timer.stamp()

            totals["get_env_ids"] += t1 - t0
            totals["dist"] += t2 - t1
            totals["resample"] += t3 - t2
            totals["total"] += t3 - t0
            for k, v in t_get.items():
                totals[k] += v
            for k, v in t_dist.items():
                totals[k] += v
            for k, v in t_resample.items():
                totals[k] += v
            resample_counts.append(int(env_ids.numel()))

        print("Mode: cmd3")
        print("Benchmark config:")
        print(f"  device: {device}")
        print(f"  num_envs: {args.num_envs}")
        print(f"  num_motions: {args.num_motions}")
        print(f"  frames_per_motion: {args.frames_per_motion}")
        print(f"  length_jitter: {args.length_jitter}")
        print(f"  time_step_total: {bench.time_step_total}")
        print(f"  bin_size: {bench._bin_size}")
        print(f"  terminated_prob: {args.terminated_prob}")
        print()
        print("Average per-iter time (ms):")
        print(f"  _get_env_ids: {ms(totals['get_env_ids']):.4f}")
        print(f"    overflow+valid: {ms(totals['cmd3_overflow']):.4f}")
        print(f"    cross_mask: {ms(totals['cmd3_cross']):.4f}")
        print(f"    nonzero: {ms(totals['cmd3_nonzero']):.4f}")
        print(f"  _update_distribution: {ms(totals['dist']):.4f}")
        print(f"    dist_loop: {ms(totals['cmd3_dist']):.4f}")
        print(f"  _resample_command: {ms(totals['resample']):.4f}")
        print(f"    empty: {ms(totals['cmd3_empty']):.4f}")
        print(f"    probs: {ms(totals['cmd3_probs']):.4f}")
        print(f"    sample: {ms(totals['cmd3_sample']):.4f}")
        print(f"    assign: {ms(totals['cmd3_assign']):.4f}")
        print(f"  total: {ms(totals['total']):.4f}")
        print()
        print(f"Avg resample env_ids per iter: {sum(resample_counts)/len(resample_counts):.2f}")
        print(f"Avg resample percent env_ids per iter: {(sum(resample_counts)/len(resample_counts))/args.num_envs:.5f}")
        print()

    if args.mode in ("cmd1", "both"):
        run_cmd1()
    if args.mode in ("cmd2", "both"):
        run_cmd2()
    if args.mode in ("cmd3", "both"):
        run_cmd3()


if __name__ == "__main__":
    main()
