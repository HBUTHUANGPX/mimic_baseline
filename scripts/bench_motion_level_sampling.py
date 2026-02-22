import argparse
import time
import torch


def _device_from_arg(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


class Timer:
    def __init__(self, device):
        self.device = device

    def stamp(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()


class MotionSamplerBench:
    def __init__(
        self,
        num_envs,
        num_motions,
        frames_per_motion,
        length_jitter,
        device,
        seed,
    ):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        self.device = device
        self.num_envs = num_envs
        self.num_motions = num_motions

        if length_jitter <= 0.0:
            lengths = torch.full(
                (num_motions,), frames_per_motion, device=device, dtype=torch.long
            )
        else:
            low = max(1, int(frames_per_motion * (1.0 - length_jitter)))
            high = max(low + 1, int(frames_per_motion * (1.0 + length_jitter)) + 1)
            lengths = torch.randint(
                low=low, high=high, size=(num_motions,), device=device
            )

        self.motion_lengths = lengths
        self.time_step_total = int(lengths.sum().item())

        ends = torch.cumsum(lengths, dim=0)
        starts = torch.cat(
            [torch.zeros(1, device=device, dtype=torch.long), ends[:-1]], dim=0
        )
        self.motion_indices = torch.stack([starts, ends], dim=1)
        self.new_data_flag = torch.zeros(
            self.time_step_total, dtype=torch.bool, device=device
        )
        self.new_data_flag[starts[1:]] = True
        self.motion_ends = self.motion_indices[:, 1].contiguous()

        self.time_steps = torch.randint(
            low=0,
            high=self.time_step_total,
            size=(num_envs,),
            device=device,
            dtype=torch.long,
        )

        total_len = float(self.motion_lengths.sum().item())
        self.target_dist = (self.motion_lengths.float() / total_len).unsqueeze(0)
        self.motion_distribution = torch.full(
            (1, num_motions), 1.0 / num_motions, dtype=torch.float32, device=device
        )

        # failure stats per motion (for improved sampler)
        self.motion_fail_counts = torch.zeros(
            num_motions, dtype=torch.float32, device=device
        )
        self.motion_fail_weights = torch.ones(
            num_motions, dtype=torch.float32, device=device
        )
        self._fail_update_step = 0

        self.terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def update_termination(self, terminated_prob):
        self.terminated = (
            torch.rand(self.num_envs, device=self.device) < terminated_prob
        )

    def _get_env_ids_to_resample(self):
        overflow_mask = self.time_steps >= self.time_step_total
        valid_mask = ~overflow_mask
        cross_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if valid_mask.any():
            valid_ids = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
            cross_flags = self.new_data_flag[self.time_steps[valid_ids]]
            cross_mask[valid_ids] = cross_flags
        total_mask = overflow_mask | cross_mask | self.terminated
        env_ids = torch.nonzero(total_mask, as_tuple=False).squeeze(-1)
        return env_ids

    def _update_distribution(self):
        # for i in range(self.num_motions):
        #     start, end = self.motion_indices[i]
        #     mask = (self.time_steps >= start) & (self.time_steps < end)
        #     self.motion_distribution[0, i] = mask.sum().float() / float(self.num_envs)

        # Vectorized: map time_steps to motion id, then bincount
        ts = torch.clamp(self.time_steps, 0, self.time_step_total - 1)
        # Intervals are [start, end); right=True ensures ts==end maps to next motion
        self.motion_ids = torch.bucketize(ts, self.motion_ends, right=True)
        self.counts = torch.bincount(
            self.motion_ids, minlength=self.num_motions
        ).float()
        self.motion_distribution = (self.counts / self.num_envs).unsqueeze(0)

    # ===== baseline: commands_3 style =====
    def resample_baseline(self, env_ids, timer):
        t0 = timer.stamp()
        if len(env_ids) == 0:
            t1 = timer.stamp()
            return {
                "resample_empty": t1 - t0,
                "resample_probs": 0.0,
                "resample_sample": 0.0,
                "resample_assign": 0.0,
            }

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
        local_steps = (
            torch.rand((len(env_ids),), device=self.device) * (selected_lengths - 1)
        ).long()
        self.time_steps[env_ids] = selected_starts + local_steps
        t3 = timer.stamp()

        return {
            "resample_empty": 0.0,
            "resample_probs": t1 - t0,
            "resample_sample": t2 - t1,
            "resample_assign": t3 - t2,
        }

    # ===== improved: motion-level with failure weights =====
    def _update_failure_weights(self, update_interval, timer, momentum=0.9):
        t0 = timer.stamp()
        if update_interval <= 0:
            t1 = timer.stamp()
            return {"fail_gate": t1 - t0, "fail_map": 0.0, "fail_count": 0.0, "fail_ema": 0.0, "fail_norm": 0.0}
        if (self._fail_update_step % update_interval) != 0:
            self._fail_update_step += 1
            t1 = timer.stamp()
            return {"fail_gate": t1 - t0, "fail_map": 0.0, "fail_count": 0.0, "fail_ema": 0.0, "fail_norm": 0.0}
        t1 = timer.stamp()

        # motion_id per env
        # ts = torch.clamp(self.time_steps, 0, self.time_step_total - 1)
        # motion_ids = torch.bucketize(ts, self.motion_ends, right=True)
        fail_motion_ids = self.motion_ids[self.terminated]
        t2 = timer.stamp()
        if fail_motion_ids.numel() > 0:
            counts = torch.bincount(fail_motion_ids, minlength=self.num_motions).float()
        else:
            counts = torch.zeros(
                self.num_motions, dtype=torch.float32, device=self.device
            )
        t3 = timer.stamp()

        # EMA update of motion-level failure counts
        self.motion_fail_counts = (
            momentum * self.motion_fail_counts + (1.0 - momentum) * counts
        )
        t4 = timer.stamp()
        # normalize to weights (avoid zero)
        eps = 1e-6
        w = self.motion_fail_counts + eps
        self.motion_fail_weights = w / w.mean()
        self._fail_update_step += 1
        t5 = timer.stamp()

        return {
            "fail_gate": t1 - t0,
            "fail_map": t2 - t1,
            "fail_count": t3 - t2,
            "fail_ema": t4 - t3,
            "fail_norm": t5 - t4,
        }

    def resample_improved(self, env_ids, timer, update_interval):
        t0 = timer.stamp()
        if len(env_ids) == 0:
            t1 = timer.stamp()
            return {
                "resample_empty": t1 - t0,
                "resample_total": t1 - t0,
                "fail_update": 0.0,
                "probs_total": 0.0,
                "probs_base": 0.0,
                "probs_fail": 0.0,
                "probs_norm": 0.0,
                "sample": 0.0,
                "assign": 0.0,
            }

        # low-frequency update of failure weights
        t_fail = self._update_failure_weights(update_interval, timer)
        t1 = timer.stamp()

        epsilon = 1e-6
        current_dist = self.motion_distribution.squeeze(0)
        target_dist = self.target_dist.squeeze(0)
        base_weights = target_dist / (current_dist + epsilon)
        t2 = timer.stamp()
        weights = base_weights * self.motion_fail_weights
        t3 = timer.stamp()
        probs = weights / weights.sum()
        t4 = timer.stamp()
        motion_ids = torch.multinomial(probs, len(env_ids), replacement=True)
        t5 = timer.stamp()

        selected_starts = self.motion_indices[motion_ids, 0]
        selected_lengths = self.motion_lengths[motion_ids]
        local_steps = (
            torch.rand((len(env_ids),), device=self.device) * (selected_lengths - 1)
        ).long()
        self.time_steps[env_ids] = selected_starts + local_steps
        t6 = timer.stamp()

        return {
            "resample_empty": 0.0,
            "resample_total": t6 - t0,
            "fail_update": t1 - t0,
            "fail_gate": t_fail["fail_gate"],
            "fail_map": t_fail["fail_map"],
            "fail_count": t_fail["fail_count"],
            "fail_ema": t_fail["fail_ema"],
            "fail_norm": t_fail["fail_norm"],
            "probs_total": t4 - t1,
            "probs_base": t2 - t1,
            "probs_fail": t3 - t2,
            "probs_norm": t4 - t3,
            "sample": t5 - t4,
            "assign": t6 - t5,
        }


def main():
    p = argparse.ArgumentParser(
        description="Motion-level sampling: baseline vs improved"
    )
    p.add_argument("--num_envs", type=int, default=4096 * 4 * 8)
    p.add_argument("--num_motions", type=int, default=800)
    p.add_argument("--frames_per_motion", type=int, default=3000)
    p.add_argument("--length_jitter", type=float, default=0.05)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"]
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--terminated_prob", type=float, default=0.9)
    p.add_argument(
        "--update_interval",
        type=int,
        default=20,
        help="Failure weight update interval (steps)",
    )
    args = p.parse_args()

    device = _device_from_arg(args.device)
    torch.set_grad_enabled(False)

    bench = MotionSamplerBench(
        num_envs=args.num_envs,
        num_motions=args.num_motions,
        frames_per_motion=args.frames_per_motion,
        length_jitter=args.length_jitter,
        device=device,
        seed=args.seed,
    )
    timer = Timer(device)

    # Warmup
    for _ in range(args.warmup):
        bench.update_termination(args.terminated_prob)
        bench.time_steps += 1
        env_ids = bench._get_env_ids_to_resample()
        bench._update_distribution()
        bench.resample_baseline(env_ids, timer)

    # Baseline timing
    totals_base = {"resample": 0.0, "probs": 0.0, "sample": 0.0, "assign": 0.0}
    for _ in range(args.iters):
        bench.update_termination(args.terminated_prob)
        bench.time_steps += 1
        env_ids = bench._get_env_ids_to_resample()
        bench._update_distribution()
        t = bench.resample_baseline(env_ids, timer)
        totals_base["resample"] += sum(t.values())
        totals_base["probs"] += t["resample_probs"]
        totals_base["sample"] += t["resample_sample"]
        totals_base["assign"] += t["resample_assign"]

    # Improved timing
    totals_imp = {
        "resample_total": 0.0,
        "fail_update": 0.0,
        "fail_gate": 0.0,
        "fail_map": 0.0,
        "fail_count": 0.0,
        "fail_ema": 0.0,
        "fail_norm": 0.0,
        "probs_total": 0.0,
        "probs_base": 0.0,
        "probs_fail": 0.0,
        "probs_norm": 0.0,
        "sample": 0.0,
        "assign": 0.0,
    }
    for _ in range(args.iters):
        bench.update_termination(args.terminated_prob)
        bench.time_steps += 1
        env_ids = bench._get_env_ids_to_resample()
        bench._update_distribution()
        t = bench.resample_improved(env_ids, timer, args.update_interval)
        totals_imp["resample_total"] += t["resample_total"]
        totals_imp["fail_update"] += t["fail_update"]
        totals_imp["fail_gate"] += t["fail_gate"]
        totals_imp["fail_map"] += t["fail_map"]
        totals_imp["fail_count"] += t["fail_count"]
        totals_imp["fail_ema"] += t["fail_ema"]
        totals_imp["fail_norm"] += t["fail_norm"]
        totals_imp["probs_total"] += t["probs_total"]
        totals_imp["probs_base"] += t["probs_base"]
        totals_imp["probs_fail"] += t["probs_fail"]
        totals_imp["probs_norm"] += t["probs_norm"]
        totals_imp["sample"] += t["sample"]
        totals_imp["assign"] += t["assign"]

    def ms(x):
        return x * 1000.0 / float(args.iters)

    print("Benchmark config:")
    print(f"  device: {device}")
    print(f"  num_envs: {args.num_envs}")
    print(f"  num_motions: {args.num_motions}")
    print(f"  frames_per_motion: {args.frames_per_motion}")
    print(f"  length_jitter: {args.length_jitter}")
    print(f"  time_step_total: {bench.time_step_total}")
    print(f"  terminated_prob: {args.terminated_prob}")
    print(f"  update_interval: {args.update_interval}")
    print()

    print("Baseline (length-balanced motion sampling):")
    print(f"  resample total: {ms(totals_base['resample']):.4f}")
    print(f"    probs: {ms(totals_base['probs']):.4f}")
    print(f"    sample: {ms(totals_base['sample']):.4f}")
    print(f"    assign: {ms(totals_base['assign']):.4f}")
    print()

    print("Improved (motion-level + failure weights, low-freq update):")
    print(f"  resample total: {ms(totals_imp['resample_total']):.4f}")
    print(f"    fail_update: {ms(totals_imp['fail_update']):.4f}")
    print(f"      gate: {ms(totals_imp['fail_gate']):.4f}")
    print(f"      map: {ms(totals_imp['fail_map']):.4f}")
    print(f"      count: {ms(totals_imp['fail_count']):.4f}")
    print(f"      ema: {ms(totals_imp['fail_ema']):.4f}")
    print(f"      norm: {ms(totals_imp['fail_norm']):.4f}")
    print(f"    probs total: {ms(totals_imp['probs_total']):.4f}")
    print(f"      base_weights: {ms(totals_imp['probs_base']):.4f}")
    print(f"      fail_weights: {ms(totals_imp['probs_fail']):.4f}")
    print(f"      norm: {ms(totals_imp['probs_norm']):.4f}")
    print(f"    sample: {ms(totals_imp['sample']):.4f}")
    print(f"    assign: {ms(totals_imp['assign']):.4f}")


if __name__ == "__main__":
    main()
