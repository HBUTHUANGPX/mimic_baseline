import argparse
import csv
import importlib.util
import os
import time
from itertools import product

import torch


def load_bench_module():
    path = os.path.join("scripts", "bench_update_command_v2.py")
    spec = importlib.util.spec_from_file_location("bench_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _device_from_arg(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def run_cmd1(mod, device, cfg):
    bench = mod.Bench(
        num_envs=cfg["num_envs"],
        num_motions=cfg["num_motions"],
        frames_per_motion=cfg["frames_per_motion"],
        length_jitter=cfg["length_jitter"],
        bin_size=cfg["bin_size"],
        device=device,
        seed=cfg["seed"],
        adaptive_kernel_size=cfg["adaptive_kernel_size"],
        adaptive_lambda=cfg["adaptive_lambda"],
    )
    timer = mod.Timer(device)

    for _ in range(cfg["warmup"]):
        bench.update_termination(cfg["terminated_prob"])
        bench.time_steps += 1
        env_ids, _ = bench._get_env_ids_to_resample(timer)
        bench._post_update_command(cfg["alpha"], timer, cfg["use_bincount"])
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

    for _ in range(cfg["iters"]):
        bench.update_termination(cfg["terminated_prob"])
        bench.time_steps += 1

        t0 = timer.stamp()
        env_ids, t_get = bench._get_env_ids_to_resample(timer)
        t1 = timer.stamp()

        t_post = bench._post_update_command(cfg["alpha"], timer, cfg["use_bincount"])
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

    return totals, bench.time_step_total, bench._bin_count, sum(resample_counts) / len(resample_counts)


def run_cmd2(mod, device, cfg):
    bench = mod.Bench(
        num_envs=cfg["num_envs"],
        num_motions=cfg["num_motions"],
        frames_per_motion=cfg["frames_per_motion"],
        length_jitter=cfg["length_jitter"],
        bin_size=cfg["bin_size"],
        device=device,
        seed=cfg["seed"],
        adaptive_kernel_size=cfg["adaptive_kernel_size"],
        adaptive_lambda=cfg["adaptive_lambda"],
    )
    timer = mod.Timer(device)

    for _ in range(cfg["warmup"]):
        bench.update_termination(cfg["terminated_prob"])
        bench.time_steps += 1
        env_ids, _ = bench._cmd2_get_env_ids(timer)
        bench._cmd2_adaptive_sampling(env_ids, timer, cfg["adaptive_uniform_ratio"], cfg["adaptive_kernel_size"])
        bench._cmd2_post_update(cfg["adaptive_alpha"], timer)

    totals = {
        "get_env_ids": 0.0,
        "adaptive": 0.0,
        "post_update": 0.0,
        "total": 0.0,
        "cmd2_overflow": 0.0,
        "cmd2_cross": 0.0,
        "cmd2_nonzero": 0.0,
        "cmd2_mask": 0.0,
        "cmd2_fail_bins": 0.0,
        "cmd2_probs": 0.0,
        "cmd2_sample": 0.0,
        "cmd2_assign": 0.0,
        "cmd2_metrics": 0.0,
        "cmd2_ema": 0.0,
    }
    resample_counts = []

    for _ in range(cfg["iters"]):
        bench.update_termination(cfg["terminated_prob"])
        bench.time_steps += 1

        t0 = timer.stamp()
        env_ids, t_get = bench._cmd2_get_env_ids(timer)
        t1 = timer.stamp()

        t_adapt = bench._cmd2_adaptive_sampling(env_ids, timer, cfg["adaptive_uniform_ratio"], cfg["adaptive_kernel_size"])
        t2 = timer.stamp()

        t_post = bench._cmd2_post_update(cfg["adaptive_alpha"], timer)
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

    return totals, bench.time_step_total, bench.bin_count2, sum(resample_counts) / len(resample_counts)


def run_cmd3(mod, device, cfg):
    bench = mod.Bench(
        num_envs=cfg["num_envs"],
        num_motions=cfg["num_motions"],
        frames_per_motion=cfg["frames_per_motion"],
        length_jitter=cfg["length_jitter"],
        bin_size=cfg["bin_size"],
        device=device,
        seed=cfg["seed"],
        adaptive_kernel_size=cfg["adaptive_kernel_size"],
        adaptive_lambda=cfg["adaptive_lambda"],
    )
    timer = mod.Timer(device)

    for _ in range(cfg["warmup"]):
        bench.update_termination(cfg["terminated_prob"])
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

    for _ in range(cfg["iters"]):
        bench.update_termination(cfg["terminated_prob"])
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

    return totals, bench.time_step_total, bench._bin_count, sum(resample_counts) / len(resample_counts)


def main():
    p = argparse.ArgumentParser(description="Sweep cmd1/cmd2/cmd3 over parameter grid")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--frames_per_motion", type=int, default=3000)
    p.add_argument("--length_jitter", type=float, default=0.05)
    p.add_argument("--bin_size", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.001)
    p.add_argument("--use_bincount", action="store_true")
    p.add_argument("--adaptive_kernel_size", type=int, default=1)
    p.add_argument("--adaptive_lambda", type=float, default=0.8)
    p.add_argument("--adaptive_uniform_ratio", type=float, default=0.1)
    p.add_argument("--adaptive_alpha", type=float, default=0.001)
    p.add_argument("--out_csv", type=str, default="scripts/bench_cmd123_sweep.csv")
    args = p.parse_args()

    mod = load_bench_module()
    device = _device_from_arg(args.device)

    num_envs_list = [4096, 4096 * 4, 4095 * 32]
    num_motions_list = [1, 50, 800, 16000]
    terminated_list = [0.05, 0.2, 0.6, 0.98]

    fieldnames = [
        "mode",
        "num_envs",
        "num_motions",
        "terminated_prob",
        "time_step_total",
        "bin_count",
        "avg_resample_env_ids",
        "iters",
        "warmup",
        "frames_per_motion",
        "length_jitter",
        "bin_size",
        "alpha",
        "use_bincount",
        "adaptive_kernel_size",
        "adaptive_lambda",
        "adaptive_uniform_ratio",
        "adaptive_alpha",
        "total_ms",
        "get_env_ids_ms",
        "post_update_ms",
        "resample_ms",
        "dist_ms",
        "adaptive_ms",
    ]

    def add_metrics(row, totals, iters):
        row["total_ms"] = totals.get("total", 0.0) * 1000.0 / iters
        row["get_env_ids_ms"] = totals.get("get_env_ids", 0.0) * 1000.0 / iters
        row["post_update_ms"] = totals.get("post_update", 0.0) * 1000.0 / iters
        row["resample_ms"] = totals.get("resample", 0.0) * 1000.0 / iters
        row["dist_ms"] = totals.get("dist", 0.0) * 1000.0 / iters
        row["adaptive_ms"] = totals.get("adaptive", 0.0) * 1000.0 / iters
        return row

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for num_envs, num_motions, terminated_prob in product(num_envs_list, num_motions_list, terminated_list):
            cfg = {
                "num_envs": num_envs,
                "num_motions": num_motions,
                "terminated_prob": terminated_prob,
                "iters": args.iters,
                "warmup": args.warmup,
                "seed": args.seed,
                "frames_per_motion": args.frames_per_motion,
                "length_jitter": args.length_jitter,
                "bin_size": args.bin_size,
                "alpha": args.alpha,
                "use_bincount": args.use_bincount,
                "adaptive_kernel_size": args.adaptive_kernel_size,
                "adaptive_lambda": args.adaptive_lambda,
                "adaptive_uniform_ratio": args.adaptive_uniform_ratio,
                "adaptive_alpha": args.adaptive_alpha,
            }

            for mode in ("cmd1", "cmd2", "cmd3"):
                if mode == "cmd1":
                    totals, time_step_total, bin_count, avg_env_ids = run_cmd1(mod, device, cfg)
                elif mode == "cmd2":
                    totals, time_step_total, bin_count, avg_env_ids = run_cmd2(mod, device, cfg)
                else:
                    totals, time_step_total, bin_count, avg_env_ids = run_cmd3(mod, device, cfg)

                row = {
                    "mode": mode,
                    "num_envs": num_envs,
                    "num_motions": num_motions,
                    "terminated_prob": terminated_prob,
                    "time_step_total": time_step_total,
                    "bin_count": bin_count,
                    "avg_resample_env_ids": avg_env_ids,
                    "iters": args.iters,
                    "warmup": args.warmup,
                    "frames_per_motion": args.frames_per_motion,
                    "length_jitter": args.length_jitter,
                    "bin_size": args.bin_size,
                    "alpha": args.alpha,
                    "use_bincount": args.use_bincount,
                    "adaptive_kernel_size": args.adaptive_kernel_size,
                    "adaptive_lambda": args.adaptive_lambda,
                    "adaptive_uniform_ratio": args.adaptive_uniform_ratio,
                    "adaptive_alpha": args.adaptive_alpha,
                }
                row = add_metrics(row, totals, args.iters)
                writer.writerow(row)

    print(f"Wrote CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
