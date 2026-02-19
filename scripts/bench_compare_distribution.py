# scripts/bench_compare_distribution.py
import importlib.util
import os
import torch

def main():
    path = os.path.join("scripts", "bench_update_command.py")
    spec = importlib.util.spec_from_file_location("bench", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 与你的基准配置一致
    num_envs = 4096
    num_motions = 200
    frames_per_motion = 3000
    length_jitter = 0.05
    bin_size = 50
    seed = 123

    m = bench.BenchModule(
        num_envs=num_envs,
        num_motions=num_motions,
        frames_per_motion=frames_per_motion,
        length_jitter=length_jitter,
        bin_size=bin_size,
        device=device,
        terminated_prob=0.02,
        timeout_ratio=0.5,
        seed=seed,
    )

    # 前进一步，模拟 update 后的 time_steps
    m._update_termination()
    m.time_steps += 1

    # loop
    m._distribution_loop()
    counts_loop = m.counts.clone()
    metrics_loop = {k: v.clone() for k, v in m.metrics.items()}

    # vectorized
    m._distribution_vectorized(update_metrics=True)
    counts_vec = m.counts.clone()
    metrics_vec = {k: v.clone() for k, v in m.metrics.items()}

    # compare counts
    counts_equal = torch.equal(counts_loop, counts_vec)
    max_count_diff = (counts_loop - counts_vec).abs().max().item()

    # compare metrics
    max_metric_diff = 0.0
    for k in metrics_loop.keys():
        diff = (metrics_loop[k] - metrics_vec[k]).abs().max().item()
        if diff > max_metric_diff:
            max_metric_diff = diff

    print(f"Device: {device}")
    print(f"Counts equal: {counts_equal}, max_count_diff: {max_count_diff}")
    print(f"Max metric diff: {max_metric_diff}")

if __name__ == "__main__":
    main()
