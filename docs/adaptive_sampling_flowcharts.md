# Adaptive Sampling Flowcharts

本文档梳理 `LegacyBinAdaptiveSampling` 和 `SonicBinAdaptiveSampling` 的采样流程。

相关代码：

- `general_motion_tracker_whole_body_teleoperation/general_motion_tracker_whole_body_teleoperation/tasks/tracking/mdp/adaptive_sample.py`
- `general_motion_tracker_whole_body_teleoperation/general_motion_tracker_whole_body_teleoperation/tasks/tracking/mdp/commands.py`

## Common Lifecycle

两种采样器都实现同一套接口：

```text
on_resample_start -> build_sampling_probabilities -> torch.multinomial
-> on_resample_complete -> on_step_end
```

```mermaid
flowchart TD
    A[Environment requests resampling for env_ids] --> B[on_resample_start]
    B --> C[Update statistics from previous episodes]
    C --> D[build_sampling_probabilities]
    D --> E[Build probability for each valid time bin]
    E --> F[torch.multinomial samples sampled_bins]
    F --> G[MotionCommand samples concrete time_steps from bins]
    G --> H[on_resample_complete]
    H --> I[Persist state for sampled bins]
    I --> J[Training step ends]
    J --> K[on_step_end]
```

## LegacyBinAdaptiveSampling

核心思想：统计失败发生在哪些 time bin，并用失败次数的指数滑动平均作为之后的采样权重。失败越多的 bin，后续越容易被采样。

关键状态：

- `current_bin_failed`: 当前 step 收集到的失败 bin 计数。
- `bin_failed_count`: 跨 step 保留的失败统计，使用 EMA 更新。
- `kernel`: 对失败统计做邻域平滑的卷积核。

关键参数：

- `adaptive_alpha`: EMA 更新速度。
- `adaptive_uniform_ratio`: 加到失败统计上的均匀探索底噪，默认 `0.1`。
- `adaptive_kernel_size`: 平滑卷积核大小。
- `adaptive_lambda`: 平滑卷积核衰减系数。

```mermaid
flowchart TD
    A[Initialize Legacy sampler] --> A1[bin_failed_count = zeros]
    A --> A2[current_bin_failed = zeros]
    A --> A3[Build normalized smoothing kernel]

    B[on_resample_start] --> C{update_failure_statistics and env_ids non-empty?}
    C -- No --> D[Skip statistics update]
    C -- Yes --> E[Read terminated flags for env_ids]
    E --> F{Any failed episode?}
    F -- No --> D
    F -- Yes --> G[Read previous_time_steps]
    G --> H[failed_bin_ids = failed_time_steps / bin_frame_count]
    H --> I[Add 1 to current_bin_failed at failed_bin_ids]

    I --> J[build_sampling_probabilities]
    D --> J
    J --> K[base = bin_failed_count + adaptive_uniform_ratio / bin_count]
    K --> L[Pad base distribution]
    L --> M[Apply conv1d smoothing with kernel]
    M --> N[Mask invalid bins with valid_sampling_bin_mask]
    N --> O{Probability sum <= 0?}
    O -- Yes --> P[Fallback to uniform over valid bins]
    O -- No --> Q[Normalize probabilities]
    P --> Q
    Q --> R[Return sampling probabilities]

    R --> S[torch.multinomial samples bins]
    S --> T[on_resample_complete]
    T --> U[No-op for Legacy]

    U --> V[on_step_end]
    V --> W[bin_failed_count = alpha * current_bin_failed + 1-alpha * bin_failed_count]
    W --> X[Clear current_bin_failed]
```

### Legacy Summary

```text
Failure location -> current_bin_failed -> EMA bin_failed_count
-> add uniform offset -> smoothed probability -> sample bins
```

Legacy 关注“失败发生在哪个时间位置”。它更像一种基于失败热区的采样策略。

## StratifiedLegacyBinAdaptiveSampling

核心思想：继承 Legacy 的失败统计更新方式，但不改变原始 Legacy 行为；采样概率单独改成固定比例分层混合。默认配置下，`80%` 来自失败分布，`20%` 来自 valid bins 的均匀分布。

关键参数：

- `uniform_sampling_ratio`: 固定均匀采样比例，默认 `0.2`。
- 失败采样比例为 `1 - uniform_sampling_ratio`，默认 `0.8`。

```mermaid
flowchart TD
    A[Use Legacy failure statistics update] --> B[build_sampling_probabilities]
    B --> C[failure_scores = bin_failed_count]
    C --> D[Pad failure_scores]
    D --> E[Apply conv1d smoothing with kernel]
    E --> F[Mask invalid bins]
    F --> G{failure score sum > 0?}
    G -- Yes --> H[failure_distribution = normalized failure_scores]
    G -- No --> I[failure_distribution = uniform over valid bins]
    H --> J[uniform_distribution = uniform over valid bins]
    I --> J
    J --> K[sampling_prob = 0.8 * failure_distribution + 0.2 * uniform_distribution]
    K --> L[Mask invalid bins and normalize]
    L --> M[Return sampling probabilities]
```

### Stratified Legacy Summary

```text
Legacy failure statistics -> smoothed failure distribution
-> fixed 80/20 mix with uniform -> sample bins
```

## SonicBinAdaptiveSampling

核心思想：统计每个 episode 是从哪个 bin 开始的，以及从该起始 bin 开始后是否失败。采样时使用每个起始 bin 的失败率，而不是简单失败次数。

关键状态：

- `env_start_bin_ids`: 每个环境当前 episode 的起始 bin。
- `bin_visit_count`: 每个 bin 被作为起始 bin 的次数。
- `bin_fail_count`: 从每个 bin 开始后失败的次数。

关键参数：

- `mix_alpha`: 失败率分布和均匀分布的混合比例。
- `failure_cap_beta`: 失败率上限系数，用于抑制极端 bin 权重。

```mermaid
flowchart TD
    A[Initialize SONIC sampler] --> A1[bin_visit_count = zeros]
    A --> A2[bin_fail_count = zeros]
    A --> A3[env_start_bin_ids = zeros]

    B[on_resample_start] --> C{update_failure_statistics and env_ids non-empty?}
    C -- No --> D[Skip statistics update]
    C -- Yes --> E[start_bin_ids = env_start_bin_ids for env_ids]
    E --> F[Add 1 to bin_visit_count at start_bin_ids]
    F --> G[Read terminated flags for env_ids]
    G --> H{Any failed episode?}
    H -- Yes --> I[Add 1 to bin_fail_count at failed start_bin_ids]
    H -- No --> J[Only visit count is updated]

    I --> K[build_sampling_probabilities]
    J --> K
    D --> K
    K --> L[failure_rate = bin_fail_count / bin_visit_count for visited bins]
    L --> M[Unvisited bins have failure_rate = 0]
    M --> N[Mask invalid bins]
    N --> O[Compute mean failure rate over valid bins]
    O --> P[capped_failure_rate = min failure_rate, beta * mean_failure_rate]
    P --> Q{capped failure sum > 0?}
    Q -- Yes --> R[p_hat = capped_failure_rate / capped_sum]
    Q -- No --> S[p_hat = uniform over valid bins]
    R --> T[uniform_distribution over valid bins]
    S --> T
    T --> U[sampling_prob = mix_alpha * p_hat + 1-mix_alpha * uniform]
    U --> V[Mask invalid bins and normalize]
    V --> W[Return sampling probabilities]

    W --> X[torch.multinomial samples bins]
    X --> Y[on_resample_complete]
    Y --> Z[env_start_bin_ids env_ids = sampled_bins]

    Z --> AA[on_step_end]
    AA --> AB[No-op for SONIC]
```

### Sonic Summary

```text
Episode start bin -> visit/fail counts -> failure rate
-> capped failure distribution -> mix with uniform -> sample bins
```

SONIC 关注“从哪个起始位置开始更容易失败”。它更像一种基于起点失败率的采样策略。

## Main Difference

```mermaid
flowchart LR
    A[Legacy] --> B[Failure happened at this time bin]
    B --> C[Failure count EMA]
    C --> D[Sample failure hot spots]

    E[SONIC] --> F[Episode started from this time bin]
    F --> G[Failure rate of start bin]
    G --> H[Sample high-risk starts with uniform mixing]
```

简要对比：

| Item | LegacyBinAdaptiveSampling | StratifiedLegacyBinAdaptiveSampling | SonicBinAdaptiveSampling |
|---|---|---|---|
| 统计对象 | 失败发生的 bin | 失败发生的 bin | episode 起始 bin |
| 核心指标 | 失败次数 EMA | 失败次数 EMA | 起始 bin 失败率 |
| 是否记录 sampled_bins | 不记录 | 不记录 | 记录到 `env_start_bin_ids` |
| 是否在 `on_step_end` 更新 | 是，更新 EMA 并清零临时统计 | 是，继承 Legacy 更新 | 否 |
| 探索机制 | `adaptive_uniform_ratio` offset | 固定 `uniform_sampling_ratio=0.2` 均匀采样 | `(1 - mix_alpha) * uniform_distribution` |
| 平滑/约束 | `conv1d` kernel 平滑 | `conv1d` kernel 平滑 + 固定分层混合 | `failure_cap_beta` 限制极端失败率 |
