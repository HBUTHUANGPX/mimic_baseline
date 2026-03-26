# Motion Command Sampling Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible benchmark for the `MotionCommand` adaptive sampling hot path and run the requested sweep over environment count, motion count, and mean trajectory length.

**Architecture:** Implement a standalone benchmark harness that mirrors the runtime sampling path in `MotionCommand` without allocating unused motion pose tensors. Use the artifact file only to infer dataset schema and FPS, then generate synthetic trajectory metadata that preserves the requested scale and measures each sampling sub-step independently.

**Tech Stack:** Python, PyTorch, pytest, CSV/Markdown reporting

---

### Task 1: Benchmark contract tests

**Files:**
- Create: `tests/test_bench_motion_command_sampling.py`
- Create: `scripts/bench_motion_command_sampling.py`

- [ ] **Step 1: Write the failing test**

Define tests for:
- sampled time steps always staying within each motion's valid center-frame range;
- configurations whose bin count exceeds `torch.multinomial` limits raising a clear error.

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n mimic_baseline pytest tests/test_bench_motion_command_sampling.py -q`
Expected: FAIL because the benchmark module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add a standalone benchmark module with:
- synthetic motion metadata generation from `fps`;
- SONIC sampler step timing hooks;
- a compact valid-center mapping for the resampling hot path;
- config validation for multinomial bin-count limits.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n mimic_baseline pytest tests/test_bench_motion_command_sampling.py -q`
Expected: PASS.

### Task 2: Sweep runner and artifacts

**Files:**
- Modify: `scripts/bench_motion_command_sampling.py`
- Create: `outputs/bench_motion_command_sampling/`

- [ ] **Step 1: Add sweep/reporting support**

Implement the requested sweep over:
- `num_envs`: `4096`, `8192`, `16384`
- `num_motions`: `100`, `1000`, `5000`, `10000`
- `mean_length_s`: `3`, `10`, `300`

Record per-step timing summaries and unsupported cases.

- [ ] **Step 2: Run the benchmark**

Run the benchmark in the `mimic_baseline` conda env and write CSV/Markdown outputs under `outputs/bench_motion_command_sampling/`.

- [ ] **Step 3: Inspect results for pathological cases**

Confirm whether any combination exceeds the `torch.multinomial` category limit or other runtime constraints and document that explicitly.

### Task 3: Final verification

**Files:**
- Verify: `tests/test_bench_motion_command_sampling.py`
- Verify: `outputs/bench_motion_command_sampling/*.csv`
- Verify: `outputs/bench_motion_command_sampling/*.md`

- [ ] **Step 1: Re-run tests**

Run: `conda run -n mimic_baseline pytest tests/test_bench_motion_command_sampling.py -q`
Expected: PASS.

- [ ] **Step 2: Re-run a representative benchmark sample**

Run one small benchmark command and confirm it produces step-level timing rows and summary files.

- [ ] **Step 3: Summarize findings**

Prepare a concise summary of dominant costs, scaling behavior, and any unsupported configurations.
