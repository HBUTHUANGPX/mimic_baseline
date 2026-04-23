# Motion Reconstruction Human-Only Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add modular source loading and inference-path selection to `motion_reconstruction`, then expose a thin `hdf5_parse` wrapper that visualizes `annotation_soma.npz` through the shared human-only path.

**Architecture:** Introduce a small inference/source abstraction rather than rewriting training/runtime code. Keep the existing raw training path intact, add a dedicated human-only source adapter for `hdf5_parse` exports, update reconstruction/evaluation APIs to support optional robot-origin data, and let both the CLI and `hdf5_parse` call the same package-level visualization entrypoint.

**Tech Stack:** Python, NumPy, PyTorch, MuJoCo, pytest

---

### Task 1: Contract tests for modular source loading

**Files:**
- Create: `tests/test_motion_reconstruction_sources.py`
- Modify: `motion_reconstruction/evaluation/reconstruct.py`

- [ ] **Step 1: Write the failing test**

Cover:
- raw source still resolves from config/runtime;
- `hdf5-human` source can read `human_global_pos/human_global_quat/human_joint_names`;
- center frame sampling obeys `history/future`;
- missing `robot_*` fields are accepted for `hdf5-human`;
- human anchor positions become the visualization robot anchor positions.

- [ ] **Step 2: Run test to verify it fails**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_motion_reconstruction_sources.py -q`
Expected: FAIL because the source adapter layer does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add a focused source adapter module and only extend existing runtime code where compatibility demands it.

- [ ] **Step 4: Run test to verify it passes**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_motion_reconstruction_sources.py -q`
Expected: PASS.

### Task 2: Contract tests for human-path inference

**Files:**
- Create: `tests/test_motion_reconstruction_inference.py`
- Modify: `motion_reconstruction/evaluation/reconstruct.py`

- [ ] **Step 1: Write the failing test**

Cover:
- `human` inference path works with human-only source bundles;
- `robot` path requires robot features and errors clearly otherwise;
- `ReconstructionResult.metrics()` skips unavailable branches cleanly;
- result metadata still contains names/anchors/display names for visualization.

- [ ] **Step 2: Run test to verify it fails**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_motion_reconstruction_inference.py -q`
Expected: FAIL because inference-path selection and optional result branches do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add a reusable reconstruction API that accepts a prepared source bundle plus an inference path selector.

- [ ] **Step 4: Run test to verify it passes**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_motion_reconstruction_inference.py -q`
Expected: PASS.

### Task 3: CLI and package API wiring

**Files:**
- Modify: `motion_reconstruction/cli/visualize.py`
- Modify: `motion_reconstruction/__init__.py`
- Modify: `motion_reconstruction/visualization/__init__.py`
- Create: `hdf5_parse/visualize_hdf5_soma_npz.py`

- [ ] **Step 1: Write the failing CLI/API test**

Add tests for:
- `cli.visualize` argument parsing with `--source/--motion-npz/--inference-path`;
- package-level helper callable from `hdf5_parse`;
- human-only path rejects `pair=robot` and `pair=both` with a clear message.

- [ ] **Step 2: Run test to verify it fails**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_motion_reconstruction_inference.py tests/test_motion_reconstruction_sources.py -q`
Expected: FAIL because the new CLI/API entrypoints do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Keep the old CLI defaults intact for raw sources, and add a thin `hdf5_parse` wrapper that forwards into `motion_reconstruction`.

- [ ] **Step 4: Run test to verify it passes**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_motion_reconstruction_inference.py tests/test_motion_reconstruction_sources.py -q`
Expected: PASS.

### Task 4: Documentation updates

**Files:**
- Modify: `motion_reconstruction/README.md`
- Modify: `motion_reconstruction/docs/usage.md`
- Modify: `readme.md`
- Modify: `hdf5_parse/README.md`
- Create: `docs/motion_reconstruction_hdf5_visualization.md`

- [ ] **Step 1: Update user-facing docs**

Document:
- the new source modes;
- the new inference-path flag;
- how to visualize `hdf5_parse/out/annotation_soma.npz`;
- the semantic meaning of `human encoder -> decoder` output.

- [ ] **Step 2: Update integration docs**

Document:
- why human-only source bypasses `RawMotionLoader`;
- why robot anchor uses the human anchor trajectory;
- limitations of the current viewer pairings.

### Task 5: Verification

**Files:**
- Verify: `motion_reconstruction/evaluation/reconstruct.py`
- Verify: `motion_reconstruction/cli/visualize.py`
- Verify: `hdf5_parse/visualize_hdf5_soma_npz.py`
- Verify: `tests/test_motion_reconstruction_sources.py`
- Verify: `tests/test_motion_reconstruction_inference.py`

- [ ] **Step 1: Run focused tests**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_motion_reconstruction_sources.py tests/test_motion_reconstruction_inference.py -q`
Expected: PASS.

- [ ] **Step 2: Run CLI help**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && python -m motion_reconstruction.cli.visualize --help`
Expected: usage shows `--source`, `--motion-npz`, and `--inference-path`.

- [ ] **Step 3: Run hdf5 wrapper help**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && python hdf5_parse/visualize_hdf5_soma_npz.py --help`
Expected: usage text for visualizing exported human-only `.npz`.

- [ ] **Step 4: Run a non-GUI smoke path**

Run a script-level smoke test that builds a human-only source bundle and reconstructs a few frames without launching MuJoCo, to verify the integration end to end.
Expected: successful result object with `recon_from_human_feature`.
