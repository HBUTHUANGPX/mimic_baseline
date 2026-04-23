# HDF5 to SOMA Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CUDA-only exporter under `hdf5_parse/` that converts `annotation.hdf5` full-body motion and captions into a save_retarget-style human `.npz` plus timeline/text metadata.

**Architecture:** Add a focused export module that parses HDF5 motion/text fields, prepares SMPL body tensors, runs SOMA-X pose inversion directly, converts SOMA joint transforms into save_retarget-compatible local/global arrays, and writes a custom human-only `.npz`. Keep CLI wiring thin and put reusable parsing/conversion logic in separate helpers so tests can cover them without needing a GUI path.

**Tech Stack:** Python, NumPy, h5py, PyTorch, smplx, SOMA-X, pytest

---

### Task 1: Spec-driven export contract tests

**Files:**
- Create: `tests/test_hdf5_soma_export.py`
- Modify: `hdf5_parse/smpl_motion_tools.py`

- [ ] **Step 1: Write the failing test**

Add unit tests for:
- valid-frame filtering from `Ts_world_root`, `body_quats`, and `betas`;
- `frame_nums` / `video/frame_number` alignment checks;
- caption JSON parsing into four text pools plus per-frame indices;
- `"UNKNOWN"` reserved at pool index `0`;
- interaction forward-fill rule from one timestamp to the next.

- [ ] **Step 2: Run test to verify it fails**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_hdf5_soma_export.py -q`
Expected: FAIL because the export helpers do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create parsing helpers in a new export module, and only extend `smpl_motion_tools.py` where it already owns reusable SMPL-body extraction behavior.

- [ ] **Step 4: Run test to verify it passes**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_hdf5_soma_export.py -q`
Expected: PASS.

### Task 2: SOMA conversion and payload-shaping tests

**Files:**
- Create: `tests/test_hdf5_soma_payload.py`
- Create: `hdf5_parse/hdf5_soma_export.py`

- [ ] **Step 1: Write the failing test**

Add tests that:
- build a small synthetic SOMA skeleton payload and verify human-only save format;
- zero out non-selected joints while preserving `human_reference_local_transforms`;
- drop the virtual SOMA `Root` and keep `Hips` as exported root;
- ensure exported payload excludes every `robot_*` key;
- verify `timeline_frame_indices`, `soma_*`, and `smpl_*` arrays are preserved.

- [ ] **Step 2: Run test to verify it fails**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_hdf5_soma_payload.py -q`
Expected: FAIL because the payload builder does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement focused helpers for:
- building export skeleton metadata from SOMA outputs;
- converting world transforms to local `[xyz | quat_xyzw]`;
- masking non-selected joints;
- packaging the final `.npz` payload.

- [ ] **Step 4: Run test to verify it passes**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_hdf5_soma_payload.py -q`
Expected: PASS.

### Task 3: CLI exporter and SOMA-X integration

**Files:**
- Create: `hdf5_parse/export_hdf5_to_soma_npz.py`
- Modify: `hdf5_parse/hdf5_soma_export.py`

- [ ] **Step 1: Write the failing CLI smoke test**

Extend `tests/test_hdf5_soma_export.py` or add a focused CLI test that checks:
- default input/output paths;
- CUDA-only argument defaults;
- output path naming under `hdf5_parse/out/`;
- optional overrides for input/output/model paths.

- [ ] **Step 2: Run test to verify it fails**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_hdf5_soma_export.py -q`
Expected: FAIL because the CLI does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Wire the CLI to:
- load the HDF5 file;
- run SMPL-body extraction;
- construct `smplx` and `SOMA-X` objects on CUDA;
- run pose inversion in chunks;
- save the final `.npz` to `hdf5_parse/out/annotation_soma.npz` by default.

- [ ] **Step 4: Run test to verify it passes**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_hdf5_soma_export.py -q`
Expected: PASS.

### Task 4: README and exporter documentation

**Files:**
- Modify: `readme.md`
- Create: `hdf5_parse/README.md`
- Create: `hdf5_parse/hdf5_soma_export_notes.md`

- [ ] **Step 1: Update top-level README**

Document:
- where the exporter lives;
- required CUDA/SOMA-X/SMPL assets;
- the one-command export workflow.

- [ ] **Step 2: Add `hdf5_parse` README**

Document:
- input HDF5 fields;
- output `.npz` fields;
- how text alignment works;
- why only valid human frames are saved;
- how non-selected joints are zeroed.

- [ ] **Step 3: Add detailed exporter notes**

Document:
- `SMPL-H body -> SMPL -> SOMA` conversion path;
- `frame_nums` / `device_timestamp` / caption timestamp relationship;
- `UNKNOWN` text index semantics;
- differences between `human_reference_local_transforms` and dynamic transforms.

### Task 5: Verification

**Files:**
- Verify: `tests/test_hdf5_soma_export.py`
- Verify: `tests/test_hdf5_soma_payload.py`
- Verify: `hdf5_parse/export_hdf5_to_soma_npz.py`
- Verify: `hdf5_parse/hdf5_soma_export.py`
- Verify: `readme.md`
- Verify: `hdf5_parse/README.md`
- Verify: `hdf5_parse/hdf5_soma_export_notes.md`

- [ ] **Step 1: Run focused unit tests**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_hdf5_soma_export.py tests/test_hdf5_soma_payload.py -q`
Expected: PASS.

- [ ] **Step 2: Run CLI help**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && python hdf5_parse/export_hdf5_to_soma_npz.py --help`
Expected: usage text with default input/output locations.

- [ ] **Step 3: Run a short real export smoke test**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && python hdf5_parse/export_hdf5_to_soma_npz.py --end-frame 8`
Expected: an `.npz` file under `hdf5_parse/out/` with human-only payload keys, timeline indices, and four text index arrays.

- [ ] **Step 4: Inspect the exported payload**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && python - <<'PY'\nimport numpy as np\npayload=np.load('hdf5_parse/out/annotation_soma.npz', allow_pickle=False)\nprint(sorted(payload.files))\nprint(payload['num_frames'].item(), payload['timeline_frame_indices'].shape)\nPY`
Expected: keys include `human_*`, `timeline_frame_indices`, `main_task_*`, `sub_task_*`, `current_action_*`, `interaction_*`, `smpl_*`, `soma_*`, and exclude `robot_*`.
