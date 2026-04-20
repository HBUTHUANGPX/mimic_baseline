# SMPL-H and SMPL Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse Xperience-10M `annotation.hdf5` into standard SMPL-H motion tensors, visualize the reconstructed SMPL-H body in MuJoCo, then convert the motion to SMPL and visualize that variant as well.

**Architecture:** Add a focused motion utility module under `hdf5_parse/` that turns HDF5 quaternions into SMPL-H/SMPL axis-angle pose tensors, resolves local body-model files, and exports `.npz` motion bundles. Build one MuJoCo viewer on top of that utility layer so both SMPL-H and SMPL rendering paths share the same scene setup, frame filtering, and playback logic.

**Tech Stack:** Python, NumPy, SciPy, h5py, PyTorch, smplx, MuJoCo, pytest

---

### Task 1: Motion conversion contract tests

**Files:**
- Create: `tests/test_smpl_motion_tools.py`
- Create: `hdf5_parse/smpl_motion_tools.py`

- [ ] **Step 1: Write the failing test**

Cover:
- quaternion `wxyz -> rotvec` conversion;
- SMPL-H tensor assembly from `Ts_world_root`, `body_quats`, and hand quaternions;
- SMPL-H to SMPL body-pose padding rules;
- model-path resolution behavior;
- loading a short real slice from `hdf5_parse/hdf5/annotation.hdf5`.

- [ ] **Step 2: Run test to verify it fails**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_smpl_motion_tools.py -q`
Expected: FAIL because the helper module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement a compact helper module with:
- `SMPLMotionClip` data container;
- HDF5 loading and invalid-frame filtering;
- quaternion conversion utilities;
- SMPL-H tensor assembly;
- SMPL-H -> SMPL conversion;
- body-model path resolution and `.npz` export helpers.

- [ ] **Step 4: Run test to verify it passes**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_smpl_motion_tools.py -q`
Expected: PASS.

### Task 2: Viewer contract tests

**Files:**
- Create: `tests/test_smpl_body_mujoco_viewer.py`
- Create: `hdf5_parse/smpl_body_mujoco_viewer.py`

- [ ] **Step 1: Write the failing test**

Cover:
- CLI defaults and streamlined arguments;
- model selection between `smplh` and `smpl`;
- vertex sampling behavior for MuJoCo display points.

- [ ] **Step 2: Run test to verify it fails**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_smpl_body_mujoco_viewer.py -q`
Expected: FAIL because the viewer module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement a MuJoCo viewer that:
- loads a motion clip through `smpl_motion_tools.py`;
- instantiates the requested SMPL-H or SMPL body model;
- renders joints, bone segments, root frame, and sampled surface points;
- keeps the CLI focused on essential playback and model-selection inputs.

- [ ] **Step 4: Run test to verify it passes**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_smpl_body_mujoco_viewer.py -q`
Expected: PASS.

### Task 3: Export CLI and documentation

**Files:**
- Create: `hdf5_parse/export_smpl_motion.py`
- Modify: `hdf5_parse/visualization_notes.md`
- Create or Modify: `hdf5_parse/smpl_visualization_notes.md`

- [ ] **Step 1: Add export entry point**

Create a tiny CLI that saves either SMPL-H or SMPL motion tensors to `.npz` from the HDF5 source.

- [ ] **Step 2: Update documentation**

Document:
- which HDF5 fields feed SMPL-H reconstruction;
- how local-vs-global pose is handled;
- what the SMPL-H -> SMPL conversion drops or pads;
- how to run both the viewer and export commands;
- which local model files are used by default.

### Task 4: End-to-end verification

**Files:**
- Verify: `tests/test_smpl_motion_tools.py`
- Verify: `tests/test_smpl_body_mujoco_viewer.py`
- Verify: `hdf5_parse/smpl_body_mujoco_viewer.py`
- Verify: `hdf5_parse/export_smpl_motion.py`

- [ ] **Step 1: Run focused tests**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_smpl_motion_tools.py tests/test_smpl_body_mujoco_viewer.py -q`
Expected: PASS.

- [ ] **Step 2: Run viewer help**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && python hdf5_parse/smpl_body_mujoco_viewer.py --help`
Expected: usage text showing the streamlined CLI.

- [ ] **Step 3: Run a short export sample**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && python hdf5_parse/export_smpl_motion.py --model-type smplh --end 5 --output-path outputs/smpl_preview.npz`
Expected: a small `.npz` file with the expected pose arrays.

- [ ] **Step 4: Run a short viewer startup smoke test**

Run a short-lived viewer invocation and confirm the motion clip loads, model instantiation succeeds, and the summary prints the expected frame and pose shapes.
