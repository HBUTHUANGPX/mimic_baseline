# Pruned Motion YAML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a script that scans `soma_uniform_bvh_export`, keeps one representative `.npz` per action, and writes a new YAML file compatible with the existing motion loader.

**Architecture:** Implement a focused Python CLI under `scripts/rsl_rl/` with small pure functions for action-key extraction, representative selection, and YAML emission. Cover the selection rules with pytest and verify the generated YAML can still be parsed by `scripts/rsl_rl/load_motion_file.py`.

**Tech Stack:** Python, argparse, pathlib, PyYAML, pytest

---

### Task 1: Add failing tests for pruning rules

**Files:**
- Create: `tests/test_generate_pruned_motion_yaml.py`
- Test: `tests/test_generate_pruned_motion_yaml.py`

- [ ] **Step 1: Write the failing test**

```python
def test_select_representative_prefers_sorted_non_m_variant():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_generate_pruned_motion_yaml.py -q`
Expected: FAIL because the new script module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def select_representative_motion(paths):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_generate_pruned_motion_yaml.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_generate_pruned_motion_yaml.py scripts/rsl_rl/generate_pruned_motion_yaml.py
git commit -m "feat: add pruned motion yaml generator"
```

### Task 2: Add CLI and YAML generation flow

**Files:**
- Create: `scripts/rsl_rl/generate_pruned_motion_yaml.py`
- Test: `tests/test_generate_pruned_motion_yaml.py`

- [ ] **Step 1: Write the failing test**

```python
def test_generate_yaml_writes_loader_compatible_motion_group(tmp_path):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_generate_pruned_motion_yaml.py -q`
Expected: FAIL because YAML generation is not implemented yet.

- [ ] **Step 3: Write minimal implementation**

```python
def generate_motion_yaml(...):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_generate_pruned_motion_yaml.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_generate_pruned_motion_yaml.py scripts/rsl_rl/generate_pruned_motion_yaml.py
git commit -m "feat: generate pruned motion yaml"
```

### Task 3: Verify against real dataset

**Files:**
- Modify: `scripts/rsl_rl/generate_pruned_motion_yaml.py`
- Test: `tests/test_generate_pruned_motion_yaml.py`

- [ ] **Step 1: Run the generator on the real motion directory**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && python -m scripts.rsl_rl.generate_pruned_motion_yaml --source-dir soma-retargeter/assets/motions/soma_uniform_bvh_export --output scripts/rsl_rl/motion_file_pruned.yaml`
Expected: YAML written successfully with one selected file per action.

- [ ] **Step 2: Verify the generated YAML loads**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && python scripts/rsl_rl/load_motion_file.py`
Expected: Existing loader still parses YAML format when pointed at the generated file or when reused in a targeted validation snippet.

- [ ] **Step 3: Run focused tests**

Run: `source /home/hpx/miniconda3/etc/profile.d/conda.sh && conda activate mimic_baseline && pytest tests/test_generate_pruned_motion_yaml.py -q`
Expected: PASS
