# Nymeria Motion Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `nymeria_parse` exporters that convert Nymeria MVNX body motion into standard SMPL npz, SOMA BVH, and annotation metadata npz.

**Architecture:** Mirror `hdf5_parse` with `utils/`, `motion_export/`, `scripts/`, and `docs/`. MVNX parsing and Xsens-to-SMPL conversion live in reusable modules; CLIs are thin wrappers.

**Tech Stack:** Python standard library XML/CSV parsing, NumPy, SciPy rotations, PyTorch/SOMA-X for SMPL-to-SOMA, pytest.

---

### Task 1: MVNX And Text Parsing

**Files:**
- Create: `nymeria_parse/utils/mvnx.py`
- Create: `nymeria_parse/motion_export/core.py`
- Test: `tests/test_nymeria_mvnx.py`
- Test: `tests/test_nymeria_annotation_export.py`

- [ ] Write failing tests for MVNX metadata/frame parsing.
- [ ] Implement MVNX parsing with frame slicing.
- [ ] Write failing tests for narration CSV text alignment.
- [ ] Implement `annotation.npz` payload construction.
- [ ] Run focused tests.

### Task 2: SMPL Export

**Files:**
- Create: `nymeria_parse/utils/xsens_smpl.py`
- Create: `nymeria_parse/motion_export/smpl.py`
- Create: `nymeria_parse/scripts/export_nymeria_to_smpl.py`
- Test: `tests/test_nymeria_smpl_export.py`

- [ ] Write failing tests for standard SMPL output fields.
- [ ] Implement Xsens-to-SMPL rotation conversion and local pose construction.
- [ ] Implement standard SMPL npz writer.
- [ ] Add CLI.
- [ ] Run focused tests and CLI help.

### Task 3: SOMA BVH Export

**Files:**
- Create: `nymeria_parse/motion_export/soma_bvh.py`
- Create: `nymeria_parse/scripts/export_nymeria_to_soma_bvh.py`
- Test: `tests/test_nymeria_soma_bvh_export.py`

- [ ] Write failing tests for BVH canonicalization and Root zero channels.
- [ ] Reuse `hdf5_parse` SOMA inversion and BVH writer semantics.
- [ ] Add CLI.
- [ ] Run focused tests and a short smoke export when CUDA is available.

### Task 4: Docs And Verification

**Files:**
- Create: `nymeria_parse/README.md`
- Modify: `nymeria_parse/docs/design.md`

- [ ] Document commands and output formats.
- [ ] Run the Nymeria test suite.
- [ ] Run short-range CLI smoke tests for annotation and SMPL.
