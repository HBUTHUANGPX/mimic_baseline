# Q1 Runtime State Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `infer.py`'s handwritten runtime-state assembly with a protocol-aware builder and add a Q1 protocol file so multi-input models can run through the same deployment flow.

**Architecture:** Keep protocol files as the source of model IO semantics, introduce a small `RuntimeStateBuilder` that inspects protocol state inputs and produces `RuntimeState` values, and let `infer.py` delegate state assembly to that builder. The first builder version will intentionally use the existing `policy` observation stream as a temporary shared source for all state-fed observation tensors.

**Tech Stack:** Python, pytest, numpy, ONNX Runtime, YAML protocol loader.

---

### Task 1: Add failing tests for protocol-aware runtime-state building

**Files:**
- Modify: `awesome_deploy/tests/test_infer_refactor.py`
- Test: `awesome_deploy/tests/test_infer_refactor.py`

- [ ] Step 1: Write failing tests for `RuntimeStateBuilder` and Q1 protocol loading.
- [ ] Step 2: Run `conda run -n mimic_baseline pytest awesome_deploy/tests/test_infer_refactor.py -q` and verify failure.
- [ ] Step 3: Implement the minimal production code to satisfy the tests.
- [ ] Step 4: Re-run the same test and verify pass.

### Task 2: Add protocol-aware runtime-state builder and wire `infer.py`

**Files:**
- Create: `awesome_deploy/awesome_deploy/inference/runtime_state_builder.py`
- Modify: `awesome_deploy/awesome_deploy/inference/__init__.py`
- Modify: `awesome_deploy/awesome_deploy/utils/infer.py`
- Test: `awesome_deploy/tests/test_infer_refactor.py`

- [ ] Step 1: Introduce `RuntimeStateBuilder` with protocol/signature-aware state assembly.
- [ ] Step 2: Replace handwritten `RuntimeState(...)` assembly in `infer.py` with builder usage.
- [ ] Step 3: Keep `policy_action` consumption unchanged.
- [ ] Step 4: Re-run targeted tests.

### Task 3: Add Q1 protocol file and regression coverage

**Files:**
- Create: `awesome_deploy/awesome_deploy/policy/q1/2026-03-20_15-37-52_xsens_all_fsq_s/policy.protocol.yaml`
- Modify: `awesome_deploy/tests/test_protocol_loader.py`
- Test: `awesome_deploy/tests/test_protocol_loader.py`

- [ ] Step 1: Write the Q1 protocol using real ONNX tensor names and `policy_action` as the primary semantic key.
- [ ] Step 2: Add a test that loads the checked-in Q1 protocol and validates its bindings.
- [ ] Step 3: Run protocol-loader tests and verify pass.

### Task 4: Full verification

**Files:**
- Modify: none
- Test: `awesome_deploy/tests/test_protocol_loader.py`, `awesome_deploy/tests/test_inference_engine.py`, `awesome_deploy/tests/test_infer_refactor.py`

- [ ] Step 1: Run `conda run -n mimic_baseline pytest awesome_deploy/tests/test_protocol_loader.py awesome_deploy/tests/test_inference_engine.py awesome_deploy/tests/test_infer_refactor.py -q`.
- [ ] Step 2: Run `conda run -n mimic_baseline python -m compileall awesome_deploy/awesome_deploy/inference awesome_deploy/awesome_deploy/utils`.
- [ ] Step 3: Summarize current temporary limitation: multiple protocol state inputs still reuse the existing `policy` observation source until observation producers are split.
