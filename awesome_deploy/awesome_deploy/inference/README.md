# awesome_deploy.inference Roadmap

This document explains how to externalize model IO semantics into
`policy.protocol.yaml` so that inference code stays decoupled from concrete
model names, tensor counts, and backend implementations.

## 1. Goal

`awesome_deploy.inference` is meant to solve one narrow problem:

- The deployment framework should know how to run a model.
- The framework should not hardcode what a specific model calls its inputs or outputs.
- Model-specific IO semantics should live in a protocol file, not in Python control flow.

That means:

- `InferenceEngine` owns backend loading, execution, and persistent buffers.
- `ProtocolAdapter` executes a declarative `ModelProtocol`.
- `policy.protocol.yaml` describes how runtime state, buffers, and outputs bind to a specific model.
- `infer.py` should gradually stop constructing model IO rules in `_build_model_protocol()` and load them from disk with `load_protocol_from_file()`.

## 2. Core Principle

Always start from the actual model signature, not from handwritten guesses.

The correct workflow is:

1. Inspect the exported model and list all real input names, output names, and shapes.
2. Decide the semantic meaning of each tensor.
3. Write `policy.protocol.yaml` that binds those raw tensor names to semantic runtime resources.
4. Validate the protocol with unit tests or a minimal dry run.

If a model changes from:

- `obs -> actions`

to:

- `actor_obs + actor_fsq_obs -> actions`

the framework code should ideally not change. Only the protocol and the runtime state producer should change.

## 3. Current Architecture

The protocol system is centered on these classes:

- [InferenceEngine](/home/hpx/HPX_LOCO_2/mimic_baseline/awesome_deploy/awesome_deploy/inference/engine.py)
  Runs one inference step by composing backend, adapter, and buffers.
- [ModelProtocol](/home/hpx/HPX_LOCO_2/mimic_baseline/awesome_deploy/awesome_deploy/inference/protocol.py)
  Pure declarative schema for inputs, outputs, and persistent buffers.
- [ProtocolAdapter](/home/hpx/HPX_LOCO_2/mimic_baseline/awesome_deploy/awesome_deploy/inference/io_adapters/protocol_adapter.py)
  Executes a `ModelProtocol` without knowing model-specific tensor names ahead of time.
- [load_protocol_from_file](/home/hpx/HPX_LOCO_2/mimic_baseline/awesome_deploy/awesome_deploy/inference/protocol_loader.py)
  Parses `policy.protocol.yaml` into `ModelProtocol`.
- [TRANSFORM_REGISTRY](/home/hpx/HPX_LOCO_2/mimic_baseline/awesome_deploy/awesome_deploy/inference/transform_registry.py)
  Named transforms referenced by protocol files.

## 4. Authoring Workflow

### Step 1: Inspect the model signature

For ONNX models, first inspect the real IO signature. Example:

```bash
conda run -n mimic_baseline python - <<'PY'
import onnxruntime as ort

model_path = "/home/hpx/HPX_LOCO_2/mimic_baseline/logs/rsl_rl/q1_flat/2026-03-20_15-37-52_xsens_all_fsq_s/exported/policy.onnx"
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

print("inputs:")
for tensor in session.get_inputs():
    print(" ", tensor.name, tensor.shape, tensor.type)

print("outputs:")
for tensor in session.get_outputs():
    print(" ", tensor.name, tensor.shape, tensor.type)
PY
```

For the Q1 example model, the inspected result was:

- inputs: `actor_obs [1, 581]`, `actor_fsq_obs [1, 704]`
- outputs: `actions [1, 29]`

This is the authoritative source of truth.

### Step 2: Decide semantic names

Now decide what each raw tensor means in your deployment pipeline.

Example mapping:

- raw input `actor_obs` means "main actor observation"
- raw input `actor_fsq_obs` means "FSQ observation branch"
- raw output `actions` means "policy action"

These semantic names do not need to match the raw tensor names. A protocol exists specifically to bridge those two layers.

### Step 3: Decide runtime sources

Each input must come from one of four sources:

- `state`: per-step values provided by the simulator or caller
- `buffer`: persistent values stored across timesteps
- `result`: values produced by the current inference result
- `constant`: hardcoded literal values

Typical examples:

- observation tensors usually come from `state`
- `time_step`, `prev_action`, recurrent state, history windows usually come from `buffer`
- action history roll-forward often comes from `result`
- small fixed flags can come from `constant`

### Step 4: Add transforms only when needed

Protocol transforms should only solve representation conversion, not business logic.

Good transform examples:

- 1-D vector -> batch-first tensor
- scalar -> `float32[1, 1]`
- model output -> flattened `float32` action vector

Bad transform examples:

- constructing observations by reading simulator internals directly
- robot-specific motion logic
- complex branching based on policy version

If a transform begins to contain domain logic, it likely belongs in runtime-state production, not in the protocol layer.

### Step 5: Write the protocol file

Recommended location:

- put `policy.protocol.yaml` next to the exported model directory

Example:

- [policy.protocol.yaml](/home/hpx/HPX_LOCO_2/mimic_baseline/awesome_deploy/awesome_deploy/policy/g1/2026-02-26_22-16-14_G1_slowly_walk/policy.protocol.yaml)

### Step 6: Validate

Validation should happen in two layers:

1. Static load validation
   - `load_protocol_from_file()` should parse without errors.
2. Runtime validation
   - the protocol should build tensors whose names and shapes match the real model.

Recommended verification commands:

```bash
conda run -n mimic_baseline pytest awesome_deploy/tests/test_protocol_loader.py -q
conda run -n mimic_baseline python -m compileall awesome_deploy/awesome_deploy/inference
```

## 5. Protocol File Schema

### Top-level keys

Every protocol file currently supports:

```yaml
version: 1
inputs: {}
outputs: {}
buffer_initializers: {}
per_step_buffer_updates: {}
```

Rules:

- `version` must be `1`
- `inputs` is required
- `outputs` is required
- `buffer_initializers` is optional
- `per_step_buffer_updates` is optional

### `inputs`

`inputs` maps raw backend input tensor names to binding declarations.

Example:

```yaml
inputs:
  obs:
    source_kind: state
    source_key: policy_obs
    transform: as_batch_vector
```

Meaning:

- the model has a real input tensor named `obs`
- its value comes from `runtime_state.values["policy_obs"]`
- before inference, it is converted through `as_batch_vector`

Supported `source_kind` values:

- `state`
- `buffer`
- `result`
- `constant`

Field rules:

- `source_key` is required for `state`, `buffer`, and `result`
- `value` is required for `constant`
- `transform` is optional and must exist in `TRANSFORM_REGISTRY`

### `outputs`

`outputs` maps raw backend output tensor names to semantic output declarations.

Example:

```yaml
outputs:
  actions:
    target_kind: primary
    target_key: policy_action
    transform: flatten_float32
```

Meaning:

- the model has a real output tensor named `actions`
- it is the main action output of this policy
- its semantic name is `policy_action`
- it is flattened before being exposed to the caller

Supported `target_kind` values:

- `primary`
- `output`

Rules:

- only one output may be marked as `primary`
- `target_key` is always required
- `transform` is optional

### `buffer_initializers`

`buffer_initializers` declares persistent buffers that must exist before the first step.

Example:

```yaml
buffer_initializers:
  time_step:
    init_kind: constant
    value: 1

  prev_action:
    init_kind: zeros_from_output
    tensor_name: actions
    axis: 1
```

Supported `init_kind` values:

- `constant`
- `zeros_from_output`

Rules:

- `constant` requires `value`
- `zeros_from_output` requires `tensor_name`
- `axis` defaults to `0`

Use `zeros_from_output` when buffer size should follow a real model output dimension such as action dimension.

### `per_step_buffer_updates`

`per_step_buffer_updates` defines how persistent buffers are rolled forward after each inference step.

Example:

```yaml
per_step_buffer_updates:
  time_step:
    source_kind: buffer
    source_key: time_step
    transform: increment_int

  prev_action:
    source_kind: buffer
    source_key: action

  action:
    source_kind: result
    source_key: policy_action
```

Typical use cases:

- increment rollout step
- move `action -> prev_action`
- copy current policy action into the persistent `action` buffer

## 6. Two Concrete Examples

### Example A: single-observation model

This is the current G1-style pattern:

```yaml
version: 1

inputs:
  obs:
    source_kind: state
    source_key: policy_obs
    transform: as_batch_vector

  time_step:
    source_kind: buffer
    source_key: time_step
    transform: as_batch_scalar

outputs:
  actions:
    target_kind: primary
    target_key: policy_action
    transform: flatten_float32
```

Interpretation:

- runtime code provides one semantic observation vector: `policy_obs`
- protocol binds it to the model tensor named `obs`

### Example B: dual-observation model

For a Q1-style model with two observation branches:

```yaml
version: 1

inputs:
  actor_obs:
    source_kind: state
    source_key: actor_obs
    transform: as_batch_vector

  actor_fsq_obs:
    source_kind: state
    source_key: actor_fsq_obs
    transform: as_batch_vector

outputs:
  actions:
    target_kind: primary
    target_key: q1_action
    transform: flatten_float32

buffer_initializers:
  action:
    init_kind: zeros_from_output
    tensor_name: actions
    axis: 1

per_step_buffer_updates:
  action:
    source_kind: result
    source_key: q1_action
```

Interpretation:

- the framework does not need to know the names `actor_obs` or `actor_fsq_obs`
- those names are local to the model and therefore belong in the protocol
- the only remaining requirement is that runtime-state construction must provide both semantic state entries

## 7. Future Direction for Multi-Observation Models

This is the main follow-up point you raised around `obs_manager`.

Today, `infer.py` still constructs observations explicitly and is effectively optimized for a single `policy_obs` stream. The protocol system already supports multiple state-fed inputs, but the observation-production side is not yet equally declarative.

The right direction is:

1. Keep model IO naming in protocol files.
2. Introduce a runtime-state builder layer that produces one or more semantic observation entries.
3. Let the protocol bind those semantic entries to raw model tensor names.

A future abstraction should look conceptually like this:

- `ObservationProducer`
  Produces one named semantic observation stream.
- `RuntimeStateBuilder`
  Collects all required streams and writes them into `RuntimeState.values`.
- `ModelProtocol`
  Binds those semantic keys to raw backend tensor names.

That separation is important:

- observation construction is robot/business logic
- protocol binding is model IO declaration
- backend execution is infrastructure

Do not merge those three concerns back together.

## 8. Recommended Runtime-State Contract

When writing future multi-input inference code, use this rule:

- `RuntimeState.values` should contain semantic resources, not model-specific tensor names unless they are already the clearest semantic name.

Good examples:

- `policy_obs`
- `actor_obs`
- `actor_fsq_obs`
- `command`
- `phase_signal`
- `history_action`

Bad examples:

- `input_0`
- `input_1`
- `net_a_tensor`

The point is not to invent generic names. The point is to keep names semantic and stable at the deployment layer.

## 9. Transform Registry Guidance

`TRANSFORM_REGISTRY` should remain small and generic.

Good candidates:

- reshape helpers
- dtype conversion helpers
- simple scalar arithmetic
- flatten helpers

Bad candidates:

- robot-specific feature extraction
- simulator reads
- protocol-specific hacks for one export

If a transform is only valid for one model family and encodes domain meaning, it is probably the wrong abstraction layer.

## 10. Common Errors

### Error: protocol uses names not present in the model

Symptom:

- backend inference fails because input or output names do not exist

Fix:

- re-check the real exported model signature

### Error: protocol uses one semantic source for two actually different observations

Symptom:

- model runs but behavior is wrong

Fix:

- split runtime-state production into separate semantic entries
- bind them independently in the protocol

### Error: buffer dimension mismatches action dimension

Symptom:

- action history buffers have the wrong shape

Fix:

- use `zeros_from_output` with the true action output tensor and axis

### Error: transform registry grows into business logic

Symptom:

- protocol parsing is technically valid but deployment logic becomes opaque

Fix:

- move business logic back into observation construction or runtime-state building

## 11. Minimal Checklist for Humans or AI

When creating a new `policy.protocol.yaml`, follow this checklist exactly:

1. Inspect the real model signature.
2. Write down every raw input name and output name.
3. Assign a semantic meaning to each tensor.
4. Decide whether each input comes from `state`, `buffer`, `result`, or `constant`.
5. Add only the transforms required for shape and dtype conversion.
6. Define persistent buffers only when the model or rollout logic truly needs them.
7. Define per-step buffer roll-forward rules.
8. Load the protocol with `load_protocol_from_file()`.
9. Run unit or smoke tests against the actual model.
10. Only after this, wire the protocol into `infer.py`.

## 12. Recommended Next Refactor

The next engineering step after protocol externalization should be:

- replace `infer.py::_build_model_protocol()` with loading from `policy.protocol.yaml`

After that, the next step should be:

- separate observation production from `infer.py` into a runtime-state builder that can emit multiple observation streams

That order matters:

- first externalize model IO semantics
- then externalize multi-stream observation construction

This keeps each refactor small, testable, and reversible.
