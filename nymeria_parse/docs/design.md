# Nymeria Motion Export Design

## Goal

Build `nymeria_parse` as a clean sibling of `hdf5_parse`, using the same package shape and export semantics where possible. The first supported sequence is:

`nymeria_parse/test_data/20230607_s0_james_johnson_act0_e72nhq`

The source body motion is the Xsens MVNX file:

`nymeria_parse/test_data/20230607_s0_james_johnson_act0_e72nhq/body_xdata_mvnx`

The target outputs are:

- `nymeria_parse/out/annotation.npz`
- `nymeria_parse/out/smpl/*.npz`
- `nymeria_parse/out/soma_bvh/*.bvh`

The implementation should stay structurally close to `hdf5_parse` so the two projects can later be merged with minimal reshaping.

## Source Data

The current Nymeria test data does not contain official SMPL or MHR files:

- Missing: `body/xdata_smpl_neutral.npz`
- Missing: `body/xdata_mhr.glb`

The available body source for this pipeline is the raw Xsens MVNX XML. The processed `body/xdata.npz` can be used for validation, but the export pipeline should read `body_xdata_mvnx` first.

The MVNX data contains per-frame Xsens segment poses and positions. These become the source of the SMPL pose estimate.

## Directory Layout

Mirror the current `hdf5_parse` structure:

```text
nymeria_parse/
├── docs/
├── motion_export/
├── scripts/
├── utils/
├── test_data/
└── out/
    ├── annotation.npz
    ├── smpl/
    └── soma_bvh/
```

Responsibilities:

- `scripts/`: command-line entry points only.
- `utils/`: MVNX parsing, Xsens constants/mapping, rotation utilities.
- `motion_export/`: reusable export logic.
- `docs/`: user-facing notes and implementation design.

## Annotation Export

`nymeria_parse/out/annotation.npz` stores timeline and text metadata only. It must not contain human skeleton or SMPL arrays.

Required fields:

- `fps`
- `num_frames`
- `timeline_frame_indices`
- `frame_timestamps`
- `main_task_texts`
- `main_task_text_indices`
- `sub_task_texts`
- `sub_task_text_indices`
- `current_action_texts`
- `current_action_text_indices`
- `interaction_texts`
- `interaction_text_indices`

Text mapping:

- `Main Task`: no reliable source in the current test data; use `UNKNOWN`.
- `Sub Task`: use `narration/activity_summarization.csv`, column `Describe my activity`.
- `Current Action`: use `narration/atomic_action.csv`, column `Describe my atomic actions`.
- `Interaction`: no reliable source in the current test data; use `UNKNOWN`.

All four text pools should put `UNKNOWN` at index `0`. Per-frame index arrays should be aligned to the MVNX frame timeline.

## Timestamp Alignment

This part needs careful validation.

The MVNX frames define the body motion timeline. Narration CSV rows use `start_time` and `end_time` values. The export must determine and document how those CSV times map to MVNX frame timestamps.

Validation requirements:

- Parse MVNX frame timestamps or derive frame times from MVNX metadata.
- Compare the MVNX time range with both narration CSV time ranges.
- If the timelines overlap, align text by interval membership.
- If the timelines do not overlap, leave the relevant text indices as `UNKNOWN` and emit a clear warning.
- Save `frame_timestamps` in the same time unit used for text alignment, and document that unit.

The first implementation should include a diagnostic summary in logs:

- MVNX first/last timestamp.
- Activity CSV first/last time.
- Atomic action CSV first/last time.
- Number and percentage of frames covered by each text source.

## SMPL Export

SMPL output should be as standard as possible and avoid extra project-specific aliases.

Required fields:

- `global_orient`: `(F, 3)`, axis-angle.
- `body_pose`: `(F, 69)`, axis-angle for the 23 non-root SMPL joints.
- `transl`: `(F, 3)`.
- `betas`: `(F, 10)`.

Missing semantic values:

- `betas` has no source in MVNX; fill with zeros.
- Any SMPL joint not represented by Xsens mapping should use identity rotation.

Do not add duplicated names such as `smpl_body_pose`, timeline metadata, or debug-only pose arrays in the standard SMPL output file. Timeline metadata belongs in `annotation.npz`. Debug outputs can be added later as a separate diagnostic artifact if needed.

## Xsens To SMPL

Use `nymeria_smpl_processor` as reference code, not as a runtime dependency.

First-pass mapping:

- Parse MVNX per-frame Xsens segment quaternion and root position.
- Convert Xsens coordinates into the SMPL coordinate convention used by the reference processor.
- Map 23 Xsens segments to 24 SMPL joints with the reference `XSENS_TO_SMPL` mapping.
- Convert mapped global rotations to local SMPL rotations.
- Convert rotation matrices to axis-angle.

The SMPL model path should follow the same default convention used by `hdf5_parse` and remain configurable from CLI.

## SOMA BVH Export

SOMA BVH export should reuse the semantics already calibrated in `hdf5_parse`:

- Run SOMA-X from SMPL motion to SOMA local transforms.
- Write BVH with `Root` 6 channels all zero.
- Let `Hips` carry the body root motion.
- Keep the BVH readable by `soma-retargeter/app/bvh_to_csv_converter.py`.

Output directory:

`nymeria_parse/out/soma_bvh`

The first implementation should support `--start-frame` and `--end-frame`. The recommended short validation range is:

`--start-frame 0 --end-frame 1000`

Full export should be possible by explicitly passing `--end-frame -1`.

Full-sequence export must not send every frame through CUDA at once. The SOMA-X inversion path should run in chunks, defaulting to `--batch-size 256`, and keep only the current chunk's SMPL tensors, SOMA rotations, and skinning transforms on GPU. Chunk outputs are moved back to CPU before writing BVH. If memory is still insufficient on a target GPU, reduce `--batch-size`.

## CLI Design

Initial scripts:

- `nymeria_parse/scripts/export_nymeria_annotation.py`
- `nymeria_parse/scripts/export_nymeria_to_smpl.py`
- `nymeria_parse/scripts/export_nymeria_to_soma_bvh.py`

Shared defaults:

- Input sequence: `nymeria_parse/test_data/20230607_s0_james_johnson_act0_e72nhq`
- MVNX path: `<sequence>/body_xdata_mvnx`
- Output root: `nymeria_parse/out`
- Device: `cuda`
- `--start-frame 0`
- `--end-frame 1000`
- `--batch-size 256` for SOMA BVH export

Long offline loops should expose progress with `tqdm`. Newton-based applications should keep the standard `newton.examples.create_parser()` setup and default `quiet=True` when running batch/offline conversion, leaving progress bars and error messages readable.

## Validation Plan

Unit tests:

- MVNX parser returns expected frame count, fps, segment count, and first-frame arrays.
- Text alignment handles overlap and non-overlap cases.
- SMPL output contains exactly the standard fields plus metadata fields.
- `betas` is zero-filled.
- `annotation.npz` has text pools and index arrays with length equal to selected frames.

Smoke tests:

- Export annotation for frames `0..1000`.
- Export SMPL for frames `0..1000`.
- Export SOMA BVH for frames `0..1000`.
- Confirm `Root` motion channels in BVH are zero.
- Confirm `Hips` position channels are nonzero when motion exists.

Manual checks:

- Compare MVNX-derived segment positions with `body/xdata.npz` for a short range.
- Visualize the resulting SOMA BVH via the existing `soma-retargeter` flow.

## Open Risks

The largest risk is not file I/O, but semantic alignment:

- MVNX timestamp units and narration CSV time units may be offset or expressed in different domains.
- Xsens-to-SMPL coordinate conversion may need visual or numerical validation.
- Xsens has 23 segments while SMPL has 24 joints; unmapped joints must be identity rotations.
- This is a skeleton retargeting route, not a full SMPL body shape fitting route.

These risks should be exposed through logs, tests, and short-range visual diagnostics before full-length export.
