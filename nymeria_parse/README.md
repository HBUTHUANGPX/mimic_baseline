# nymeria_parse

Utilities for converting Nymeria MVNX body motion into project-compatible motion assets.

The current pipeline uses:

`nymeria_parse/test_data/20230607_s0_james_johnson_act0_e72nhq/body_xdata_mvnx`

and exports:

- `nymeria_parse/out/annotation.npz`: timeline and text metadata only.
- `nymeria_parse/out/smpl/nymeria_smpl.npz`: standard SMPL motion arrays.
- `nymeria_parse/out/soma_bvh/nymeria_soma.bvh`: SOMA BVH for downstream retargeting.

See [docs/design.md](/home/hpx/HPX_LOCO_2/mimic_baseline/nymeria_parse/docs/design.md) for the data semantics and alignment decisions.

## Commands

Export annotation/text metadata:

```bash
python nymeria_parse/scripts/export_nymeria_annotation.py \
  --start-frame 0 \
  --end-frame 1000
```

Export SMPL motion:

```bash
python nymeria_parse/scripts/export_nymeria_to_smpl.py \
  --start-frame 0 \
  --end-frame 1000
```

Export SOMA BVH:

```bash
python nymeria_parse/scripts/export_nymeria_to_soma_bvh.py \
  --smpl-model-path /home/hpx/HPX_LOCO_2/SOMA-X/assets/SMPL/SMPL_NEUTRAL.npz \
  --start-frame 0 \
  --end-frame 1000 \
  --batch-size 256
```

Use `--end-frame -1` for full-sequence export. SOMA-X inversion is CUDA-only and can be memory-heavy, so the exporter defaults to chunked processing with `--batch-size 256`. If the full sequence still runs out of memory, lower this value, for example `--batch-size 64` or `--batch-size 32`.

## Notes

`activity_summarization.csv` is exported as `Sub Task`, and `atomic_action.csv` is exported as `Current Action`. `Main Task` and `Interaction` currently have no reliable source in this test sequence and are filled with `UNKNOWN`.

SMPL output intentionally stores only standard motion fields plus minimal metadata:

- `global_orient`
- `body_pose`
- `transl`
- `betas`

Timeline metadata is kept in `annotation.npz`, not duplicated into the SMPL file.

Long-running export loops use `tqdm` progress bars. Newton-based batch applications should keep `newton.examples.create_parser()` and default to `quiet=True` for large offline processing, so Warp/Newton logs do not drown out progress and errors.
