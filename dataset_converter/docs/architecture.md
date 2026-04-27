# Architecture

`dataset_converter` is the new package-level API. The legacy folders remain in place while the package is introduced.

## Package Layout

```text
dataset_converter/
├── common/
│   ├── batch.py
│   ├── cli.py
│   └── paths.py
├── hdf5/
│   ├── batch.py
│   └── cli/
└── nymeria/
    ├── batch.py
    └── cli/
```

## Shared Semantics

`dataset_converter.common.paths` owns path resolution:

- repository-relative defaults for test data and output roots;
- environment variables for external assets;
- no machine-specific absolute paths in the new package.

`dataset_converter.common.batch` owns the shared batch result model and execution helpers:

- `BatchExportResult`
- multiprocess execution for CPU/IO stages;
- sequential execution for CUDA/SOMA stages.

`dataset_converter.common.cli` owns JSONL summaries and stage printing.

## HDF5 Pipeline

HDF5 tasks are discovered from:

```text
<test-data-root>/<subset_id>/<episode_id>/annotation.hdf5
```

Each task writes under:

```text
<output-root>/<subset_id>/<episode_id>/
```

The package currently calls the stable implementation in `hdf5_parse.motion_export`.

## Nymeria Pipeline

Nymeria tasks are discovered from:

```text
<test-data-root>/<sequence_id>/body_xdata_mvnx
```

Each task writes under:

```text
<output-root>/<sequence_id>/
```

The package currently calls the stable implementation in `nymeria_parse.motion_export`.

## Migration Plan

The compatibility direction is:

1. Keep `hdf5_parse` and `nymeria_parse` working.
2. Add new code through `dataset_converter`.
3. Move shared behavior into `dataset_converter.common`.
4. Gradually migrate old scripts to thin wrappers or retire them once downstream users move to the new package.
