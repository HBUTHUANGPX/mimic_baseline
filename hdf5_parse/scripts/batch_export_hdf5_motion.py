from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _bootstrap import ensure_repo_root_on_sys_path

ensure_repo_root_on_sys_path(__file__)

from hdf5_parse.motion_export.batch import (  # noqa: E402
    DEFAULT_BATCH_OUTPUT_ROOT,
    DEFAULT_TEST_DATA_ROOT,
    BatchExportResult,
    discover_hdf5_episode_tasks,
    export_hdf5_batch_annotation,
    export_hdf5_batch_smpl,
    export_hdf5_batch_soma_bvh,
)


EXPORT_CHOICES = ("annotation", "smpl", "soma-bvh")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch export HDF5 test_data motion assets. SMPL/annotation can use multiprocessing; SOMA BVH is sequential."
    )
    parser.add_argument("--test-data-root", type=Path, default=DEFAULT_TEST_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_BATCH_OUTPUT_ROOT)
    parser.add_argument("--exports", nargs="+", choices=EXPORT_CHOICES, default=["annotation", "smpl"])
    parser.add_argument("--workers", type=int, default=1, help="Process count for annotation/SMPL exports. SOMA BVH ignores this.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--filename-prefix", default="annotation")
    parser.add_argument("--smpl-frame", choices=("soma_y_up", "raw"), default="soma_y_up")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--soma-x-root", type=Path, default=Path("/home/hpx/HPX_LOCO_2/SOMA-X"))
    parser.add_argument("--smpl-model-path", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _result_to_json(stage: str, result: BatchExportResult) -> str:
    return json.dumps(
        {
            "stage": stage,
            "task_id": result.task_id,
            "ok": result.ok,
            "outputs": [str(path) for path in result.outputs],
            "error": result.error,
        },
        ensure_ascii=False,
    )


def _write_summary(summary_path: Path | None, rows: list[str]) -> None:
    if summary_path is None:
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _print_stage(stage: str, results: list[BatchExportResult]) -> None:
    ok_count = sum(result.ok for result in results)
    print(f"{stage}: {ok_count}/{len(results)} tasks ok")
    for result in results:
        if not result.ok:
            print(f"[FAILED] {result.task_id}: {result.error}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    tasks = discover_hdf5_episode_tasks(args.test_data_root, output_root=args.output_root)
    print(f"Discovered {len(tasks)} HDF5 episode tasks under {args.test_data_root}")

    summary_rows: list[str] = []
    exit_code = 0

    if "annotation" in args.exports:
        results = export_hdf5_batch_annotation(
            tasks,
            workers=args.workers,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            stride=args.stride,
            skip_existing=args.skip_existing,
        )
        _print_stage("annotation", results)
        summary_rows.extend(_result_to_json("annotation", result) for result in results)
        if any(not result.ok for result in results):
            exit_code = 1
            if args.fail_fast:
                _write_summary(args.summary_path, summary_rows)
                return exit_code

    if "smpl" in args.exports:
        results = export_hdf5_batch_smpl(
            tasks,
            workers=args.workers,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            stride=args.stride,
            filename_prefix=args.filename_prefix,
            smpl_frame=args.smpl_frame,
            skip_existing=args.skip_existing,
        )
        _print_stage("smpl", results)
        summary_rows.extend(_result_to_json("smpl", result) for result in results)
        if any(not result.ok for result in results):
            exit_code = 1
            if args.fail_fast:
                _write_summary(args.summary_path, summary_rows)
                return exit_code

    if "soma-bvh" in args.exports:
        results = export_hdf5_batch_soma_bvh(
            tasks,
            workers=1,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            stride=args.stride,
            device=args.device,
            batch_size=args.batch_size,
            soma_x_root=args.soma_x_root,
            smpl_model_path=args.smpl_model_path,
            filename_prefix=args.filename_prefix,
            skip_existing=args.skip_existing,
        )
        _print_stage("soma-bvh", results)
        summary_rows.extend(_result_to_json("soma-bvh", result) for result in results)
        if any(not result.ok for result in results):
            exit_code = 1

    _write_summary(args.summary_path, summary_rows)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
