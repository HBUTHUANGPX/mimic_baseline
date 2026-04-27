from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Iterable

from tqdm.auto import tqdm

from .core import build_annotation_payload, save_annotation_payload
from .smpl import build_smpl_motion_payload, save_smpl_motion_npz
from .soma_bvh import DEFAULT_SOMA_BATCH_SIZE, export_nymeria_to_soma_bvh


DEFAULT_TEST_DATA_ROOT = Path("nymeria_parse/test_data")
DEFAULT_BATCH_OUTPUT_ROOT = Path("nymeria_parse/out/batch")


@dataclass(frozen=True)
class NymeriaSequenceTask:
    sequence_id: str
    sequence_dir: Path
    output_dir: Path

    @property
    def task_id(self) -> str:
        return self.sequence_id


@dataclass(frozen=True)
class BatchExportResult:
    task_id: str
    ok: bool
    outputs: tuple[Path, ...] = ()
    error: str = ""


def discover_nymeria_sequence_tasks(
    test_data_root: str | Path = DEFAULT_TEST_DATA_ROOT,
    *,
    output_root: str | Path = DEFAULT_BATCH_OUTPUT_ROOT,
) -> list[NymeriaSequenceTask]:
    test_data_root = Path(test_data_root)
    output_root = Path(output_root)
    tasks: list[NymeriaSequenceTask] = []
    for mvnx_path in sorted(test_data_root.glob("*/body_xdata_mvnx")):
        sequence_dir = mvnx_path.parent
        tasks.append(NymeriaSequenceTask(sequence_dir.name, sequence_dir, output_root / sequence_dir.name))
    return tasks


def _export_nymeria_annotation_task(
    task: NymeriaSequenceTask,
    *,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    skip_existing: bool = False,
) -> BatchExportResult:
    output_path = task.output_dir / "annotation.npz"
    if skip_existing and output_path.is_file():
        return BatchExportResult(task_id=task.task_id, ok=True, outputs=(output_path,))
    try:
        payload, _ = build_annotation_payload(
            task.sequence_dir,
            start_frame=start_frame,
            end_frame=end_frame,
            stride=stride,
        )
        return BatchExportResult(task_id=task.task_id, ok=True, outputs=(save_annotation_payload(payload, output_path),))
    except Exception as exc:  # pragma: no cover - exercised by real batch runs.
        return BatchExportResult(task_id=task.task_id, ok=False, error=repr(exc))


def _export_nymeria_smpl_task(
    task: NymeriaSequenceTask,
    *,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    skip_existing: bool = False,
) -> BatchExportResult:
    output_path = task.output_dir / "smpl" / "nymeria_smpl.npz"
    if skip_existing and output_path.is_file():
        return BatchExportResult(task_id=task.task_id, ok=True, outputs=(output_path,))
    try:
        payload = build_smpl_motion_payload(
            task.sequence_dir,
            start_frame=start_frame,
            end_frame=end_frame,
            stride=stride,
        )
        return BatchExportResult(task_id=task.task_id, ok=True, outputs=(save_smpl_motion_npz(payload, output_path),))
    except Exception as exc:  # pragma: no cover - exercised by real batch runs.
        return BatchExportResult(task_id=task.task_id, ok=False, error=repr(exc))


def _export_nymeria_soma_bvh_task(
    task: NymeriaSequenceTask,
    *,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    device: str = "cuda",
    batch_size: int | None = DEFAULT_SOMA_BATCH_SIZE,
    soma_x_root: str | Path = "/home/hpx/HPX_LOCO_2/SOMA-X",
    smpl_model_path: str | Path | None = None,
    skip_existing: bool = False,
) -> BatchExportResult:
    output_path = task.output_dir / "soma_bvh" / "nymeria_soma.bvh"
    if skip_existing and output_path.is_file():
        return BatchExportResult(task_id=task.task_id, ok=True, outputs=(output_path,))
    try:
        return BatchExportResult(
            task_id=task.task_id,
            ok=True,
            outputs=(
                export_nymeria_to_soma_bvh(
                    task.sequence_dir,
                    output_path=output_path,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    stride=stride,
                    device=device,
                    batch_size=batch_size,
                    soma_x_root=soma_x_root,
                    smpl_model_path=smpl_model_path,
                ),
            ),
        )
    except Exception as exc:  # pragma: no cover - exercised by real batch runs.
        return BatchExportResult(task_id=task.task_id, ok=False, error=repr(exc))


def _run_tasks_multiprocess(
    tasks: Iterable[NymeriaSequenceTask],
    *,
    worker: Callable[[NymeriaSequenceTask], BatchExportResult],
    workers: int,
    executor_cls=ProcessPoolExecutor,
    desc: str,
) -> list[BatchExportResult]:
    task_list = list(tasks)
    if int(workers) <= 1:
        return [worker(task) for task in tqdm(task_list, desc=desc, unit="file", dynamic_ncols=True)]
    with executor_cls(max_workers=int(workers)) as executor:
        return list(tqdm(executor.map(worker, task_list), total=len(task_list), desc=desc, unit="file", dynamic_ncols=True))


def export_nymeria_batch_annotation(
    tasks: Iterable[NymeriaSequenceTask],
    *,
    workers: int = 1,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    skip_existing: bool = False,
    executor_cls=ProcessPoolExecutor,
) -> list[BatchExportResult]:
    worker = partial(
        _export_nymeria_annotation_task,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
        skip_existing=skip_existing,
    )
    return _run_tasks_multiprocess(tasks, worker=worker, workers=workers, executor_cls=executor_cls, desc="Nymeria annotation")


def export_nymeria_batch_smpl(
    tasks: Iterable[NymeriaSequenceTask],
    *,
    workers: int = 1,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    skip_existing: bool = False,
    executor_cls=ProcessPoolExecutor,
) -> list[BatchExportResult]:
    worker = partial(
        _export_nymeria_smpl_task,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
        skip_existing=skip_existing,
    )
    return _run_tasks_multiprocess(tasks, worker=worker, workers=workers, executor_cls=executor_cls, desc="Nymeria SMPL")


def export_nymeria_batch_soma_bvh(
    tasks: Iterable[NymeriaSequenceTask],
    *,
    workers: int = 1,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    device: str = "cuda",
    batch_size: int | None = DEFAULT_SOMA_BATCH_SIZE,
    soma_x_root: str | Path = "/home/hpx/HPX_LOCO_2/SOMA-X",
    smpl_model_path: str | Path | None = None,
    skip_existing: bool = False,
) -> list[BatchExportResult]:
    del workers
    task_list = list(tasks)
    results: list[BatchExportResult] = []
    for task in tqdm(task_list, desc="Nymeria SOMA BVH", unit="file", dynamic_ncols=True):
        results.append(
            _export_nymeria_soma_bvh_task(
                task,
                start_frame=start_frame,
                end_frame=end_frame,
                stride=stride,
                device=device,
                batch_size=batch_size,
                soma_x_root=soma_x_root,
                smpl_model_path=smpl_model_path,
                skip_existing=skip_existing,
            )
        )
    return results
