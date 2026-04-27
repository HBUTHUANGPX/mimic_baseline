from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Iterable

from tqdm.auto import tqdm

from .core import export_hdf5_to_soma_payload, save_hdf5_soma_payload
from .segmented import (
    DEFAULT_FILENAME_PREFIX,
    export_segmented_smpl_npz,
    export_segmented_soma_bvh,
)


DEFAULT_TEST_DATA_ROOT = Path("hdf5_parse/test_data")
DEFAULT_BATCH_OUTPUT_ROOT = Path("hdf5_parse/out/batch")


@dataclass(frozen=True)
class HDF5EpisodeTask:
    subset_id: str
    episode_id: str
    hdf5_path: Path
    output_dir: Path

    @property
    def task_id(self) -> str:
        return f"{self.subset_id}/{self.episode_id}"


@dataclass(frozen=True)
class BatchExportResult:
    task_id: str
    ok: bool
    outputs: tuple[Path, ...] = ()
    error: str = ""


def discover_hdf5_episode_tasks(
    test_data_root: str | Path = DEFAULT_TEST_DATA_ROOT,
    *,
    output_root: str | Path = DEFAULT_BATCH_OUTPUT_ROOT,
) -> list[HDF5EpisodeTask]:
    test_data_root = Path(test_data_root)
    output_root = Path(output_root)
    tasks: list[HDF5EpisodeTask] = []
    for hdf5_path in sorted(test_data_root.glob("*/*/annotation.hdf5")):
        episode_dir = hdf5_path.parent
        subset_dir = episode_dir.parent
        tasks.append(
            HDF5EpisodeTask(
                subset_id=subset_dir.name,
                episode_id=episode_dir.name,
                hdf5_path=hdf5_path,
                output_dir=output_root / subset_dir.name / episode_dir.name,
            )
        )
    return tasks


def _should_skip_dir(path: Path, pattern: str, *, skip_existing: bool) -> bool:
    return bool(skip_existing and path.is_dir() and any(path.glob(pattern)))


def _export_hdf5_annotation_task(
    task: HDF5EpisodeTask,
    *,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    skip_existing: bool = False,
) -> BatchExportResult:
    output_path = task.output_dir / "annotation_soma.npz"
    if skip_existing and output_path.is_file():
        return BatchExportResult(task_id=task.task_id, ok=True, outputs=(output_path,))
    try:
        payload = export_hdf5_to_soma_payload(
            hdf5_path=task.hdf5_path,
            start_frame=start_frame,
            end_frame=end_frame,
            stride=stride,
        )
        return BatchExportResult(
            task_id=task.task_id,
            ok=True,
            outputs=(save_hdf5_soma_payload(payload, output_path),),
        )
    except Exception as exc:  # pragma: no cover - exercised by real batch runs.
        return BatchExportResult(task_id=task.task_id, ok=False, error=repr(exc))


def _export_hdf5_smpl_task(
    task: HDF5EpisodeTask,
    *,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    filename_prefix: str = DEFAULT_FILENAME_PREFIX,
    smpl_frame: str = "soma_y_up",
    skip_existing: bool = False,
) -> BatchExportResult:
    output_dir = task.output_dir / "smpl"
    if _should_skip_dir(output_dir, "*.npz", skip_existing=skip_existing):
        return BatchExportResult(task_id=task.task_id, ok=True, outputs=tuple(sorted(output_dir.glob("*.npz"))))
    try:
        outputs = export_segmented_smpl_npz(
            hdf5_path=task.hdf5_path,
            smpl_output_dir=output_dir,
            start_frame=start_frame,
            end_frame=end_frame,
            stride=stride,
            filename_prefix=filename_prefix,
            smpl_frame=smpl_frame,
        )
        return BatchExportResult(task_id=task.task_id, ok=True, outputs=tuple(outputs))
    except Exception as exc:  # pragma: no cover - exercised by real batch runs.
        return BatchExportResult(task_id=task.task_id, ok=False, error=repr(exc))


def _export_hdf5_soma_bvh_task(
    task: HDF5EpisodeTask,
    *,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    device: str = "cuda",
    batch_size: int | None = None,
    soma_x_root: str | Path = "/home/hpx/HPX_LOCO_2/SOMA-X",
    smpl_model_path: str | Path | None = None,
    filename_prefix: str = DEFAULT_FILENAME_PREFIX,
    skip_existing: bool = False,
) -> BatchExportResult:
    output_dir = task.output_dir / "soma_bvh"
    if _should_skip_dir(output_dir, "*.bvh", skip_existing=skip_existing):
        return BatchExportResult(task_id=task.task_id, ok=True, outputs=tuple(sorted(output_dir.glob("*.bvh"))))
    try:
        outputs = export_segmented_soma_bvh(
            hdf5_path=task.hdf5_path,
            soma_bvh_output_dir=output_dir,
            start_frame=start_frame,
            end_frame=end_frame,
            stride=stride,
            device=device,
            batch_size=batch_size,
            soma_x_root=soma_x_root,
            smpl_model_path=smpl_model_path,
            filename_prefix=filename_prefix,
        )
        return BatchExportResult(task_id=task.task_id, ok=True, outputs=tuple(outputs))
    except Exception as exc:  # pragma: no cover - exercised by real batch runs.
        return BatchExportResult(task_id=task.task_id, ok=False, error=repr(exc))


def _run_tasks_multiprocess(
    tasks: Iterable[HDF5EpisodeTask],
    *,
    worker: Callable[[HDF5EpisodeTask], BatchExportResult],
    workers: int,
    executor_cls=ProcessPoolExecutor,
    desc: str,
) -> list[BatchExportResult]:
    task_list = list(tasks)
    if int(workers) <= 1:
        return [worker(task) for task in tqdm(task_list, desc=desc, unit="file", dynamic_ncols=True)]
    with executor_cls(max_workers=int(workers)) as executor:
        return list(tqdm(executor.map(worker, task_list), total=len(task_list), desc=desc, unit="file", dynamic_ncols=True))


def export_hdf5_batch_annotation(
    tasks: Iterable[HDF5EpisodeTask],
    *,
    workers: int = 1,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    skip_existing: bool = False,
    executor_cls=ProcessPoolExecutor,
) -> list[BatchExportResult]:
    worker = partial(
        _export_hdf5_annotation_task,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
        skip_existing=skip_existing,
    )
    return _run_tasks_multiprocess(tasks, worker=worker, workers=workers, executor_cls=executor_cls, desc="HDF5 annotation")


def export_hdf5_batch_smpl(
    tasks: Iterable[HDF5EpisodeTask],
    *,
    workers: int = 1,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    filename_prefix: str = DEFAULT_FILENAME_PREFIX,
    smpl_frame: str = "soma_y_up",
    skip_existing: bool = False,
    executor_cls=ProcessPoolExecutor,
) -> list[BatchExportResult]:
    worker = partial(
        _export_hdf5_smpl_task,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
        filename_prefix=filename_prefix,
        smpl_frame=smpl_frame,
        skip_existing=skip_existing,
    )
    return _run_tasks_multiprocess(tasks, worker=worker, workers=workers, executor_cls=executor_cls, desc="HDF5 SMPL")


def export_hdf5_batch_soma_bvh(
    tasks: Iterable[HDF5EpisodeTask],
    *,
    workers: int = 1,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    device: str = "cuda",
    batch_size: int | None = None,
    soma_x_root: str | Path = "/home/hpx/HPX_LOCO_2/SOMA-X",
    smpl_model_path: str | Path | None = None,
    filename_prefix: str = DEFAULT_FILENAME_PREFIX,
    skip_existing: bool = False,
) -> list[BatchExportResult]:
    del workers
    results: list[BatchExportResult] = []
    task_list = list(tasks)
    for task in tqdm(task_list, desc="HDF5 SOMA BVH", unit="file", dynamic_ncols=True):
        results.append(
            _export_hdf5_soma_bvh_task(
                task,
                start_frame=start_frame,
                end_frame=end_frame,
                stride=stride,
                device=device,
                batch_size=batch_size,
                soma_x_root=soma_x_root,
                smpl_model_path=smpl_model_path,
                filename_prefix=filename_prefix,
                skip_existing=skip_existing,
            )
        )
    return results
