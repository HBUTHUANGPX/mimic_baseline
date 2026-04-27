from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_discover_hdf5_episode_tasks_uses_subset_episode_layout(tmp_path: Path) -> None:
    from hdf5_parse.motion_export.batch import discover_hdf5_episode_tasks

    (tmp_path / "subset_a" / "ep1").mkdir(parents=True)
    (tmp_path / "subset_a" / "ep1" / "annotation.hdf5").write_bytes(b"fake")
    (tmp_path / "subset_a" / "ep2").mkdir(parents=True)
    (tmp_path / "subset_a" / "ep2" / "annotation.hdf5").write_bytes(b"fake")
    (tmp_path / "subset_b" / "notes").mkdir(parents=True)

    tasks = discover_hdf5_episode_tasks(tmp_path, output_root=tmp_path / "out")

    assert [(task.subset_id, task.episode_id) for task in tasks] == [("subset_a", "ep1"), ("subset_a", "ep2")]
    assert tasks[0].hdf5_path == tmp_path / "subset_a" / "ep1" / "annotation.hdf5"
    assert tasks[0].output_dir == tmp_path / "out" / "subset_a" / "ep1"


def test_export_hdf5_batch_smpl_uses_executor_for_multiple_workers(tmp_path: Path, monkeypatch) -> None:
    from hdf5_parse.motion_export import batch

    task = batch.HDF5EpisodeTask(
        subset_id="subset_a",
        episode_id="ep1",
        hdf5_path=tmp_path / "subset_a" / "ep1" / "annotation.hdf5",
        output_dir=tmp_path / "out" / "subset_a" / "ep1",
    )
    calls = []

    def fake_worker(task, **kwargs):
        calls.append((task.task_id, kwargs["smpl_frame"]))
        return batch.BatchExportResult(task_id=task.task_id, ok=True, outputs=(task.output_dir / "smpl" / "x.npz",))

    class RecordingExecutor:
        max_workers_seen = None

        def __init__(self, max_workers):
            type(self).max_workers_seen = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, items):
            return [func(item) for item in items]

    monkeypatch.setattr(batch, "_export_hdf5_smpl_task", fake_worker)

    results = batch.export_hdf5_batch_smpl(
        [task],
        workers=3,
        smpl_frame="raw",
        executor_cls=RecordingExecutor,
    )

    assert RecordingExecutor.max_workers_seen == 3
    assert calls == [("subset_a/ep1", "raw")]
    assert results[0].ok is True


def test_export_hdf5_batch_soma_bvh_is_sequential(tmp_path: Path, monkeypatch) -> None:
    from hdf5_parse.motion_export import batch

    tasks = [
        batch.HDF5EpisodeTask("subset_a", "ep1", tmp_path / "a.hdf5", tmp_path / "out" / "a"),
        batch.HDF5EpisodeTask("subset_b", "ep2", tmp_path / "b.hdf5", tmp_path / "out" / "b"),
    ]
    calls = []

    def fake_worker(task, **kwargs):
        calls.append(task.task_id)
        return batch.BatchExportResult(task_id=task.task_id, ok=True, outputs=(task.output_dir / "soma_bvh" / "x.bvh",))

    monkeypatch.setattr(batch, "_export_hdf5_soma_bvh_task", fake_worker)

    results = batch.export_hdf5_batch_soma_bvh(tasks, workers=99)

    assert calls == ["subset_a/ep1", "subset_b/ep2"]
    assert [result.task_id for result in results] == calls


def test_hdf5_batch_cli_help_runs_without_pythonpath() -> None:
    result = subprocess.run(
        [sys.executable, "hdf5_parse/scripts/batch_export_hdf5_motion.py", "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--test-data-root" in result.stdout
    assert "--exports" in result.stdout
    assert "--workers" in result.stdout
