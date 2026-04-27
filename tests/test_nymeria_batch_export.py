from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_discover_nymeria_sequence_tasks_uses_sequence_layout(tmp_path: Path) -> None:
    from nymeria_parse.motion_export.batch import discover_nymeria_sequence_tasks

    (tmp_path / "seq_a").mkdir(parents=True)
    (tmp_path / "seq_a" / "body_xdata_mvnx").write_bytes(b"fake")
    (tmp_path / "seq_b").mkdir(parents=True)
    (tmp_path / "seq_b" / "body_xdata_mvnx").write_bytes(b"fake")
    (tmp_path / "not_a_sequence").mkdir(parents=True)

    tasks = discover_nymeria_sequence_tasks(tmp_path, output_root=tmp_path / "out")

    assert [task.sequence_id for task in tasks] == ["seq_a", "seq_b"]
    assert tasks[0].sequence_dir == tmp_path / "seq_a"
    assert tasks[0].output_dir == tmp_path / "out" / "seq_a"


def test_export_nymeria_batch_smpl_uses_executor_for_multiple_workers(tmp_path: Path, monkeypatch) -> None:
    from nymeria_parse.motion_export import batch

    task = batch.NymeriaSequenceTask("seq_a", tmp_path / "seq_a", tmp_path / "out" / "seq_a")
    calls = []

    def fake_worker(task, **kwargs):
        calls.append((task.task_id, kwargs["end_frame"]))
        return batch.BatchExportResult(task_id=task.task_id, ok=True, outputs=(task.output_dir / "smpl" / "nymeria_smpl.npz",))

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

    monkeypatch.setattr(batch, "_export_nymeria_smpl_task", fake_worker)

    results = batch.export_nymeria_batch_smpl(
        [task],
        workers=4,
        end_frame=-1,
        executor_cls=RecordingExecutor,
    )

    assert RecordingExecutor.max_workers_seen == 4
    assert calls == [("seq_a", -1)]
    assert results[0].ok is True


def test_export_nymeria_batch_soma_bvh_is_sequential(tmp_path: Path, monkeypatch) -> None:
    from nymeria_parse.motion_export import batch

    tasks = [
        batch.NymeriaSequenceTask("seq_a", tmp_path / "seq_a", tmp_path / "out" / "seq_a"),
        batch.NymeriaSequenceTask("seq_b", tmp_path / "seq_b", tmp_path / "out" / "seq_b"),
    ]
    calls = []

    def fake_worker(task, **kwargs):
        calls.append(task.task_id)
        return batch.BatchExportResult(task_id=task.task_id, ok=True, outputs=(task.output_dir / "soma_bvh" / "nymeria_soma.bvh",))

    monkeypatch.setattr(batch, "_export_nymeria_soma_bvh_task", fake_worker)

    results = batch.export_nymeria_batch_soma_bvh(tasks, workers=99)

    assert calls == ["seq_a", "seq_b"]
    assert [result.task_id for result in results] == calls


def test_nymeria_batch_cli_help_runs_without_pythonpath() -> None:
    result = subprocess.run(
        [sys.executable, "nymeria_parse/scripts/batch_export_nymeria_motion.py", "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--test-data-root" in result.stdout
    assert "--exports" in result.stdout
    assert "--workers" in result.stdout
