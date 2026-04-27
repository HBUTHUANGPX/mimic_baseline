from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "dataset_converter"


def test_dataset_converter_imports_common_hdf5_and_nymeria_modules() -> None:
    import dataset_converter
    from dataset_converter.common.paths import get_repo_root
    from dataset_converter.hdf5.batch import discover_hdf5_episode_tasks
    from dataset_converter.nymeria.batch import discover_nymeria_sequence_tasks

    assert dataset_converter.__version__
    assert get_repo_root() == REPO_ROOT
    assert callable(discover_hdf5_episode_tasks)
    assert callable(discover_nymeria_sequence_tasks)


def test_path_resolver_uses_environment_before_candidates(tmp_path: Path, monkeypatch) -> None:
    from dataset_converter.common.paths import resolve_optional_path

    env_path = tmp_path / "external_soma"
    env_path.mkdir()
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    monkeypatch.setenv("SOMA_X_ROOT", str(env_path))

    resolved = resolve_optional_path(
        explicit=None,
        env_var="SOMA_X_ROOT",
        candidates=[fallback],
    )

    assert resolved == env_path.resolve()


def test_hdf5_and_nymeria_discovery_share_batch_task_shape(tmp_path: Path) -> None:
    from dataset_converter.hdf5.batch import discover_hdf5_episode_tasks
    from dataset_converter.nymeria.batch import discover_nymeria_sequence_tasks

    (tmp_path / "hdf5" / "subset" / "ep1").mkdir(parents=True)
    (tmp_path / "hdf5" / "subset" / "ep1" / "annotation.hdf5").write_bytes(b"fake")
    (tmp_path / "nymeria" / "seq").mkdir(parents=True)
    (tmp_path / "nymeria" / "seq" / "body_xdata_mvnx").write_bytes(b"fake")

    hdf5_task = discover_hdf5_episode_tasks(tmp_path / "hdf5", output_root=tmp_path / "out" / "hdf5")[0]
    nymeria_task = discover_nymeria_sequence_tasks(tmp_path / "nymeria", output_root=tmp_path / "out" / "nymeria")[0]

    assert hdf5_task.task_id == "subset/ep1"
    assert nymeria_task.task_id == "seq"
    assert hdf5_task.output_dir == tmp_path / "out" / "hdf5" / "subset" / "ep1"
    assert nymeria_task.output_dir == tmp_path / "out" / "nymeria" / "seq"


def test_dataset_converter_module_cli_help_runs_without_pythonpath() -> None:
    commands = [
        [sys.executable, "-m", "dataset_converter.hdf5.cli.batch_export", "--help"],
        [sys.executable, "-m", "dataset_converter.nymeria.cli.batch_export", "--help"],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "--test-data-root" in result.stdout
        assert "--exports" in result.stdout


def test_pyproject_exposes_dataset_converter_console_scripts() -> None:
    assert not (REPO_ROOT / "pyproject.toml").exists()
    pyproject = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "dataset-converter"
    scripts = pyproject["project"]["scripts"]
    assert scripts["dataset-converter-hdf5-batch"] == "dataset_converter.hdf5.cli.batch_export:main"
    assert scripts["dataset-converter-nymeria-batch"] == "dataset_converter.nymeria.cli.batch_export:main"


def test_dataset_converter_source_does_not_lock_to_local_home_paths() -> None:
    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((PACKAGE_ROOT / "src" / "dataset_converter").rglob("*.py"))
    )

    assert "/home/hpx" not in source_text
    assert "HPX_LOCO_2" not in source_text


def test_dataset_converter_annotation_and_smpl_stages_do_not_import_legacy_packages() -> None:
    source_paths = sorted((PACKAGE_ROOT / "src" / "dataset_converter").rglob("*.py"))
    legacy_import_lines = []
    for path in source_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "hdf5_parse." in line or "nymeria_parse." in line:
                legacy_import_lines.append((path.relative_to(PACKAGE_ROOT), line.strip()))

    assert legacy_import_lines == [
        (
            Path("src/dataset_converter/hdf5/batch.py"),
            "from hdf5_parse.motion_export.segmented import export_segmented_soma_bvh",
        ),
        (
            Path("src/dataset_converter/nymeria/batch.py"),
            "from nymeria_parse.motion_export.soma_bvh import export_nymeria_to_soma_bvh",
        ),
    ]
