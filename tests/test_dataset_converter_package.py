from __future__ import annotations

import subprocess
import sys
import tomllib
import configparser
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

    env_path = tmp_path / "external_soma_assets"
    env_path.mkdir()
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    monkeypatch.setenv("SOMA_ASSETS_ROOT", str(env_path))

    resolved = resolve_optional_path(
        explicit=None,
        env_var="SOMA_ASSETS_ROOT",
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

    assert pyproject["build-system"]["build-backend"] == "setuptools.build_meta"
    assert "project" not in pyproject


def test_setup_cfg_declares_package_metadata_entry_points_and_extras() -> None:
    parser = configparser.ConfigParser()
    parser.read(PACKAGE_ROOT / "setup.cfg", encoding="utf-8")

    assert parser["metadata"]["name"] == "dataset-converter"
    assert parser["options"]["python_requires"] == ">=3.11"
    assert "dataset_converter*" in parser["options.packages.find"]["include"]
    assert "soma*" in parser["options.packages.find"]["include"]

    entry_points = parser["options.entry_points"]["console_scripts"]
    assert "dataset-converter-hdf5-batch = dataset_converter.hdf5.cli.batch_export:main" in entry_points
    assert "dataset-converter-nymeria-batch = dataset_converter.nymeria.cli.batch_export:main" in entry_points

    extras = parser["options.extras_require"]
    assert "torch" in extras["soma"]
    assert "smplx" in extras["soma"]
    assert "warp-lang" in extras["soma"]
    assert "trimesh" in extras["soma"]
    assert "torch" in extras["gpu"]
    assert "smplx" in extras["gpu"]
    assert "pytest" in extras["dev"]


def test_requirements_txt_contains_cpu_base_dependencies_only() -> None:
    requirements = (PACKAGE_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()

    assert requirements == ["h5py", "numpy", "scipy", "tqdm"]


def test_vendored_soma_runtime_imports_from_dataset_converter_src() -> None:
    env = {"PYTHONPATH": str(PACKAGE_ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import soma; from soma.soma import SOMALayer; from soma.pose_inversion import PoseInversion; "
            "print(soma.__file__); print(SOMALayer.__name__, PoseInversion.__name__)",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert str(PACKAGE_ROOT / "src" / "soma") in result.stdout
    assert "SOMALayer PoseInversion" in result.stdout


def test_dataset_converter_source_does_not_lock_to_local_home_paths() -> None:
    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((PACKAGE_ROOT / "src" / "dataset_converter").rglob("*.py"))
    )

    assert "/home/hpx" not in source_text
    assert "HPX_LOCO_2" not in source_text


def test_dataset_converter_does_not_import_legacy_parse_packages() -> None:
    source_paths = sorted((PACKAGE_ROOT / "src" / "dataset_converter").rglob("*.py"))
    legacy_import_lines = []
    for path in source_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "hdf5_parse." in line or "nymeria_parse." in line:
                legacy_import_lines.append((path.relative_to(PACKAGE_ROOT), line.strip()))

    assert legacy_import_lines == []


def test_dataset_converter_soma_cli_uses_assets_root_not_soma_x_root() -> None:
    commands = [
        [sys.executable, "-m", "dataset_converter.hdf5.cli.batch_export", "--help"],
        [sys.executable, "-m", "dataset_converter.nymeria.cli.batch_export", "--help"],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "--soma-assets-root" in result.stdout
        assert "--soma-x-root" not in result.stdout
