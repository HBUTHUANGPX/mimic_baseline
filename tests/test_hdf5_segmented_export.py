from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_MODULE_PATH = REPO_ROOT / "hdf5_parse" / "export_hdf5_segmented_motion.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_module():
    module = importlib.import_module("hdf5_parse.motion_export.segmented")
    return importlib.reload(module)


def test_split_frame_sequences_by_gap_uses_frame_nums() -> None:
    module = load_module()

    ranges = module.split_contiguous_frame_ranges(np.array([10, 11, 12, 20, 21, 30], dtype=np.int32), expected_step=1)

    assert ranges == [(0, 3), (3, 5), (5, 6)]


def test_build_segment_file_stem_uses_timestamp_range() -> None:
    module = load_module()

    stem = module.build_segment_file_stem(np.array([1740000010000, 1740000010050, 1740000010100], dtype=np.int64))

    assert stem == "annotation_1740000010000_1740000010100"


def test_save_smpl_motion_npz_writes_expected_fields(tmp_path: Path) -> None:
    module = load_module()
    hdf5_module = importlib.reload(importlib.import_module("hdf5_parse.motion_export.core"))
    motion = hdf5_module.SMPLBodyMotion(
        global_orient=np.zeros((2, 3), dtype=np.float32),
        body_pose=np.ones((2, 69), dtype=np.float32),
        transl=np.full((2, 3), 2.0, dtype=np.float32),
        betas=np.full((2, 10), 3.0, dtype=np.float32),
        frame_nums=np.array([7, 8], dtype=np.int32),
        frame_timestamps=np.array([1000, 1050], dtype=np.int64),
        fps=20.0,
    )

    output_path = tmp_path / "annotation_1000_1050.npz"
    module.save_smpl_motion_npz(motion, output_path)

    payload = np.load(output_path, allow_pickle=True)
    assert payload["fps"].item() == 20
    assert payload["num_frames"].item() == 2
    np.testing.assert_array_equal(payload["frame_nums"], np.array([7, 8], dtype=np.int32))
    np.testing.assert_array_equal(payload["frame_timestamps"], np.array([1000, 1050], dtype=np.int64))
    np.testing.assert_allclose(payload["smpl_body_pose"], np.ones((2, 69), dtype=np.float32))


def test_export_segmented_motion_saves_smpl_and_bvh_files(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    hdf5_module = importlib.reload(importlib.import_module("hdf5_parse.motion_export.core"))

    selection = hdf5_module.BodyFrameSelection(
        root_pose7=np.zeros((4, 7), dtype=np.float32),
        body_quats=np.zeros((4, 21, 4), dtype=np.float32),
        betas=np.zeros((4, 10), dtype=np.float32),
        frame_nums=np.array([10, 11, 20, 21], dtype=np.int32),
        frame_timestamps=np.array([1000, 1050, 2000, 2050], dtype=np.int64),
        fps=20.0,
    )
    motion = hdf5_module.SMPLBodyMotion(
        global_orient=np.zeros((4, 3), dtype=np.float32),
        body_pose=np.zeros((4, 69), dtype=np.float32),
        transl=np.zeros((4, 3), dtype=np.float32),
        betas=np.zeros((4, 10), dtype=np.float32),
        frame_nums=selection.frame_nums,
        frame_timestamps=selection.frame_timestamps,
        fps=20.0,
    )

    monkeypatch.setattr(module, "load_body_frame_selection", lambda *args, **kwargs: selection)
    monkeypatch.setattr(module, "selection_to_smpl_body_motion", lambda current: motion)
    monkeypatch.setattr(
        module,
        "run_soma_inversion",
        lambda *args, **kwargs: {
            "joint_names": ["Root", "Hips", "Head"],
            "parent_indices": np.array([-1, 0, 1], dtype=np.int32),
            "reference_local_transforms": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            "local_transforms": np.tile(
                np.array(
                    [
                        [0.2, -0.1, 0.3, 0.0, 0.0, -0.70710677, 0.70710677],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                )[None, ...],
                (4, 1, 1),
            ),
        },
    )
    monkeypatch.setattr(module, "normalize_root_parent_index", lambda parent_indices: np.asarray(parent_indices, dtype=np.int32))

    ensure_calls: list[tuple[int, ...]] = []

    def fake_ensure(*, local_transforms, parent_indices, joint_names):
        ensure_calls.append(tuple(np.asarray(local_transforms).shape))
        return np.asarray(local_transforms, dtype=np.float32)

    monkeypatch.setattr(
        module,
        "ensure_local_transforms_pre_visualization_frame",
        fake_ensure,
    )

    result = module.export_segmented_smpl_and_soma_bvh(
        hdf5_path="ignored.hdf5",
        smpl_output_dir=tmp_path / "smpl",
        soma_bvh_output_dir=tmp_path / "soma_bvh",
    )

    smpl_paths = result["smpl_paths"]
    bvh_paths = result["soma_bvh_paths"]
    assert [path.name for path in smpl_paths] == [
        "annotation_1000_1050.npz",
        "annotation_2000_2050.npz",
    ]
    assert [path.name for path in bvh_paths] == [
        "annotation_1000_1050.bvh",
        "annotation_2000_2050.bvh",
    ]
    assert all(path.is_file() for path in smpl_paths)
    assert all(path.is_file() for path in bvh_paths)
    assert ensure_calls == [(4, 3, 7)]

    bvh_lines = bvh_paths[0].read_text(encoding="utf-8").splitlines()
    frame_idx = bvh_lines.index("Frame Time: 0.050000") + 1
    frame_values = [float(value) for value in bvh_lines[frame_idx].split()]
    np.testing.assert_allclose(frame_values[:6], np.zeros(6, dtype=np.float32), atol=1e-6)
    assert any(abs(value) > 1e-6 for value in frame_values[6:12])


def test_segmented_cli_parser_uses_expected_defaults() -> None:
    spec = importlib.util.spec_from_file_location("export_hdf5_segmented_motion", CLI_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    args = module.build_arg_parser().parse_args([])

    assert args.hdf5_path == Path("hdf5_parse/hdf5/annotation.hdf5")
    assert args.smpl_output_dir == Path("hdf5_parse/out/smpl")
    assert args.soma_bvh_output_dir == Path("hdf5_parse/out/soma_bvh")
    assert args.device == "cuda"
    assert args.end_frame == -1


def test_segmented_cli_script_help_runs_without_repo_pythonpath() -> None:
    import os
    import subprocess

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(CLI_MODULE_PATH), "--help"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--smpl-output-dir" in result.stdout
