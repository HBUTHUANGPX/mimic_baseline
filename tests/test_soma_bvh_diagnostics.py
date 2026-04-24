from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "hdf5_parse" / "soma_bvh_diagnostics.py"


def load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_demo_payload():
    joint_names = ["Root", "Hips", "Spine1"]
    parent_indices = np.array([-1, 0, 1], dtype=np.int32)
    reference = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    local = np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 100.0, 0.0, 0.0, 0.0, np.sin(np.pi / 8.0), np.cos(np.pi / 8.0)],
                [0.0, 20.0, 0.0, 0.0, np.sin(np.pi / 12.0), 0.0, np.cos(np.pi / 12.0)],
            ]
        ],
        dtype=np.float32,
    )
    return joint_names, parent_indices, reference, local


def test_compare_annotation_npz_against_bvh_reports_zero_diff_for_matching_export(tmp_path: Path) -> None:
    diagnostics = load_module("soma_bvh_diagnostics_match", MODULE_PATH)
    from hdf5_parse.motion_export.bvh import write_soma_bvh

    joint_names, parent_indices, reference, local = _make_demo_payload()
    bvh_path = tmp_path / "sample.bvh"
    write_soma_bvh(
        output_path=bvh_path,
        joint_names=joint_names,
        parent_indices=parent_indices,
        reference_local_transforms=reference,
        local_transforms=local,
        fps=20.0,
    )
    _, global_pos, global_quat, _, _ = diagnostics.load_bvh_visualized_globals(bvh_path)
    npz_path = tmp_path / "sample.npz"
    np.savez(
        npz_path,
        fps=np.asarray(20.0, dtype=np.float32),
        scalar_first=np.asarray(False),
        human_joint_names=np.asarray(joint_names, dtype="<U16"),
        human_parent_indices=parent_indices,
        human_local_transforms=local,
        human_global_pos=global_pos,
        human_global_quat=global_quat,
    )

    report = diagnostics.compare_annotation_npz_against_bvh(
        npz_path=npz_path,
        bvh_path=bvh_path,
        focus_joint_names=None,
    )

    assert report.frame_count == 1
    assert report.npz_fps == 20.0
    assert report.bvh_fps == 20.0
    assert report.bvh_resampled_to_npz_fps is False
    assert report.overall_position_max_abs_diff < 1e-4
    assert report.overall_quaternion_max_abs_diff < 1e-4
    assert report.joint_stats["Hips"].position_max_abs_diff < 1e-4


def test_compare_annotation_npz_against_bvh_detects_root_basis_mismatch(tmp_path: Path) -> None:
    diagnostics = load_module("soma_bvh_diagnostics_mismatch", MODULE_PATH)
    from hdf5_parse.motion_export.bvh import write_soma_bvh

    joint_names, parent_indices, reference, local = _make_demo_payload()
    good_bvh_path = tmp_path / "sample_good.bvh"
    write_soma_bvh(
        output_path=good_bvh_path,
        joint_names=joint_names,
        parent_indices=parent_indices,
        reference_local_transforms=reference,
        local_transforms=local,
        fps=20.0,
    )
    _, global_pos, global_quat, _, _ = diagnostics.load_bvh_visualized_globals(good_bvh_path)
    npz_path = tmp_path / "sample.npz"
    np.savez(
        npz_path,
        fps=np.asarray(20.0, dtype=np.float32),
        scalar_first=np.asarray(False),
        human_joint_names=np.asarray(joint_names, dtype="<U16"),
        human_parent_indices=parent_indices,
        human_local_transforms=local,
        human_global_pos=global_pos,
        human_global_quat=global_quat,
    )
    bad_local = local.copy()
    bad_local[0, 0, 3:7] = np.array([-np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32)
    bvh_path = tmp_path / "sample_bad.bvh"
    write_soma_bvh(
        output_path=bvh_path,
        joint_names=joint_names,
        parent_indices=parent_indices,
        reference_local_transforms=reference,
        local_transforms=bad_local,
        fps=20.0,
    )

    report = diagnostics.compare_annotation_npz_against_bvh(
        npz_path=npz_path,
        bvh_path=bvh_path,
        focus_joint_names=None,
    )

    assert report.overall_position_max_abs_diff > 1.0
    assert report.overall_quaternion_max_abs_diff > 0.1


def test_compare_annotation_npz_against_bvh_resamples_bvh_when_fps_differs(tmp_path: Path, caplog) -> None:
    diagnostics = load_module("soma_bvh_diagnostics_resample", MODULE_PATH)
    from hdf5_parse.motion_export.bvh import write_soma_bvh

    joint_names, parent_indices, reference, local = _make_demo_payload()
    local = np.repeat(local, 3, axis=0)
    bvh_path = tmp_path / "sample_resample.bvh"
    write_soma_bvh(
        output_path=bvh_path,
        joint_names=joint_names,
        parent_indices=parent_indices,
        reference_local_transforms=reference,
        local_transforms=local,
        fps=20.0,
    )

    _, global_pos, global_quat, _, _ = diagnostics.load_bvh_visualized_globals(bvh_path, target_fps=50.0)
    npz_path = tmp_path / "sample_50fps.npz"
    np.savez(
        npz_path,
        fps=np.asarray(50.0, dtype=np.float32),
        scalar_first=np.asarray(False),
        human_joint_names=np.asarray(joint_names, dtype="<U16"),
        human_parent_indices=parent_indices,
        human_global_pos=global_pos,
        human_global_quat=global_quat,
    )

    with caplog.at_level("INFO"):
        report = diagnostics.compare_annotation_npz_against_bvh(
            npz_path=npz_path,
            bvh_path=bvh_path,
            focus_joint_names=None,
        )

    assert report.npz_fps == 50.0
    assert report.bvh_fps == 20.0
    assert report.bvh_resampled_to_npz_fps is True
    assert report.frame_count == global_pos.shape[0]
    assert report.overall_position_max_abs_diff < 1e-4
    assert report.overall_quaternion_max_abs_diff < 1e-4
    assert "resampling BVH to match NPZ fps" in caplog.text


def test_build_arg_parser_accepts_npz_and_bvh_paths() -> None:
    diagnostics = load_module("soma_bvh_diagnostics_parser", MODULE_PATH)

    args = diagnostics.build_arg_parser().parse_args(["--npz", "a.npz", "--bvh", "b.bvh"])
    assert args.npz == Path("a.npz")
    assert args.bvh == Path("b.bvh")
