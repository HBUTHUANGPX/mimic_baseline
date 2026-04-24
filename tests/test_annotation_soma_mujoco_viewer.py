from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWER_MODULE_PATH = REPO_ROOT / "hdf5_parse" / "annotation_soma_mujoco_viewer.py"
REFERENCE_MODULE_PATH = REPO_ROOT / "soma-retargeter" / "app" / "motion_npz_player_common.py"


def load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reference_math_matches_soma_retargeter() -> None:
    viewer = load_module("annotation_soma_mujoco_viewer", VIEWER_MODULE_PATH)
    reference = load_module("motion_npz_player_common", REFERENCE_MODULE_PATH)

    local_transforms = np.array(
        [
            [
                [0.5, -0.2, 1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.8, 0.0, 0.0, np.sin(np.pi / 8.0), np.cos(np.pi / 8.0)],
                [0.2, 0.1, 0.4, 0.0, np.sin(np.pi / 12.0), 0.0, np.cos(np.pi / 12.0)],
            ]
        ],
        dtype=np.float32,
    )
    parent_indices = np.array([-1, 0, 1], dtype=np.int32)

    ref_pos, ref_quat = reference.compute_global_joint_transforms(local_transforms, parent_indices)
    got_pos, got_quat = viewer.compute_global_joint_transforms(local_transforms, parent_indices)
    np.testing.assert_allclose(got_pos, ref_pos, atol=1e-6)
    np.testing.assert_allclose(got_quat, ref_quat, atol=1e-6)

    ref_vis_pos, ref_vis_quat = reference.apply_visualization_frame(ref_pos, ref_quat)
    got_vis_pos, got_vis_quat = viewer.apply_visualization_frame(got_pos, got_quat)
    np.testing.assert_allclose(got_vis_pos, ref_vis_pos, atol=1e-6)
    np.testing.assert_allclose(got_vis_quat, ref_vis_quat, atol=1e-6)


def test_load_human_motion_npz_reads_annotation_soma_style_payload(tmp_path: Path) -> None:
    viewer = load_module("annotation_soma_mujoco_viewer_loader", VIEWER_MODULE_PATH)
    npz_path = tmp_path / "annotation_soma.npz"
    np.savez(
        npz_path,
        human_local_transforms=np.array(
            [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]],
            dtype=np.float32,
        ),
        human_parent_indices=np.array([-1], dtype=np.int32),
        human_joint_names=np.array(["Hips"], dtype="<U16"),
        fps=np.asarray(20.0, dtype=np.float32),
        scalar_first=np.asarray(False),
    )

    motion = viewer.load_human_motion_npz(npz_path)

    assert motion.fps == 20.0
    assert motion.scalar_first is False
    np.testing.assert_array_equal(motion.parent_indices, np.array([-1], dtype=np.int32))
    assert motion.joint_names == ["Hips"]


def test_compute_visualized_global_transforms_matches_saved_globals_shape() -> None:
    viewer = load_module("annotation_soma_mujoco_viewer_visualized", VIEWER_MODULE_PATH)
    local_transforms = np.array(
        [
            [
                [0.0, 0.0, 0.0, -np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=np.float32,
    )
    parent_indices = np.array([-1, 0], dtype=np.int32)

    positions, rotations = viewer.compute_visualized_global_transforms(local_transforms, parent_indices)

    np.testing.assert_allclose(positions[0, 1] - positions[0, 0], np.array([0.0, 0.0, 1.0], dtype=np.float32), atol=1e-6)
    assert rotations.shape == (1, 2, 4)


def test_build_arg_parser_shows_axes_by_default() -> None:
    viewer = load_module("annotation_soma_mujoco_viewer_parser", VIEWER_MODULE_PATH)

    with pytest.raises(SystemExit):
        viewer.build_arg_parser().parse_args([])

    args = viewer.build_arg_parser().parse_args(["--npz", "sample.npz"])
    assert args.hide_axes is False

    args = viewer.build_arg_parser().parse_args(["--npz", "sample.npz", "--hide-axes"])
    assert args.hide_axes is True
