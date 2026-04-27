from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
HDF5_PATH = next(
    iter(sorted((REPO_ROOT / "hdf5_parse" / "test_data").glob("*/*/annotation.hdf5"))),
    REPO_ROOT / "hdf5_parse" / "hdf5" / "annotation.hdf5",
)

from soma_retargeter.assets.bvh import load_bvh


CLI_MODULE_PATH = REPO_ROOT / "hdf5_parse" / "scripts" / "export_hdf5_to_soma_bvh.py"


def load_bvh_module():
    module = importlib.import_module("hdf5_parse.motion_export.bvh")
    return importlib.reload(module)


def _pose7(position: list[float], euler_zyx_deg: list[float]) -> np.ndarray:
    quat = Rotation.from_euler("ZYX", euler_zyx_deg, degrees=True).as_quat().astype(np.float32)
    return np.asarray([*position, *quat.tolist()], dtype=np.float32)


def _transform_to_pose7(transform) -> np.ndarray:
    if isinstance(transform, np.ndarray):
        array = np.asarray(transform, dtype=np.float32)
        if array.shape != (7,):
            raise ValueError(f"Expected ndarray transform shape (7,), got {array.shape}.")
        return array
    return np.asarray(
        [
            float(transform.p[0]),
            float(transform.p[1]),
            float(transform.p[2]),
            float(transform.q[0]),
            float(transform.q[1]),
            float(transform.q[2]),
            float(transform.q[3]),
        ],
        dtype=np.float32,
    )


def _assert_quat_allclose(actual: np.ndarray, expected: np.ndarray, atol: float = 1e-4) -> None:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    same = np.allclose(actual, expected, atol=atol)
    flipped = np.allclose(actual, -expected, atol=atol)
    if not (same or flipped):
        np.testing.assert_allclose(actual, expected, atol=atol)


def test_write_soma_bvh_round_trips_with_soma_loader(tmp_path: Path) -> None:
    write_soma_bvh = load_bvh_module().write_soma_bvh

    joint_names = ["Root", "Hips", "Head"]
    parent_indices = np.asarray([-1, 0, 1], dtype=np.int32)
    reference_local_transforms = np.asarray(
        [
            _pose7([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            _pose7([0.0, 1.0, 0.0], [0.0, 0.0, 0.0]),
            _pose7([0.0, 0.5, 0.0], [0.0, 0.0, 0.0]),
        ],
        dtype=np.float32,
    )
    human_local_transforms = np.asarray(
        [
            [
                _pose7([0.10, 0.20, 0.30], [10.0, 20.0, 30.0]),
                _pose7([0.00, 1.05, 0.00], [-5.0, 15.0, -10.0]),
                _pose7([0.00, 0.50, 0.00], [3.0, -2.0, 7.0]),
            ],
            [
                _pose7([-0.20, 0.10, 0.00], [-12.0, 5.0, 8.0]),
                _pose7([0.00, 0.95, 0.10], [6.0, -9.0, 4.0]),
                _pose7([0.00, 0.50, 0.00], [0.0, 11.0, -13.0]),
            ],
        ],
        dtype=np.float32,
    )
    output_path = tmp_path / "demo_soma.bvh"

    write_soma_bvh(
        output_path=output_path,
        joint_names=joint_names,
        parent_indices=parent_indices,
        reference_local_transforms=reference_local_transforms,
        local_transforms=human_local_transforms,
        fps=20.0,
    )

    assert output_path.is_file()
    contents = output_path.read_text(encoding="utf-8")
    assert "ROOT Root" in contents
    assert "JOINT Hips" in contents
    assert "CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation" in contents
    assert "Frames: 2" in contents
    assert "Frame Time: 0.050000" in contents

    skeleton, animation = load_bvh(str(output_path))

    assert skeleton.joint_names == joint_names
    np.testing.assert_array_equal(skeleton.parent_indices, parent_indices)
    assert animation.num_frames == 2
    assert animation.sample_rate == 20.0

    loaded_frame0 = np.stack([_transform_to_pose7(tx) for tx in animation.get_local_transforms(0)], axis=0)
    loaded_frame1 = np.stack([_transform_to_pose7(tx) for tx in animation.get_local_transforms(1)], axis=0)

    np.testing.assert_allclose(loaded_frame0[:, :3], human_local_transforms[0, :, :3] * 0.01, atol=1e-4)
    np.testing.assert_allclose(loaded_frame1[:, :3], human_local_transforms[1, :, :3] * 0.01, atol=1e-4)
    for joint_idx in range(len(joint_names)):
        _assert_quat_allclose(loaded_frame0[joint_idx, 3:7], human_local_transforms[0, joint_idx, 3:7], atol=1e-4)
        _assert_quat_allclose(loaded_frame1[joint_idx, 3:7], human_local_transforms[1, joint_idx, 3:7], atol=1e-4)


def test_cli_parser_uses_expected_defaults() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("export_hdf5_to_soma_bvh", CLI_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    args = module.build_arg_parser().parse_args([])

    assert args.hdf5_path == HDF5_PATH
    assert args.output_path == Path("hdf5_parse/out/annotation_soma.bvh")
    assert args.device == "cuda"
    assert args.end_frame == -1
    assert args.batch_size is None


def test_cli_script_help_runs_without_repo_pythonpath() -> None:
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
    assert "--smpl-model-path" in result.stdout


def test_export_hdf5_to_soma_bvh_data_keeps_identity_root_rotation(monkeypatch) -> None:
    module = load_bvh_module()

    class DummySelection:
        frame_nums = np.asarray([10, 11], dtype=np.int32)

    class DummyMotion:
        fps = 20.0

    root_identity = _pose7([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    hips_pose = _pose7([0.1, 1.2, -0.3], [20.0, -10.0, 5.0])
    local_transforms = np.asarray([[root_identity, hips_pose]], dtype=np.float32)
    reference_local_transforms = np.asarray([root_identity, _pose7([0.0, 1.0, 0.0], [0.0, 0.0, 0.0])], dtype=np.float32)

    monkeypatch.setattr(
        "hdf5_parse.motion_export.bvh.load_body_frame_selection",
        lambda *args, **kwargs: DummySelection(),
    )
    monkeypatch.setattr(
        "hdf5_parse.motion_export.bvh.selection_to_smpl_body_motion",
        lambda selection: DummyMotion(),
    )
    monkeypatch.setattr(
        "hdf5_parse.motion_export.bvh.run_soma_inversion",
        lambda *args, **kwargs: {
            "joint_names": ["Root", "Hips"],
            "parent_indices": np.asarray([-1, 0], dtype=np.int32),
            "reference_local_transforms": reference_local_transforms,
            "local_transforms": local_transforms,
        },
    )

    payload = module.export_hdf5_to_soma_bvh_data()

    np.testing.assert_allclose(payload["reference_local_transforms"][0, 3:7], np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(payload["human_local_transforms"][0, 0, 3:7], np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32))


def test_canonicalize_motion_local_transforms_for_bvh_preserves_visualized_pose() -> None:
    canonicalize_motion_local_transforms_for_bvh = load_bvh_module().canonicalize_motion_local_transforms_for_bvh
    from motion_reconstruction.human_pose import (
        apply_visualization_frame_xyzw,
        compute_global_joint_transforms_xyzw,
    )

    parent_indices = np.asarray([-1, 0, 1], dtype=np.int32)
    local_transforms = np.asarray(
        [
            [
                _pose7([0.0, 0.0, 0.0], [0.0, 0.0, -90.0]),
                _pose7([-0.2, 0.1, -0.5], [-37.0, -80.0, -88.0]),
                _pose7([0.0, 0.5, 0.0], [5.0, -3.0, 7.0]),
            ]
        ],
        dtype=np.float32,
    )

    expected_pos, expected_quat = apply_visualization_frame_xyzw(
        *compute_global_joint_transforms_xyzw(local_transforms, parent_indices)
    )

    canonical = canonicalize_motion_local_transforms_for_bvh(
        local_transforms=local_transforms,
        parent_indices=parent_indices,
    )
    actual_pos, actual_quat = apply_visualization_frame_xyzw(
        *compute_global_joint_transforms_xyzw(canonical, parent_indices)
    )

    np.testing.assert_allclose(canonical[0, 0, :3], np.zeros(3, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(canonical[0, 0, 3:7], np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(actual_pos, expected_pos, atol=1e-5)
    for joint_idx in range(1, actual_quat.shape[1]):
        _assert_quat_allclose(actual_quat[0, joint_idx], expected_quat[0, joint_idx], atol=1e-5)
