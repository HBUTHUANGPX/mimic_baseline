from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "hdf5_parse" / "scripts" / "smpl_body_mujoco_viewer.py"
HDF5_PATH = REPO_ROOT / "hdf5_parse" / "hdf5" / "annotation.hdf5"


def load_module():
    spec = importlib.util.spec_from_file_location("smpl_body_mujoco_viewer", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_args_uses_streamlined_defaults():
    module = load_module()

    args = module.parse_args([])

    assert args.hdf5_path == HDF5_PATH
    assert args.model_type == "smplh"
    assert args.start == 0
    assert args.end == -1
    assert args.stride == 1
    assert args.loop is False
    assert args.root_frame is False
    assert args.mesh_points == 400


def test_parse_args_accepts_explicit_body_model_paths(tmp_path: Path):
    module = load_module()

    smplh_path = tmp_path / "SMPLH_NEUTRAL.pkl"
    smpl_path = tmp_path / "SMPL_NEUTRAL.pkl"
    smplh_path.write_bytes(b"smplh")
    smpl_path.write_bytes(b"smpl")

    args = module.parse_args(
        [
            "--model-type",
            "smpl",
            "--smplh-model-path",
            str(smplh_path),
            "--smpl-model-path",
            str(smpl_path),
            "--mesh-points",
            "128",
        ]
    )

    assert args.model_type == "smpl"
    assert args.smplh_model_path == smplh_path
    assert args.smpl_model_path == smpl_path
    assert args.mesh_points == 128


def test_sample_vertex_indices_caps_count_and_stays_sorted():
    module = load_module()

    indices = module.sample_vertex_indices(num_vertices=10, max_points=4)

    assert indices.shape == (4,)
    assert np.all(indices[:-1] <= indices[1:])
    np.testing.assert_array_equal(indices, [0, 3, 6, 9])


def test_prepare_motion_clip_switches_between_smplh_and_smpl():
    module = load_module()

    smplh_clip = module.load_motion_clip_for_viewer(HDF5_PATH, model_type="smplh", start_frame=0, end_frame=2)
    smpl_clip = module.load_motion_clip_for_viewer(HDF5_PATH, model_type="smpl", start_frame=0, end_frame=2)

    assert smplh_clip.model_type == "smplh"
    assert smplh_clip.body_pose.shape == (2, 63)
    assert smpl_clip.model_type == "smpl"
    assert smpl_clip.body_pose.shape == (2, 69)


def test_instantiate_body_model_loads_smpl_npz_without_create(tmp_path: Path, monkeypatch):
    module = load_module()

    npz_path = tmp_path / "SMPL_NEUTRAL.npz"
    np.savez(npz_path, dummy=np.array([1], dtype=np.float32))

    clip = SimpleNamespace(betas=np.zeros((2, 16), dtype=np.float32))
    args = SimpleNamespace(model_type="smpl", smplh_model_path=None, smpl_model_path=npz_path)
    sentinel = object()
    calls: dict[str, object] = {}

    def fake_create(**kwargs):
        raise AssertionError("smplx.create should not be used for SMPL .npz files")

    def fake_smpl(model_path, **kwargs):
        calls["model_path"] = model_path
        calls["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(module.smplx, "create", fake_create)
    monkeypatch.setattr(module.smplx, "SMPL", fake_smpl)

    body_model = module.instantiate_body_model(args, clip)

    assert body_model is sentinel
    assert calls["model_path"] == str(npz_path)
    assert "data_struct" in calls["kwargs"]
