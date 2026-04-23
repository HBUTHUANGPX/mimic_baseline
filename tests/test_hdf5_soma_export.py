from __future__ import annotations

import importlib.util
from pathlib import Path
import os
import subprocess
import sys

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
HDF5_PATH = REPO_ROOT / "hdf5_parse" / "hdf5" / "annotation.hdf5"
EXPORT_MODULE_PATH = REPO_ROOT / "hdf5_parse" / "hdf5_soma_export.py"
CLI_MODULE_PATH = REPO_ROOT / "hdf5_parse" / "export_hdf5_to_soma_npz.py"


def load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_body_valid_mask_drops_non_finite_frames() -> None:
    module = load_module("hdf5_soma_export", EXPORT_MODULE_PATH)
    root_pose7 = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, np.nan, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    body_quats = np.ones((3, 21, 4), dtype=np.float32)
    body_quats[2, 0, 0] = np.inf
    betas = np.zeros((3, 10), dtype=np.float32)

    valid_mask = module.build_body_valid_mask(root_pose7=root_pose7, body_quats=body_quats, betas=betas)

    np.testing.assert_array_equal(valid_mask, np.array([True, False, False]))


def test_load_caption_json_reads_scalar_json_dataset() -> None:
    module = load_module("hdf5_soma_export", EXPORT_MODULE_PATH)
    caption = module.load_caption_json(HDF5_PATH)

    assert caption["config"]["Main Task"] == "Sorting colorful star-shaped paper origami"
    assert len(caption["segments"]) == 18


def test_build_frame_timestamp_lookup_matches_frame_numbers() -> None:
    module = load_module("hdf5_soma_export", EXPORT_MODULE_PATH)
    with h5py.File(HDF5_PATH, "r") as h5_file:
        frame_nums = np.asarray(h5_file["full_body_mocap/frame_nums"][:8], dtype=np.int32)
        timestamp_lookup = module.build_frame_timestamp_lookup(
            video_frame_numbers=np.asarray(h5_file["video/frame_number"][:], dtype=np.int32),
            video_timestamps=np.asarray(h5_file["video/device_timestamp"][:], dtype=np.int64),
        )

    resolved = np.asarray([timestamp_lookup[int(frame_num)] for frame_num in frame_nums], dtype=np.int64)
    assert resolved.shape == (8,)
    assert np.all(np.diff(resolved) >= 0)


def test_selection_to_smpl_body_motion_pads_smplh_body_to_smpl_dims() -> None:
    module = load_module("hdf5_soma_export", EXPORT_MODULE_PATH)
    selection = module.BodyFrameSelection(
        root_pose7=np.array([[1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3]], dtype=np.float32),
        body_quats=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (1, 21, 1)),
        betas=np.zeros((1, 16), dtype=np.float32),
        frame_nums=np.array([7], dtype=np.int32),
        frame_timestamps=np.array([123], dtype=np.int64),
        fps=20.0,
    )

    motion = module.selection_to_smpl_body_motion(selection)

    assert motion.body_pose.shape == (1, 69)
    np.testing.assert_allclose(motion.body_pose[:, 63:], 0.0)


def test_align_caption_texts_to_frames_uses_unknown_slot_zero() -> None:
    module = load_module("hdf5_soma_export", EXPORT_MODULE_PATH)
    caption = {
        "config": {"Main Task": "Main Task Example"},
        "segments": [
            {
                "start_frame": "100",
                "end_frame": "299",
                "Sub Task": "Sub A",
                "Current Action": [
                    {
                        "label": "Action A",
                        "description": "first action",
                        "start_frame": 100,
                        "end_frame": 199,
                    }
                ],
                "interaction": {
                    "100": "Interaction A",
                    "220": "Interaction B",
                },
            }
        ],
    }
    frame_timestamps = np.array([50, 120, 210, 260, 320], dtype=np.int64)

    text_payload = module.align_caption_texts_to_frames(
        caption=caption,
        frame_timestamps=frame_timestamps,
    )

    assert text_payload["main_task_texts"][0] == module.UNKNOWN_TEXT
    assert text_payload["sub_task_texts"][0] == module.UNKNOWN_TEXT
    assert text_payload["current_action_texts"][0] == module.UNKNOWN_TEXT
    assert text_payload["interaction_texts"][0] == module.UNKNOWN_TEXT
    np.testing.assert_array_equal(text_payload["main_task_text_indices"], np.array([1, 1, 1, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(text_payload["sub_task_text_indices"], np.array([0, 1, 1, 1, 0], dtype=np.int32))
    np.testing.assert_array_equal(text_payload["current_action_text_indices"], np.array([0, 1, 0, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(text_payload["interaction_text_indices"], np.array([0, 1, 1, 2, 0], dtype=np.int32))


def test_align_caption_texts_to_real_hdf5_frames() -> None:
    module = load_module("hdf5_soma_export", EXPORT_MODULE_PATH)
    caption = module.load_caption_json(HDF5_PATH)
    with h5py.File(HDF5_PATH, "r") as h5_file:
        frame_nums = np.asarray(h5_file["full_body_mocap/frame_nums"][:6], dtype=np.int32)
        timestamp_lookup = module.build_frame_timestamp_lookup(
            video_frame_numbers=np.asarray(h5_file["video/frame_number"][:], dtype=np.int32),
            video_timestamps=np.asarray(h5_file["video/device_timestamp"][:], dtype=np.int64),
        )
    frame_timestamps = np.asarray([timestamp_lookup[int(frame_num)] for frame_num in frame_nums], dtype=np.int64)

    text_payload = module.align_caption_texts_to_frames(caption=caption, frame_timestamps=frame_timestamps)

    assert text_payload["main_task_texts"][1] == "Sorting colorful star-shaped paper origami"
    assert text_payload["sub_task_text_indices"].shape == (6,)
    assert text_payload["current_action_text_indices"].shape == (6,)
    assert text_payload["interaction_text_indices"].shape == (6,)
    assert np.all(text_payload["main_task_text_indices"] == 1)


def test_cli_parser_uses_expected_defaults() -> None:
    module = load_module("export_hdf5_to_soma_npz", CLI_MODULE_PATH)
    args = module.build_arg_parser().parse_args([])

    assert args.hdf5_path == Path("hdf5_parse/hdf5/annotation.hdf5")
    assert args.output_path == Path("hdf5_parse/out/annotation_soma.npz")
    assert args.device == "cuda"
    assert args.end_frame == -1
    assert args.batch_size is None


def test_cli_script_help_runs_without_repo_pythonpath() -> None:
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


def test_export_hdf5_to_soma_payload_matches_reference_player_display_semantics(monkeypatch) -> None:
    module = load_module("hdf5_soma_export_for_export_test", EXPORT_MODULE_PATH)
    selection = module.BodyFrameSelection(
        root_pose7=np.zeros((1, 7), dtype=np.float32),
        body_quats=np.zeros((1, 21, 4), dtype=np.float32),
        betas=np.zeros((1, 10), dtype=np.float32),
        frame_nums=np.array([7], dtype=np.int32),
        frame_timestamps=np.array([123], dtype=np.int64),
        fps=20.0,
    )
    motion = module.SMPLBodyMotion(
        global_orient=np.zeros((1, 3), dtype=np.float32),
        body_pose=np.zeros((1, 69), dtype=np.float32),
        transl=np.zeros((1, 3), dtype=np.float32),
        betas=np.zeros((1, 10), dtype=np.float32),
        frame_nums=np.array([7], dtype=np.int32),
        frame_timestamps=np.array([123], dtype=np.int64),
        fps=20.0,
    )
    monkeypatch.setattr(module, "load_body_frame_selection", lambda *args, **kwargs: selection)
    monkeypatch.setattr(module, "selection_to_smpl_body_motion", lambda current: motion)
    monkeypatch.setattr(module, "load_caption_json", lambda *args, **kwargs: {"config": {"Main Task": "Main"}, "segments": []})
    expected_local = np.array(
        [
            [
                [0.0, 0.0, 0.0, -np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=np.float32,
    )
    expected_parents = np.array([-1, 0, 1], dtype=np.int32)
    expected_pos, expected_quat = module.apply_visualization_frame(
        *module.compute_global_joint_transforms(expected_local, expected_parents)
    )
    monkeypatch.setattr(
        module,
        "align_caption_texts_to_frames",
        lambda **kwargs: {
            "main_task_texts": np.array(["UNKNOWN", "Main"], dtype=object),
            "sub_task_texts": np.array(["UNKNOWN"], dtype=object),
            "current_action_texts": np.array(["UNKNOWN"], dtype=object),
            "interaction_texts": np.array(["UNKNOWN"], dtype=object),
            "main_task_text_indices": np.array([1], dtype=np.int32),
            "sub_task_text_indices": np.array([0], dtype=np.int32),
            "current_action_text_indices": np.array([0], dtype=np.int32),
            "interaction_text_indices": np.array([0], dtype=np.int32),
        },
    )
    monkeypatch.setattr(module, "load_selected_joint_names", lambda *args, **kwargs: {"Hips", "Head"})
    monkeypatch.setattr(
        module,
        "run_soma_inversion",
        lambda *args, **kwargs: {
            "joint_names": ["Root", "Hips", "Head"],
            "parent_indices": expected_parents,
            "reference_local_transforms": np.array(
                [
                    [0.0, 0.0, 0.0, -np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            "local_transforms": np.array(
                [
                    [
                        [0.0, 0.0, 0.0, -np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    ]
                ],
                dtype=np.float32,
            ),
            "world_transforms": np.array(
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    ]
                ],
                dtype=np.float32,
            ),
            "soma_poses": np.zeros((1, 3, 3), dtype=np.float32),
            "soma_transl": np.zeros((1, 3), dtype=np.float32),
            "soma_joint_orient": np.zeros((3, 3, 3), dtype=np.float32),
            "per_vertex_error": np.zeros((1,), dtype=np.float32),
        },
    )

    payload = module.export_hdf5_to_soma_payload("unused.hdf5")

    assert payload["human_joint_names"].tolist()[0] == "Root"
    hips = payload["human_joint_names"].tolist().index("Hips")
    head = payload["human_joint_names"].tolist().index("Head")
    np.testing.assert_allclose(payload["human_global_pos"][0, [0, hips, head]], expected_pos[0], atol=1e-6)
    np.testing.assert_allclose(payload["human_global_quat"][0, [0, hips, head]], expected_quat[0], atol=1e-6)
    np.testing.assert_allclose(
        payload["human_local_transforms"][0, 0, 3:7],
        np.array([-np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32),
        atol=1e-6,
    )
    calc_pos, calc_quat = module.apply_visualization_frame(
        *module.compute_global_joint_transforms(
            payload["human_local_transforms"],
            payload["human_parent_indices"],
        )
    )
    np.testing.assert_allclose(calc_pos[:, [0, hips, head]], payload["human_global_pos"][:, [0, hips, head]], atol=1e-6)
    np.testing.assert_allclose(calc_quat[:, [0, hips, head]], payload["human_global_quat"][:, [0, hips, head]], atol=1e-6)


def test_export_hdf5_to_soma_payload_preserves_root_based_reference_semantics(monkeypatch) -> None:
    module = load_module("hdf5_soma_export_root_semantics_test", EXPORT_MODULE_PATH)
    sample = np.load(
        REPO_ROOT / "soma-retargeter" / "assets" / "motions" / "soma_uniform_bvh_export" / "240918" / "body_check_001__A548.npz",
        allow_pickle=False,
    )
    sample_local = np.asarray(sample["human_local_transforms"][:1], dtype=np.float32)
    sample_parents = np.asarray(sample["human_parent_indices"], dtype=np.int32)
    sample_names = sample["human_joint_names"].tolist()
    sample_reference = np.asarray(sample["human_reference_local_transforms"], dtype=np.float32)
    expected_pos, expected_quat = module.apply_visualization_frame(
        *module.compute_global_joint_transforms(sample_local, sample_parents)
    )

    selection = module.BodyFrameSelection(
        root_pose7=np.zeros((1, 7), dtype=np.float32),
        body_quats=np.zeros((1, 21, 4), dtype=np.float32),
        betas=np.zeros((1, 10), dtype=np.float32),
        frame_nums=np.array([7], dtype=np.int32),
        frame_timestamps=np.array([123], dtype=np.int64),
        fps=20.0,
    )
    motion = module.SMPLBodyMotion(
        global_orient=np.zeros((1, 3), dtype=np.float32),
        body_pose=np.zeros((1, 69), dtype=np.float32),
        transl=np.zeros((1, 3), dtype=np.float32),
        betas=np.zeros((1, 10), dtype=np.float32),
        frame_nums=np.array([7], dtype=np.int32),
        frame_timestamps=np.array([123], dtype=np.int64),
        fps=20.0,
    )

    monkeypatch.setattr(module, "load_body_frame_selection", lambda *args, **kwargs: selection)
    monkeypatch.setattr(module, "selection_to_smpl_body_motion", lambda current: motion)
    monkeypatch.setattr(module, "load_caption_json", lambda *args, **kwargs: {"config": {"Main Task": "Main"}, "segments": []})
    monkeypatch.setattr(
        module,
        "align_caption_texts_to_frames",
        lambda **kwargs: {
            "main_task_texts": np.array(["UNKNOWN", "Main"], dtype=object),
            "sub_task_texts": np.array(["UNKNOWN"], dtype=object),
            "current_action_texts": np.array(["UNKNOWN"], dtype=object),
            "interaction_texts": np.array(["UNKNOWN"], dtype=object),
            "main_task_text_indices": np.array([1], dtype=np.int32),
            "sub_task_text_indices": np.array([0], dtype=np.int32),
            "current_action_text_indices": np.array([0], dtype=np.int32),
            "interaction_text_indices": np.array([0], dtype=np.int32),
        },
    )
    monkeypatch.setattr(module, "load_selected_joint_names", lambda *args, **kwargs: {"Hips", "Head"})
    monkeypatch.setattr(
        module,
        "run_soma_inversion",
        lambda *args, **kwargs: {
            "joint_names": sample_names,
            "parent_indices": sample_parents,
            "reference_local_transforms": sample_reference,
            "local_transforms": sample_local,
            "world_transforms": np.zeros_like(sample_local),
            "soma_poses": np.zeros((1, len(sample_names), 3), dtype=np.float32),
            "soma_transl": np.zeros((1, 3), dtype=np.float32),
            "soma_joint_orient": np.zeros((len(sample_names), 3, 3), dtype=np.float32),
            "per_vertex_error": np.zeros((1,), dtype=np.float32),
        },
    )

    payload = module.export_hdf5_to_soma_payload("unused.hdf5")

    assert payload["human_joint_names"].tolist()[0] == "Root"
    hips = payload["human_joint_names"].tolist().index("Hips")
    head = payload["human_joint_names"].tolist().index("Head")
    expected_hips = sample_names.index("Hips")
    expected_head = sample_names.index("Head")
    np.testing.assert_allclose(payload["human_global_quat"][0, hips], expected_quat[0, expected_hips], atol=1e-6)
    np.testing.assert_allclose(payload["human_global_pos"][0, head] - payload["human_global_pos"][0, hips], expected_pos[0, expected_head] - expected_pos[0, expected_hips], atol=1e-6)


def test_export_hdf5_to_soma_payload_normalizes_root_parent_to_minus_one(monkeypatch) -> None:
    module = load_module("hdf5_soma_export_root_parent_test", EXPORT_MODULE_PATH)
    selection = module.BodyFrameSelection(
        root_pose7=np.zeros((1, 7), dtype=np.float32),
        body_quats=np.zeros((1, 21, 4), dtype=np.float32),
        betas=np.zeros((1, 10), dtype=np.float32),
        frame_nums=np.array([7], dtype=np.int32),
        frame_timestamps=np.array([123], dtype=np.int64),
        fps=20.0,
    )
    motion = module.SMPLBodyMotion(
        global_orient=np.zeros((1, 3), dtype=np.float32),
        body_pose=np.zeros((1, 69), dtype=np.float32),
        transl=np.zeros((1, 3), dtype=np.float32),
        betas=np.zeros((1, 10), dtype=np.float32),
        frame_nums=np.array([7], dtype=np.int32),
        frame_timestamps=np.array([123], dtype=np.int64),
        fps=20.0,
    )

    monkeypatch.setattr(module, "load_body_frame_selection", lambda *args, **kwargs: selection)
    monkeypatch.setattr(module, "selection_to_smpl_body_motion", lambda current: motion)
    monkeypatch.setattr(module, "load_caption_json", lambda *args, **kwargs: {"config": {"Main Task": "Main"}, "segments": []})
    monkeypatch.setattr(
        module,
        "align_caption_texts_to_frames",
        lambda **kwargs: {
            "main_task_texts": np.array(["UNKNOWN", "Main"], dtype=object),
            "sub_task_texts": np.array(["UNKNOWN"], dtype=object),
            "current_action_texts": np.array(["UNKNOWN"], dtype=object),
            "interaction_texts": np.array(["UNKNOWN"], dtype=object),
            "main_task_text_indices": np.array([1], dtype=np.int32),
            "sub_task_text_indices": np.array([0], dtype=np.int32),
            "current_action_text_indices": np.array([0], dtype=np.int32),
            "interaction_text_indices": np.array([0], dtype=np.int32),
        },
    )
    monkeypatch.setattr(module, "load_selected_joint_names", lambda *args, **kwargs: {"Hips"})
    monkeypatch.setattr(
        module,
        "run_soma_inversion",
        lambda *args, **kwargs: {
            "joint_names": ["Root", "Hips"],
            "parent_indices": np.array([0, 0], dtype=np.int32),
            "reference_local_transforms": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            "local_transforms": np.array(
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0],
                    ]
                ],
                dtype=np.float32,
            ),
            "world_transforms": np.zeros((1, 2, 7), dtype=np.float32),
            "soma_poses": np.zeros((1, 2, 3), dtype=np.float32),
            "soma_transl": np.zeros((1, 3), dtype=np.float32),
            "soma_joint_orient": np.zeros((2, 3, 3), dtype=np.float32),
            "per_vertex_error": np.zeros((1,), dtype=np.float32),
        },
    )

    payload = module.export_hdf5_to_soma_payload("unused.hdf5")

    np.testing.assert_array_equal(payload["human_parent_indices"], np.array([-1, 0], dtype=np.int32))


def test_export_hdf5_to_soma_payload_converts_local_root_to_pre_visualization_frame(monkeypatch) -> None:
    module = load_module("hdf5_soma_export_previs_root_test", EXPORT_MODULE_PATH)
    selection = module.BodyFrameSelection(
        root_pose7=np.zeros((1, 7), dtype=np.float32),
        body_quats=np.zeros((1, 21, 4), dtype=np.float32),
        betas=np.zeros((1, 10), dtype=np.float32),
        frame_nums=np.array([7], dtype=np.int32),
        frame_timestamps=np.array([123], dtype=np.int64),
        fps=20.0,
    )
    motion = module.SMPLBodyMotion(
        global_orient=np.zeros((1, 3), dtype=np.float32),
        body_pose=np.zeros((1, 69), dtype=np.float32),
        transl=np.zeros((1, 3), dtype=np.float32),
        betas=np.zeros((1, 10), dtype=np.float32),
        frame_nums=np.array([7], dtype=np.int32),
        frame_timestamps=np.array([123], dtype=np.int64),
        fps=20.0,
    )

    monkeypatch.setattr(module, "load_body_frame_selection", lambda *args, **kwargs: selection)
    monkeypatch.setattr(module, "selection_to_smpl_body_motion", lambda current: motion)
    monkeypatch.setattr(module, "load_caption_json", lambda *args, **kwargs: {"config": {"Main Task": "Main"}, "segments": []})
    monkeypatch.setattr(
        module,
        "align_caption_texts_to_frames",
        lambda **kwargs: {
            "main_task_texts": np.array(["UNKNOWN", "Main"], dtype=object),
            "sub_task_texts": np.array(["UNKNOWN"], dtype=object),
            "current_action_texts": np.array(["UNKNOWN"], dtype=object),
            "interaction_texts": np.array(["UNKNOWN"], dtype=object),
            "main_task_text_indices": np.array([1], dtype=np.int32),
            "sub_task_text_indices": np.array([0], dtype=np.int32),
            "current_action_text_indices": np.array([0], dtype=np.int32),
            "interaction_text_indices": np.array([0], dtype=np.int32),
        },
    )
    monkeypatch.setattr(module, "load_selected_joint_names", lambda *args, **kwargs: {"Hips", "Head"})
    monkeypatch.setattr(
        module,
        "run_soma_inversion",
        lambda *args, **kwargs: {
            "joint_names": ["Root", "Hips", "Head"],
            "parent_indices": np.array([0, 0, 1], dtype=np.int32),
            "reference_local_transforms": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            "local_transforms": np.array(
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    ]
                ],
                dtype=np.float32,
            ),
            "world_transforms": np.zeros((1, 3, 7), dtype=np.float32),
            "soma_poses": np.zeros((1, 3, 3), dtype=np.float32),
            "soma_transl": np.zeros((1, 3), dtype=np.float32),
            "soma_joint_orient": np.zeros((3, 3, 3), dtype=np.float32),
            "per_vertex_error": np.zeros((1,), dtype=np.float32),
        },
    )

    payload = module.export_hdf5_to_soma_payload("unused.hdf5")
    expected_root = np.array([-np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32)

    np.testing.assert_allclose(payload["human_local_transforms"][0, 0, 3:7], expected_root, atol=1e-6)
    hips = payload["human_joint_names"].tolist().index("Hips")
    head = payload["human_joint_names"].tolist().index("Head")
    np.testing.assert_allclose(
        payload["human_global_pos"][0, head] - payload["human_global_pos"][0, hips],
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        atol=1e-6,
    )
