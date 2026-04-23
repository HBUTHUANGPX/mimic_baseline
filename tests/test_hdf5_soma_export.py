from __future__ import annotations

import importlib.util
from pathlib import Path
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
