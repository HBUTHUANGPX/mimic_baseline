from __future__ import annotations

import importlib

import numpy as np


def load_module():
    module = importlib.import_module("hdf5_parse.motion_export.core")
    return importlib.reload(module)


def test_build_annotation_export_payload_preserves_timeline_and_text_fields() -> None:
    module = load_module()
    text_payload = {
        "main_task_texts": np.array(["UNKNOWN", "Task"], dtype=object),
        "sub_task_texts": np.array(["UNKNOWN", "Sub"], dtype=object),
        "current_action_texts": np.array(["UNKNOWN", "Act"], dtype=object),
        "interaction_texts": np.array(["UNKNOWN", "Touch"], dtype=object),
        "main_task_text_indices": np.array([1, 1], dtype=np.int32),
        "sub_task_text_indices": np.array([0, 1], dtype=np.int32),
        "current_action_text_indices": np.array([1, 0], dtype=np.int32),
        "interaction_text_indices": np.array([0, 1], dtype=np.int32),
    }
    payload = module.build_annotation_export_payload(
        fps=30,
        frame_nums=np.array([10, 20], dtype=np.int32),
        frame_timestamps=np.array([1000, 2000], dtype=np.int64),
        text_payload=text_payload,
    )

    assert payload["fps"].item() == 30
    assert payload["num_frames"].item() == 2
    np.testing.assert_array_equal(payload["timeline_frame_indices"], np.array([10, 20], dtype=np.int32))
    np.testing.assert_array_equal(payload["frame_timestamps"], np.array([1000, 2000], dtype=np.int64))
    assert payload["main_task_texts"].tolist() == ["UNKNOWN", "Task"]
    assert payload["sub_task_texts"].tolist() == ["UNKNOWN", "Sub"]
    np.testing.assert_array_equal(payload["current_action_text_indices"], np.array([1, 0], dtype=np.int32))
    assert "human_local_transforms" not in payload
    assert "human_global_pos" not in payload
    assert "smpl_transl" not in payload
    assert "soma_poses" not in payload


def test_build_annotation_export_payload_coerces_string_lists_and_object_arrays() -> None:
    module = load_module()
    payload = module.build_annotation_export_payload(
        fps=20,
        frame_nums=np.array([3], dtype=np.int32),
        frame_timestamps=np.array([33], dtype=np.int64),
        text_payload={
            "main_task_texts": ["UNKNOWN", "Task"],
            "main_task_text_indices": np.array([1], dtype=np.int32),
        },
        extra_payload={
            "sub_task_texts": np.array(["UNKNOWN", "Sub"], dtype=object),
            "sub_task_text_indices": np.array([0], dtype=np.int32),
            "note_count": np.array([5], dtype=np.int32),
        },
    )

    assert payload["main_task_texts"].dtype.kind == "U"
    assert payload["sub_task_texts"].dtype.kind == "U"
    np.testing.assert_array_equal(payload["note_count"], np.array([5], dtype=np.int32))
