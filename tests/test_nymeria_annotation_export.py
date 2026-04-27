import numpy as np

from nymeria_parse.motion_export.core import (
    DEFAULT_SEQUENCE_DIR,
    UNKNOWN_TEXT,
    build_annotation_payload,
    load_narration_rows,
)


def test_load_narration_rows_reads_activity_and_atomic_texts():
    rows = load_narration_rows(DEFAULT_SEQUENCE_DIR / "narration" / "activity_summarization.csv")

    assert rows
    assert rows[0].start_time_ms == 5831823
    assert rows[0].end_time_ms == 5861818
    assert "talking with her peer" in rows[0].text


def test_build_annotation_payload_has_hdf5_style_text_indices():
    payload, summary = build_annotation_payload(DEFAULT_SEQUENCE_DIR, start_frame=3500, end_frame=4500)

    assert set(payload) == {
        "fps",
        "num_frames",
        "timeline_frame_indices",
        "frame_timestamps",
        "main_task_texts",
        "main_task_text_indices",
        "sub_task_texts",
        "sub_task_text_indices",
        "current_action_texts",
        "current_action_text_indices",
        "interaction_texts",
        "interaction_text_indices",
    }
    assert int(payload["num_frames"]) == 1000
    assert payload["timeline_frame_indices"].shape == (1000,)
    assert payload["frame_timestamps"].shape == (1000,)
    assert payload["main_task_texts"][0] == UNKNOWN_TEXT
    assert payload["interaction_texts"][0] == UNKNOWN_TEXT
    assert payload["sub_task_text_indices"].shape == (1000,)
    assert payload["current_action_text_indices"].shape == (1000,)
    assert np.max(payload["sub_task_text_indices"]) > 0
    assert np.max(payload["current_action_text_indices"]) > 0
    assert summary["activity_covered_frames"] > 0
    assert summary["atomic_action_covered_frames"] > 0
