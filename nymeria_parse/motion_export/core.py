from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nymeria_parse.utils.mvnx import DEFAULT_SEQUENCE_DIR, load_mvnx_motion


UNKNOWN_TEXT = "UNKNOWN"
DEFAULT_OUTPUT_PATH = Path("nymeria_parse/out/annotation.npz")


@dataclass(frozen=True)
class NarrationRow:
    start_time_ms: int
    end_time_ms: int
    text: str


def _normalize_text_value(value: Any) -> str:
    text = str(value).strip()
    return text if text else UNKNOWN_TEXT


def _make_text_array(values: list[str]) -> np.ndarray:
    if not values:
        values = [UNKNOWN_TEXT]
    max_len = max(len(value) for value in values)
    return np.asarray(values, dtype=f"<U{max(1, max_len)}")


def _build_index_sequence(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    text_to_index = {UNKNOWN_TEXT: 0}
    deduped = [UNKNOWN_TEXT]
    indices = np.zeros(values.shape[0], dtype=np.int32)
    for frame_idx, raw_text in enumerate(values):
        text = _normalize_text_value(raw_text)
        if text not in text_to_index:
            text_to_index[text] = len(deduped)
            deduped.append(text)
        indices[frame_idx] = text_to_index[text]
    return _make_text_array(deduped), indices


def _seconds_to_ms(value: str) -> int:
    return int(float(value) * 1000.0)


def load_narration_rows(csv_path: str | Path) -> list[NarrationRow]:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return []
    rows: list[NarrationRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        text_columns = [name for name in (reader.fieldnames or []) if name.startswith("Describe my")]
        if not text_columns:
            return []
        text_column = text_columns[0]
        for row in reader:
            rows.append(
                NarrationRow(
                    start_time_ms=_seconds_to_ms(row["start_time"]),
                    end_time_ms=_seconds_to_ms(row["end_time"]),
                    text=_normalize_text_value(row[text_column]),
                )
            )
    return rows


def _align_rows_to_frames(rows: list[NarrationRow], frame_timestamps: np.ndarray) -> np.ndarray:
    values = np.full(frame_timestamps.shape[0], UNKNOWN_TEXT, dtype=object)
    for row in rows:
        mask = (frame_timestamps >= row.start_time_ms) & (frame_timestamps <= row.end_time_ms)
        values[mask] = row.text
    return values


def build_text_payload(
    *,
    sequence_dir: str | Path,
    frame_timestamps: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sequence_dir = Path(sequence_dir)
    frame_timestamps = np.asarray(frame_timestamps, dtype=np.int64).reshape(-1)
    activity_rows = load_narration_rows(sequence_dir / "narration" / "activity_summarization.csv")
    atomic_rows = load_narration_rows(sequence_dir / "narration" / "atomic_action.csv")

    main_values = np.full(frame_timestamps.shape[0], UNKNOWN_TEXT, dtype=object)
    sub_values = _align_rows_to_frames(activity_rows, frame_timestamps)
    action_values = _align_rows_to_frames(atomic_rows, frame_timestamps)
    interaction_values = np.full(frame_timestamps.shape[0], UNKNOWN_TEXT, dtype=object)

    main_texts, main_indices = _build_index_sequence(main_values)
    sub_texts, sub_indices = _build_index_sequence(sub_values)
    action_texts, action_indices = _build_index_sequence(action_values)
    interaction_texts, interaction_indices = _build_index_sequence(interaction_values)

    summary = {
        "mvnx_start_ms": int(frame_timestamps[0]) if frame_timestamps.size else None,
        "mvnx_end_ms": int(frame_timestamps[-1]) if frame_timestamps.size else None,
        "activity_start_ms": min((row.start_time_ms for row in activity_rows), default=None),
        "activity_end_ms": max((row.end_time_ms for row in activity_rows), default=None),
        "atomic_action_start_ms": min((row.start_time_ms for row in atomic_rows), default=None),
        "atomic_action_end_ms": max((row.end_time_ms for row in atomic_rows), default=None),
        "activity_covered_frames": int(np.count_nonzero(sub_indices)),
        "atomic_action_covered_frames": int(np.count_nonzero(action_indices)),
    }

    return (
        {
            "main_task_texts": main_texts,
            "sub_task_texts": sub_texts,
            "current_action_texts": action_texts,
            "interaction_texts": interaction_texts,
            "main_task_text_indices": main_indices,
            "sub_task_text_indices": sub_indices,
            "current_action_text_indices": action_indices,
            "interaction_text_indices": interaction_indices,
        },
        summary,
    )


def build_annotation_payload(
    sequence_dir: str | Path = DEFAULT_SEQUENCE_DIR,
    *,
    start_frame: int = 0,
    end_frame: int = 1000,
    stride: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sequence_dir = Path(sequence_dir)
    motion = load_mvnx_motion(
        sequence_dir / "body_xdata_mvnx",
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
    )
    text_payload, summary = build_text_payload(
        sequence_dir=sequence_dir,
        frame_timestamps=motion.frame_timestamps,
    )
    payload = {
        "fps": np.asarray(int(round(float(motion.fps))), dtype=np.int32),
        "num_frames": np.asarray(motion.num_frames, dtype=np.int32),
        "timeline_frame_indices": np.asarray(motion.frame_indices, dtype=np.int32),
        "frame_timestamps": np.asarray(motion.frame_timestamps, dtype=np.int64),
        **text_payload,
    }
    return payload, summary


def save_annotation_payload(
    payload: dict[str, np.ndarray],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return output_path
