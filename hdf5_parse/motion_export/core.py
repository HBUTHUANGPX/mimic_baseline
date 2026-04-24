from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..utils.smpl_motion_tools import DEFAULT_HDF5_PATH
from .smpl_soma import (
    BodyFrameSelection,
    DEFAULT_SOMA_X_ROOT,
    SMPLBodyMotion,
    build_body_valid_mask,
    build_frame_timestamp_lookup,
    load_body_frame_selection,
)


UNKNOWN_TEXT = "UNKNOWN"
DEFAULT_OUTPUT_PATH = Path("hdf5_parse/out/annotation_soma.npz")


def load_caption_json(hdf5_path: str | Path = DEFAULT_HDF5_PATH) -> dict[str, Any]:
    with h5py.File(Path(hdf5_path), "r") as h5_file:
        raw = h5_file["caption"][()]
    if isinstance(raw, bytes):
        raw = raw.decode()
    return json.loads(raw)


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


def align_caption_texts_to_frames(*, caption: dict[str, Any], frame_timestamps: np.ndarray) -> dict[str, np.ndarray]:
    frame_timestamps = np.asarray(frame_timestamps, dtype=np.int64).reshape(-1)

    main_task_values = np.full(frame_timestamps.shape[0], _normalize_text_value(caption["config"]["Main Task"]), dtype=object)
    sub_task_values = np.full(frame_timestamps.shape[0], UNKNOWN_TEXT, dtype=object)
    action_values = np.full(frame_timestamps.shape[0], UNKNOWN_TEXT, dtype=object)
    interaction_values = np.full(frame_timestamps.shape[0], UNKNOWN_TEXT, dtype=object)

    for segment in caption.get("segments", []):
        segment_start = int(segment["start_frame"])
        segment_end = int(segment["end_frame"])
        segment_mask = (frame_timestamps >= segment_start) & (frame_timestamps <= segment_end)
        sub_task_values[segment_mask] = _normalize_text_value(segment.get("Sub Task", UNKNOWN_TEXT))

        for action in segment.get("Current Action", []):
            action_start = int(action["start_frame"])
            action_end = int(action["end_frame"])
            action_mask = (frame_timestamps >= action_start) & (frame_timestamps <= action_end)
            action_values[action_mask] = _normalize_text_value(action.get("label", UNKNOWN_TEXT))

        interaction_items = sorted(
            ((int(timestamp), _normalize_text_value(text)) for timestamp, text in segment.get("interaction", {}).items()),
            key=lambda item: item[0],
        )
        for item_idx, (interaction_start, interaction_text) in enumerate(interaction_items):
            interaction_end = interaction_items[item_idx + 1][0] - 1 if item_idx + 1 < len(interaction_items) else segment_end
            interaction_mask = (frame_timestamps >= interaction_start) & (frame_timestamps <= interaction_end)
            interaction_values[interaction_mask] = interaction_text

    main_task_texts, main_task_indices = _build_index_sequence(main_task_values)
    sub_task_texts, sub_task_indices = _build_index_sequence(sub_task_values)
    action_texts, action_indices = _build_index_sequence(action_values)
    interaction_texts, interaction_indices = _build_index_sequence(interaction_values)

    return {
        "main_task_texts": main_task_texts,
        "sub_task_texts": sub_task_texts,
        "current_action_texts": action_texts,
        "interaction_texts": interaction_texts,
        "main_task_text_indices": main_task_indices,
        "sub_task_text_indices": sub_task_indices,
        "current_action_text_indices": action_indices,
        "interaction_text_indices": interaction_indices,
    }


def build_annotation_export_payload(
    *,
    fps: float,
    frame_nums: np.ndarray,
    frame_timestamps: np.ndarray,
    text_payload: dict[str, Any],
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    frame_nums = np.asarray(frame_nums, dtype=np.int32).reshape(-1)
    frame_timestamps = np.asarray(frame_timestamps, dtype=np.int64).reshape(-1)
    if frame_nums.shape != frame_timestamps.shape:
        raise ValueError(
            "frame_nums and frame_timestamps must have the same shape, "
            f"got {frame_nums.shape} and {frame_timestamps.shape}."
        )

    payload = {
        "fps": np.asarray(int(round(float(fps))), dtype=np.int32),
        "num_frames": np.asarray(frame_nums.shape[0], dtype=np.int32),
        "timeline_frame_indices": frame_nums,
        "frame_timestamps": frame_timestamps,
    }
    for key, value in {**text_payload, **(extra_payload or {})}.items():
        if isinstance(value, list) and value and isinstance(value[0], str):
            payload[key] = _make_text_array(list(value))
        elif isinstance(value, np.ndarray) and value.dtype == object:
            if value.ndim == 1 and all(isinstance(item, str) for item in value.tolist()):
                payload[key] = _make_text_array(value.tolist())
            else:
                payload[key] = value
        else:
            payload[key] = np.asarray(value)
    return payload


def export_hdf5_to_soma_payload(
    hdf5_path: str | Path = DEFAULT_HDF5_PATH,
    *,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
) -> dict[str, np.ndarray]:
    selection = load_body_frame_selection(
        hdf5_path,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
    )
    caption = load_caption_json(hdf5_path)
    text_payload = align_caption_texts_to_frames(caption=caption, frame_timestamps=selection.frame_timestamps)
    return build_annotation_export_payload(
        fps=selection.fps,
        frame_nums=selection.frame_nums,
        frame_timestamps=selection.frame_timestamps,
        text_payload=text_payload,
        extra_payload={
            "source_caption": np.asarray(json.dumps(caption, ensure_ascii=False)),
        },
    )


def save_hdf5_soma_payload(payload: dict[str, np.ndarray], output_path: str | Path = DEFAULT_OUTPUT_PATH) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return output_path
