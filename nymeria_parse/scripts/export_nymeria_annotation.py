from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from nymeria_parse.motion_export.core import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SEQUENCE_DIR,
    build_annotation_payload,
    save_annotation_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Nymeria timeline/text annotation payload.")
    parser.add_argument("--sequence-dir", default=str(DEFAULT_SEQUENCE_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload, summary = build_annotation_payload(
        Path(args.sequence_dir),
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
    )
    output_path = save_annotation_payload(payload, args.output)
    print(f"Saved {int(payload['num_frames'])} frames to {output_path}")
    print(
        "Text coverage: "
        f"activity={summary['activity_covered_frames']} frames, "
        f"atomic_action={summary['atomic_action_covered_frames']} frames"
    )


if __name__ == "__main__":
    main()
