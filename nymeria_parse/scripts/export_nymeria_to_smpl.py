from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from nymeria_parse.motion_export.smpl import (  # noqa: E402
    DEFAULT_SEQUENCE_DIR,
    DEFAULT_SMPL_OUTPUT_PATH,
    build_smpl_motion_payload,
    save_smpl_motion_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Nymeria MVNX body motion to SMPL npz.")
    parser.add_argument("--sequence-dir", default=str(DEFAULT_SEQUENCE_DIR))
    parser.add_argument("--output", default=str(DEFAULT_SMPL_OUTPUT_PATH))
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_smpl_motion_payload(
        Path(args.sequence_dir),
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
    )
    output_path = save_smpl_motion_npz(payload, args.output)
    print(f"Saved {payload['global_orient'].shape[0]} SMPL frames to {output_path}")


if __name__ == "__main__":
    main()
