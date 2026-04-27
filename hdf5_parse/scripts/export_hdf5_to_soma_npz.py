from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _bootstrap import ensure_repo_root_on_sys_path

ensure_repo_root_on_sys_path(__file__)

from hdf5_parse.motion_export.core import (
    export_hdf5_to_soma_payload,
    save_hdf5_soma_payload,
)
from hdf5_parse.utils.smpl_motion_tools import DEFAULT_HDF5_PATH


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export annotation.hdf5 captions and valid frame timeline metadata to annotation_soma.npz."
    )
    parser.add_argument("--hdf5-path", type=Path, default=DEFAULT_HDF5_PATH)
    parser.add_argument("--output-path", type=Path, default=Path("hdf5_parse/out/annotation_soma.npz"))
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = export_hdf5_to_soma_payload(
        hdf5_path=args.hdf5_path,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
    )
    save_hdf5_soma_payload(payload, args.output_path)
    print(f"Saved {payload['num_frames'].item()} frames to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
