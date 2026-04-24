from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _bootstrap import ensure_repo_root_on_sys_path

ensure_repo_root_on_sys_path(__file__)

from hdf5_parse.motion_export.bvh import DEFAULT_OUTPUT_BVH_PATH, save_hdf5_soma_bvh


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export annotation.hdf5 full-body mocap to a SOMA-style BVH file."
    )
    parser.add_argument("--hdf5-path", type=Path, default=Path("hdf5_parse/hdf5/annotation.hdf5"))
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_BVH_PATH)
    parser.add_argument("--soma-x-root", type=Path, default=Path("/home/hpx/HPX_LOCO_2/SOMA-X"))
    parser.add_argument("--smpl-model-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_path = save_hdf5_soma_bvh(
        hdf5_path=args.hdf5_path,
        output_path=args.output_path,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
        device=args.device,
        batch_size=args.batch_size,
        soma_x_root=args.soma_x_root,
        smpl_model_path=args.smpl_model_path,
    )
    print(f"Saved SOMA BVH to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
