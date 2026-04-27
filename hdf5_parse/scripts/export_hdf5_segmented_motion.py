from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _bootstrap import ensure_repo_root_on_sys_path

ensure_repo_root_on_sys_path(__file__)

from hdf5_parse.motion_export.segmented import (
    DEFAULT_SMPL_OUTPUT_DIR,
    DEFAULT_SOMA_BVH_OUTPUT_DIR,
    export_segmented_smpl_npz,
    export_segmented_soma_bvh,
)
from hdf5_parse.utils.smpl_motion_tools import DEFAULT_HDF5_PATH


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export valid HDF5 motion segments to SMPL npz files and SOMA BVH files."
    )
    parser.add_argument("--hdf5-path", type=Path, default=DEFAULT_HDF5_PATH)
    parser.add_argument("--smpl-output-dir", type=Path, default=DEFAULT_SMPL_OUTPUT_DIR)
    parser.add_argument("--soma-bvh-output-dir", type=Path, default=DEFAULT_SOMA_BVH_OUTPUT_DIR)
    parser.add_argument("--soma-x-root", type=Path, default=Path("/home/hpx/HPX_LOCO_2/SOMA-X"))
    parser.add_argument("--smpl-model-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--filename-prefix", default="annotation")
    parser.add_argument(
        "--exports",
        nargs="+",
        choices=("smpl", "soma-bvh"),
        default=["smpl", "soma-bvh"],
        help="Which segmented assets to export. SOMA BVH remains a separate CUDA step.",
    )
    parser.add_argument(
        "--smpl-frame",
        choices=("soma_y_up", "raw"),
        default="soma_y_up",
        help="Coordinate frame used when saving segmented SMPL npz files. SOMA/BVH downstream uses soma_y_up.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    smpl_paths = []
    soma_bvh_paths = []
    if "smpl" in args.exports:
        smpl_paths = export_segmented_smpl_npz(
            hdf5_path=args.hdf5_path,
            smpl_output_dir=args.smpl_output_dir,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            stride=args.stride,
            filename_prefix=args.filename_prefix,
            smpl_frame=args.smpl_frame,
        )
        print(f"Saved {len(smpl_paths)} SMPL files to {args.smpl_output_dir}")
    if "soma-bvh" in args.exports:
        soma_bvh_paths = export_segmented_soma_bvh(
            hdf5_path=args.hdf5_path,
            soma_bvh_output_dir=args.soma_bvh_output_dir,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            stride=args.stride,
            device=args.device,
            batch_size=args.batch_size,
            soma_x_root=args.soma_x_root,
            smpl_model_path=args.smpl_model_path,
            filename_prefix=args.filename_prefix,
        )
        print(f"Saved {len(soma_bvh_paths)} SOMA BVH files to {args.soma_bvh_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
