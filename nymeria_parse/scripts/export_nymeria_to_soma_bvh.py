from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from hdf5_parse.motion_export.smpl_soma import DEFAULT_SOMA_X_ROOT  # noqa: E402
from nymeria_parse.motion_export.soma_bvh import (  # noqa: E402
    DEFAULT_SOMA_BATCH_SIZE,
    DEFAULT_SEQUENCE_DIR,
    DEFAULT_SOMA_BVH_OUTPUT_PATH,
    export_nymeria_to_soma_bvh,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Nymeria MVNX body motion to SOMA BVH.")
    parser.add_argument("--sequence-dir", default=str(DEFAULT_SEQUENCE_DIR))
    parser.add_argument("--output", default=str(DEFAULT_SOMA_BVH_OUTPUT_PATH))
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_SOMA_BATCH_SIZE,
        help="SOMA inversion CUDA batch size. Lower it if full-sequence export still runs out of memory.",
    )
    parser.add_argument("--soma-x-root", default=str(DEFAULT_SOMA_X_ROOT))
    parser.add_argument("--smpl-model-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = export_nymeria_to_soma_bvh(
        Path(args.sequence_dir),
        output_path=args.output,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
        device=args.device,
        batch_size=args.batch_size,
        soma_x_root=args.soma_x_root,
        smpl_model_path=args.smpl_model_path,
    )
    print(f"Saved SOMA BVH to {output_path}")


if __name__ == "__main__":
    main()
