"""复用 motion_reconstruction 可视化由 SOMA BVH 转出的 human motion npz。"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_reconstruction.cli.common import ChineseArgumentParser
from motion_reconstruction.config import load_config
from motion_reconstruction.visualization import visualize_hdf5_human_npz


def build_arg_parser() -> ChineseArgumentParser:
    parser = ChineseArgumentParser(description="可视化 human motion npz，并走 human encoder -> decoder。")
    parser.add_argument("--config", required=True, help="motion_reconstruction 使用的 YAML 配置文件路径。")
    parser.add_argument("--checkpoint", required=True, help="motion_reconstruction checkpoint 路径。")
    parser.add_argument("--xml-path", required=True, help="机器人 MuJoCo XML 路径。")
    parser.add_argument(
        "--motion-npz",
        type=Path,
        required=True,
        help="human motion npz 路径，通常来自 SOMA BVH 经过 bvh_to_csv_converter.py 的输出。",
    )
    parser.add_argument("--device", default="cpu", help="重构推理设备。")
    parser.add_argument("--batch-size", type=int, default=4096, help="重构推理 batch 大小。")
    parser.add_argument("--max-frames", type=int, default=None, help="最多播放多少个中心帧。")
    parser.add_argument("--fps", type=int, default=None, help="覆盖播放帧率。")
    parser.add_argument("--loop", action="store_true", help="循环播放。")
    parser.add_argument("--keep-world", action="store_true", help="保留世界坐标轨迹，不按 anchor 居中显示。")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    visualize_hdf5_human_npz(
        config=config,
        checkpoint_path=args.checkpoint,
        xml_path=args.xml_path,
        motion_npz=args.motion_npz,
        device=args.device,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        fps=args.fps,
        loop=args.loop,
        keep_world=args.keep_world,
    )


if __name__ == "__main__":
    main()
