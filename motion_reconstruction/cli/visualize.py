"""MuJoCo 重构可视化命令行入口。"""

from __future__ import annotations

from motion_reconstruction.cli.common import ChineseArgumentParser
from motion_reconstruction.config import load_config
from motion_reconstruction.evaluation import reconstruct_motion
from motion_reconstruction.visualization import play_reconstruction


def main() -> None:
    """加载 checkpoint 并启动 MuJoCo viewer。"""
    parser = ChineseArgumentParser(description="使用 MuJoCo 播放原始动作和重构动作。")
    parser.add_argument("--config", required=True, help="训练使用的 YAML 配置文件路径。")
    parser.add_argument("--checkpoint", required=True, help="checkpoint 路径，通常是 latest.pt。")
    parser.add_argument("--xml-path", required=True, help="机器人 MuJoCo XML 路径。")
    parser.add_argument("--device", default="cpu", help="重构推理设备。")
    parser.add_argument("--batch-size", type=int, default=4096, help="重构推理 batch 大小。")
    parser.add_argument("--max-frames", type=int, default=None, help="最多播放多少个中心帧。")
    parser.add_argument("--pair", choices=["robot", "human", "both"], default="both", help="播放哪组对比。")
    parser.add_argument("--fps", type=int, default=None, help="覆盖播放帧率。")
    parser.add_argument("--loop", action="store_true", help="循环播放。")
    parser.add_argument("--keep-world", action="store_true", help="保留世界坐标轨迹，不按 anchor 居中显示。")
    args = parser.parse_args()

    config = load_config(args.config)
    result = reconstruct_motion(
        config=config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
    )
    play_reconstruction(
        result=result,
        xml_path=args.xml_path,
        pair=args.pair,
        fps=args.fps,
        loop=args.loop,
        keep_world=args.keep_world,
    )


if __name__ == "__main__":
    main()
