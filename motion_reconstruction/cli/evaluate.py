"""重构评估命令行入口。"""

from __future__ import annotations

from pathlib import Path

from motion_reconstruction.cli.common import ChineseArgumentParser
from motion_reconstruction.config import load_config
from motion_reconstruction.evaluation import reconstruct_motion


def main() -> None:
    """生成当前帧重构误差和可复用 npz。"""
    parser = ChineseArgumentParser(description="评估 motion reconstruction checkpoint。")
    parser.add_argument("--config", required=True, help="训练使用的 YAML 配置文件路径。")
    parser.add_argument("--checkpoint", required=True, help="checkpoint 路径，通常是 latest.pt。")
    parser.add_argument("--output", required=True, help="评估输出目录。")
    parser.add_argument("--device", default="cpu", help="评估设备。")
    parser.add_argument("--batch-size", type=int, default=4096, help="评估 batch 大小。")
    parser.add_argument("--max-frames", type=int, default=None, help="最多评估多少个中心帧。")
    parser.add_argument("--no-npz", action="store_true", help="只保存 metrics.json，不保存 reconstruction.npz。")
    args = parser.parse_args()

    config = load_config(args.config)
    result = reconstruct_motion(
        config=config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
    )

    output_dir = Path(args.output)
    metrics_path = result.save_metrics_json(output_dir / "metrics.json")
    print(f"已保存评估指标: {metrics_path}")
    for name, value in result.metrics().items():
        print(f"{name}: {value:.8f}")
    if not args.no_npz:
        npz_path = result.save_npz(output_dir / "reconstruction.npz")
        print(f"已保存重构数据: {npz_path}")


if __name__ == "__main__":
    main()
