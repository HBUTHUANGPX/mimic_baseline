"""motion reconstruction 训练命令行入口。

CLI 只负责读取配置和覆盖少量运行参数；训练逻辑在
`MotionReconstructionTrainer` 中，便于其它工程复用。
"""

from __future__ import annotations

from motion_reconstruction.config import load_config
from motion_reconstruction.cli.common import ChineseArgumentParser
from motion_reconstruction.training.trainer import MotionReconstructionTrainer


def main() -> None:
    """解析命令行参数并启动训练。"""
    parser = ChineseArgumentParser(description="训练动作重构 FSQ/iFSQ 自编码器。")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径。")
    parser.add_argument("--device", default=None, help="覆盖 train.device。")
    parser.add_argument("--run-name", default=None, help="覆盖 output.run_name。")
    parser.add_argument("--no-progress", action="store_true", help="关闭 tqdm 进度条。")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device is not None:
        config.train.device = args.device
    if args.run_name is not None:
        config.output.run_name = args.run_name
    if args.no_progress:
        config.train.progress = False

    trainer = MotionReconstructionTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
