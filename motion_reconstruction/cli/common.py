"""命令行入口共享的小工具。"""

from __future__ import annotations

import argparse


class ChineseArgumentParser(argparse.ArgumentParser):
    """把 argparse 默认帮助文本里的固定标签换成中文。"""

    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "用法:")

    def format_help(self) -> str:
        return (
            super()
            .format_help()
            .replace("usage:", "用法:")
            .replace("options:", "选项:")
            .replace("show this help message and exit", "显示帮助信息并退出。")
        )
