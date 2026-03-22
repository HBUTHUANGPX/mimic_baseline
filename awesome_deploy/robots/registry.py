from __future__ import annotations

import os
import sys

from awesome_deploy.robots.base import BaseRobotCfg
from awesome_deploy.robots.g1 import G1RobotCfg


ROBOT_CFG_REGISTRY = {
    "g1": G1RobotCfg,
}


def resolve_robot_name(
    argv: list[str] | None = None,
    env_var: str = "AWESOME_DEPLOY_ROBOT_NAME",
    default: str = "g1",
) -> str:
    argv = argv if argv is not None else sys.argv
    for index, arg in enumerate(argv):
        if arg.startswith("robot_name="):
            return arg.split("=", 1)[1]
        if arg.startswith("--robot_name="):
            return arg.split("=", 1)[1]
        if arg == "--robot_name" and index + 1 < len(argv):
            return argv[index + 1]

    return os.getenv(env_var, default)


def build_robot_cfg(robot_name: str, root_dir: str | None = None) -> BaseRobotCfg:
    try:
        cfg_cls = ROBOT_CFG_REGISTRY[robot_name]
    except KeyError as exc:
        supported = ", ".join(sorted(ROBOT_CFG_REGISTRY))
        raise ValueError(
            f"Unknown robot_name '{robot_name}'. Supported robots: {supported}"
        ) from exc
    return cfg_cls(root_dir=root_dir)
