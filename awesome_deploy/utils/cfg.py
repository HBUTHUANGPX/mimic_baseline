import os

from awesome_deploy.robots import build_robot_cfg, resolve_robot_name

current_path = os.getcwd()
cfg = build_robot_cfg(resolve_robot_name(), root_dir=current_path)


def get_robot_cfg(robot_name=None, root_dir=None):
    resolved_robot_name = robot_name or resolve_robot_name()
    return build_robot_cfg(resolved_robot_name, root_dir=root_dir or current_path)
