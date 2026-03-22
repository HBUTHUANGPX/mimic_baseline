from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path

from awesome_deploy.utils.urdf_graph import UrdfGraph


class BaseRobotCfg:
    robot_name = ""
    simulator_dt = 0.002
    policy_dt = 0.02
    frame_stack = 1
    action_clip = 10.0
    action_scale = 0.25
    motion_play = False
    group = {}
    asset_dirname = ""
    mjcf_filename = ""
    urdf_filename = ""
    motion_reference_body = ""
    motion_body_names = []

    leg_P_gains = []
    leg_D_gains = []
    leg_tq_max = []
    pelvis_P_gains = []
    pelvis_D_gains = []
    pelvis_tq_max = []
    arm_P_gains = []
    arm_D_gains = []
    arm_tq_max = []

    leg_default_pos = []
    pelvis_default_pos = []
    arm_default_pos = []

    def __init__(self, root_dir: str | None = None):
        self.root_dir = root_dir or os.getcwd()

    def _resolve_existing_path(self, *candidates: str) -> str:
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]

    def _search_by_name(self, target_name: str) -> str | None:
        root_path = Path(self.root_dir)
        for path in root_path.rglob(target_name):
            return str(path)
        return None

    @property
    def policy_path(self) -> str:
        policy_dir_name = os.path.basename(self.group["policy"])
        return self._resolve_existing_path(
            os.path.join(self.root_dir, self.group["policy"], "policy.onnx"),
            os.path.join(
                self.root_dir,
                "deploy_mujoco",
                "deploy_policy",
                self.robot_name,
                policy_dir_name,
                "policy.onnx",
            ),
            os.path.join(
                self.root_dir,
                "logs",
                "rsl_rl",
                f"{self.robot_name}_flat",
                policy_dir_name,
                "exported",
                "policy.onnx",
            ),
            self._search_by_name("policy.onnx") or "",
        )

    @property
    def motion_file(self) -> str:
        motion_filename = f"{self.group['motion']}.npz"
        return self._resolve_existing_path(
            os.path.join(self.root_dir, self.group["policy"], motion_filename),
            os.path.join(
                self.root_dir,
                "deploy_mujoco",
                "artifacts",
                self.robot_name,
                "xsens_bvh",
                "251203",
                motion_filename,
            ),
            os.path.join(
                self.root_dir,
                "artifacts",
                self.robot_name,
                "xsens_bvh",
                "251203",
                motion_filename,
            ),
            self._search_by_name(motion_filename) or "",
        )

    @property
    def asset_path(self) -> str:
        return self._resolve_existing_path(
            os.path.join(self.root_dir, "deploy", "assets", self.asset_dirname),
            os.path.join(
                self.root_dir,
                "general_motion_tracker_whole_body_teleoperation",
                "general_motion_tracker_whole_body_teleoperation",
                "assets",
                self.asset_dirname,
            ),
            os.path.join(self.root_dir, "artifacts", self.asset_dirname),
        )

    @property
    def mjcf_path(self) -> str:
        return self._resolve_existing_path(
            os.path.join(self.asset_path, self.mjcf_filename),
            self._search_by_name(self.mjcf_filename) or "",
        )

    @property
    def urdf_path(self) -> str:
        return self._resolve_existing_path(
            os.path.join(self.asset_path, self.urdf_filename),
            self._search_by_name(self.urdf_filename) or "",
        )

    @cached_property
    def urdf_graph(self) -> UrdfGraph:
        return UrdfGraph(self.urdf_path)

    @property
    def isaac_sim_joint_name(self):
        return self.urdf_graph.bfs_joint_order()

    @property
    def isaac_sim_link_name(self):
        return self.urdf_graph.bfs_link_order()
