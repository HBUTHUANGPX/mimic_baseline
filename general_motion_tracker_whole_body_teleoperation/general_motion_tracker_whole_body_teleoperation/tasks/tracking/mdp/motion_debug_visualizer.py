from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from general_motion_tracker_whole_body_teleoperation.tasks.tracking.mdp.commands import (
        MotionCommand,
        MotionCommandCfg,
    )


class MotionDebugVisualizer:
    def __init__(self, cfg: MotionCommandCfg):
        self.cfg = cfg
        self._robot_visualizers_created = False
        self._human_visualizers_created = False

    def set_visibility(self, visible: bool):
        if visible:
            self._ensure_robot_visualizers()
            self._ensure_human_visualizers()

        if self._robot_visualizers_created:
            self.current_anchor_visualizer.set_visibility(visible)
            self.goal_anchor_visualizer.set_visibility(visible)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(visible)
                self.goal_body_visualizers[i].set_visibility(visible)

        if self._human_visualizers_created:
            self.human_goal_anchor_visualizer.set_visibility(visible)
            for i in range(len(self.cfg.desire_human_joint_names)):
                self.human_goal_body_visualizers[i].set_visibility(visible)

    def visualize(self, command: MotionCommand):
        if not self._robot_visualizers_created or not self._human_visualizers_created:
            return
        if not hasattr(command, "robot") or not command.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(
            command.sim_robot_anchor_pos_w, command.sim_robot_anchor_quat_w
        )
        self.goal_anchor_visualizer.visualize(
            command.ref_robot_anchor_pos_w, command.ref_robot_anchor_quat_w
        )
        self.human_goal_anchor_visualizer.visualize(
            command.ref_human_anchor_pos_w, command.ref_human_anchor_quat_w
        )

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(
                command.sim_robot_body_pos_w[:, i],
                command.sim_robot_body_quat_w[:, i],
            )
            self.goal_body_visualizers[i].visualize(
                command.yaw_aligned_ref_robot_body_pos_w[:, i],
                command.yaw_aligned_ref_robot_body_quat_w[:, i],
            )

        for i in range(len(self.cfg.desire_human_joint_names)):
            self.human_goal_body_visualizers[i].visualize(
                command.ref_human_body_pos_w[:, i],
                command.ref_human_body_quat_w[:, i],
            )

    def _ensure_robot_visualizers(self):
        if self._robot_visualizers_created:
            return

        self.current_anchor_visualizer = VisualizationMarkers(
            self.cfg.anchor_visualizer_cfg.replace(
                prim_path="/Visuals/Command/current/anchor"
            )
        )
        self.goal_anchor_visualizer = VisualizationMarkers(
            self.cfg.anchor_visualizer_cfg.replace(
                prim_path="/Visuals/Command/goal/anchor"
            )
        )

        self.current_body_visualizers = []
        self.goal_body_visualizers = []
        for name in self.cfg.body_names:
            self.current_body_visualizers.append(
                VisualizationMarkers(
                    self.cfg.body_visualizer_cfg.replace(
                        prim_path="/Visuals/Command/current/" + name
                    )
                )
            )
            self.goal_body_visualizers.append(
                VisualizationMarkers(
                    self.cfg.body_visualizer_cfg.replace(
                        prim_path="/Visuals/Command/goal/" + name
                    )
                )
            )

        self._robot_visualizers_created = True

    def _ensure_human_visualizers(self):
        if self._human_visualizers_created:
            return

        self.human_goal_anchor_visualizer = VisualizationMarkers(
            self.cfg.human_anchor_visualizer_cfg.replace(
                prim_path="/Visuals/Command/current/human_goal_anchor"
            )
        )
        self.human_goal_body_visualizers = []
        for name in self.cfg.desire_human_joint_names:
            self.human_goal_body_visualizers.append(
                VisualizationMarkers(
                    self.cfg.human_body_visualizer_cfg.replace(
                        prim_path="/Visuals/Command/human_goal_body/" + name
                    )
                )
            )

        self._human_visualizers_created = True
