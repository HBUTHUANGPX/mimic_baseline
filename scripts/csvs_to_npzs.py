"""This script replays motions from all CSV files in a specified input folder and outputs them to NPZ files in the output folder.

.. code-block:: bash

    # Usage
    python csv_to_npz.py --input_folder /path/to/csv_folder --output_folder /path/to/npz_folder --input_fps 30 --frame_range 122 722 --output_fps 50
"""

import argparse
import numpy as np
import torch
import os

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(
    description="Replay motions from all CSV files in the input folder and output to NPZ files in the output folder."
)
parser.add_argument(
    "--input_folder",
    type=str,
    required=True,
    help="The path to the input folder containing CSV motion files.",
)
parser.add_argument(
    "--output_folder",
    type=str,
    required=True,
    help="The path to the output folder for saving NPZ files.",
)
parser.add_argument(
    "--input_fps", type=int, default=30, help="The fps of the input motion."
)
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded for each file."
    ),
    default=None,
)
parser.add_argument(
    "--output_fps", type=int, default=50, help="The fps of the output motion."
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now we are ready!
print("[INFO]: Setup complete...")

from isaaclab.sim import SimulationContext
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_slerp,
)
##
# Pre-defined configs
##
from general_motion_tracker_whole_body_teleoperation.robots.q1 import Q1_CYLINDER_CFG

@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = Q1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the csv file."""
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self.motion_base_rots_input[
            :, [3, 0, 1, 2]
        ]  # convert to wxyz
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(
            f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}"
        )

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(
            0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
        )
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(
        self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
    ) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(
        self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
    ) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(
            self.motion_base_poss, spacing=self.output_dt, dim=0
        )[0]
        self.motion_dof_vels = torch.gradient(
            self.motion_dof_poss, spacing=self.output_dt, dim=0
        )[0]
        self.motion_base_ang_vels = self._so3_derivative(
            self.motion_base_rots, self.output_dt
        )

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat(
            [omega[:1], omega, omega[-1:]], dim=0
        )  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def process_single_motion(
    sim: SimulationContext,
    scene: InteractiveScene,
    joint_names: list[str],
    motion_file: str,
    output_file: str,
    input_fps: int,
    output_fps: int,
    frame_range: tuple[int, int] | None,
):
    """Processes a single motion: replays it in simulation and saves to NPZ."""
    # Load motion
    motion = MotionLoader(
        motion_file=motion_file,
        input_fps=input_fps,
        output_fps=output_fps,
        device=sim.device,
        frame_range=frame_range,
    )

    # Extract scene entities
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    # Data logger
    log = {
        "fps": [output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    # Reset simulation for this motion
    sim.reset()
    scene.reset()

    file_saved = False
    while not file_saved and simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # Set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # Set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()  # We don't want physics (sim.step())
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([3.0, 3.0, 0.5]), pos_lookat)

        log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
        log["body_lin_vel_w"].append(
            robot.data.body_lin_vel_w[0, :].cpu().numpy().copy()
        )
        log["body_ang_vel_w"].append(
            robot.data.body_ang_vel_w[0, :].cpu().numpy().copy()
        )

        if reset_flag:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            np.savez(output_file, **log)
            print(f"[INFO]: Motion saved to {output_file}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)

    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")

    # Collect all CSV files recursively
    csv_files = []
    for root, _, files in os.walk(args_cli.input_folder):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        print("[WARNING]: No CSV files found in the input folder.")
    else:
        print(f"[INFO]: Found {len(csv_files)} CSV files to process.")

    # Process each CSV file sequentially
    joint_names = [
        "L_hip_roll_joint",
        "L_hip_yaw_joint",
        "L_hip_pitch_joint",
        "L_knee_joint",
        "L_ankle_pitch_joint",
        "L_ankle_roll_joint",
        "R_hip_roll_joint",
        "R_hip_yaw_joint",
        "R_hip_pitch_joint",
        "R_knee_joint",
        "R_ankle_pitch_joint",
        "R_ankle_roll_joint",
        "pelvis_joint",
        "L_shoulder_pitch_joint",
        "L_shoulder_roll_joint",
        "L_shoulder_yaw_joint",
        "L_elbow_joint",
        "L_forearm_yaw_joint",
        "L_wrist_roll_joint",
        "L_wrist_pitch_joint",
        "R_shoulder_pitch_joint",
        "R_shoulder_roll_joint",
        "R_shoulder_yaw_joint",
        "R_elbow_joint",
        "R_forearm_yaw_joint",
        "R_wrist_roll_joint",
        "R_wrist_pitch_joint",
        "head_yaw_joint",
        "head_pitch_joint",
    ]

    for csv_path in csv_files:
        rel_path = os.path.relpath(csv_path, args_cli.input_folder)
        npz_path = os.path.join(args_cli.output_folder, rel_path.replace(".csv", ".npz"))
        print(f"[INFO]: Processing {csv_path} -> {npz_path}")
        process_single_motion(
            sim=sim,
            scene=scene,
            joint_names=joint_names,
            motion_file=csv_path,
            output_file=npz_path,
            input_fps=args_cli.input_fps,
            output_fps=args_cli.output_fps,
            frame_range=args_cli.frame_range,
        )

    print("[INFO]: All motions processed.!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # Close sim app


if __name__ == "__main__":
    # Run the main function
    main()
    simulation_app.close()
