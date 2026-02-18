"""This script replays motions from all CSV files in a specified input folder and outputs them to NPZ files in the output folder.

.. code-block:: bash

    # Usage
    python csv_to_npz.py --input_folder /path/to/csv_folder --output_folder /path/to/npz_folder --input_fps 30 --frame_range 122 722 --output_fps 50
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import os
import time
import concurrent.futures

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
parser.add_argument(
    "--num_envs",
    type=int,
    default=1600,
    help="Number of parallel environments to process motions.",
)
parser.add_argument(
    "--async_save_npz",
    action="store_true",
    help="Save .npz files asynchronously to avoid blocking the main loop.",
)
parser.add_argument(
    "--npz_save_workers",
    type=int,
    default=8,
    help="Number of background workers for async .npz saving.",
)
parser.add_argument(
    "--preload_csv",
    action="store_true",
    help="Preload all CSV files into memory before simulation.",
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

def build_traj_module(
    fps: int,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
) -> nn.Module:
    mod = nn.Module()
    mod.register_buffer("fps", torch.tensor([fps], dtype=torch.int32))
    mod.register_buffer("joint_pos", joint_pos)
    mod.register_buffer("joint_vel", joint_vel)
    mod.register_buffer("body_pos_w", body_pos_w)
    mod.register_buffer("body_quat_w", body_quat_w)
    mod.register_buffer("body_lin_vel_w", body_lin_vel_w)
    mod.register_buffer("body_ang_vel_w", body_ang_vel_w)
    return mod

def traj_module_to_numpy_dict(mod: nn.Module) -> dict[str, np.ndarray]:
    return {
        "fps": mod.fps.cpu().numpy(),
        "joint_pos": mod.joint_pos.cpu().numpy(),
        "joint_vel": mod.joint_vel.cpu().numpy(),
        "body_pos_w": mod.body_pos_w.cpu().numpy(),
        "body_quat_w": mod.body_quat_w.cpu().numpy(),
        "body_lin_vel_w": mod.body_lin_vel_w.cpu().numpy(),
        "body_ang_vel_w": mod.body_ang_vel_w.cpu().numpy(),
    }

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
        motion_data: torch.Tensor | None = None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self.motion_data = motion_data
        self.timing = {
            "load_motion": 0.0,
            "interpolate_motion": 0.0,
            "compute_velocities": 0.0,
            "slerp": 0.0,
            "lerp": 0.0,
            "compute_blend": 0.0,
            "so3_derivative": 0.0,
        }
        t0 = time.perf_counter()
        self._load_motion()
        self.timing["load_motion"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self._interpolate_motion()
        self.timing["interpolate_motion"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self._compute_velocities()
        self.timing["compute_velocities"] = time.perf_counter() - t0
        print(
            "[TIMING][MotionLoader]: "
            f"load={self.timing['load_motion']:.3f}s "
            f"interp={self.timing['interpolate_motion']:.3f}s "
            f"vel={self.timing['compute_velocities']:.3f}s "
            f"slerp={self.timing['slerp']:.3f}s "
            f"lerp={self.timing['lerp']:.3f}s "
            f"blend={self.timing['compute_blend']:.3f}s "
            f"so3={self.timing['so3_derivative']:.3f}s"
        )

    def _load_motion(self):
        """Loads the motion from the csv file."""
        if self.motion_data is not None:
            motion = self.motion_data
        else:
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
        t0 = time.perf_counter()
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.timing["compute_blend"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.timing["lerp"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.timing["slerp"] += time.perf_counter() - t0
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
        # Batch slerp to avoid Python loop bottleneck
        eps = 1e-8
        a_norm = a / (a.norm(dim=-1, keepdim=True) + eps)
        b_norm = b / (b.norm(dim=-1, keepdim=True) + eps)
        cos_theta = (a_norm * b_norm).sum(dim=-1)
        flip = cos_theta < 0.0
        b_norm = torch.where(flip.unsqueeze(-1), -b_norm, b_norm)
        cos_theta = cos_theta.abs()

        # Use nlerp for very small angles for numerical stability
        use_lerp = cos_theta > 0.9995
        theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
        sin_theta = torch.sin(theta) + eps
        w0 = torch.sin((1.0 - blend) * theta) / sin_theta
        w1 = torch.sin(blend * theta) / sin_theta

        slerp = (w0.unsqueeze(-1) * a_norm) + (w1.unsqueeze(-1) * b_norm)
        lerp = a_norm + blend.unsqueeze(-1) * (b_norm - a_norm)
        lerp = lerp / (lerp.norm(dim=-1, keepdim=True) + eps)
        return torch.where(use_lerp.unsqueeze(-1), lerp, slerp)

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
        t0 = time.perf_counter()
        self.motion_base_ang_vels = self._so3_derivative(
            self.motion_base_rots, self.output_dt
        )
        self.timing["so3_derivative"] += time.perf_counter() - t0

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


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)

    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
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

    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    pending_files = list(csv_files)
    timing = {
        "load_motion": 0.0,
        "step_total": 0.0,
        "write_state": 0.0,
        "render_update": 0.0,
        "log_copy": 0.0,
        "save_npz": 0.0,
        "preload_csv": 0.0,
        "save_submit": 0.0,
        "steps": 0,
        "motions": 0,
    }

    save_executor = None
    save_futures = []
    if args_cli.async_save_npz:
        save_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=args_cli.npz_save_workers
        )
    trajectory_modules = nn.ModuleList()

    preloaded_csv = {}
    if args_cli.preload_csv:
        t_preload = time.perf_counter()
        for csv_path in csv_files:
            if args_cli.frame_range is None:
                data = np.loadtxt(csv_path, delimiter=",")
            else:
                data = np.loadtxt(
                    csv_path,
                    delimiter=",",
                    skiprows=args_cli.frame_range[0] - 1,
                    max_rows=args_cli.frame_range[1] - args_cli.frame_range[0] + 1,
                )
            preloaded_csv[csv_path] = torch.from_numpy(data)
        timing["preload_csv"] = time.perf_counter() - t_preload

    def start_next_motion(env_id: int):
        if not pending_files:
            return None
        csv_path = pending_files.pop(0)
        rel_path = os.path.relpath(csv_path, args_cli.input_folder)
        npz_path = os.path.join(
            args_cli.output_folder, rel_path.replace(".csv", ".npz")
        )
        print(f"[INFO]: Enqueue {csv_path} -> {npz_path} on env '{env_id}'")
        t0 = time.perf_counter()
        motion = MotionLoader(
            motion_file=csv_path,
            input_fps=args_cli.input_fps,
            output_fps=args_cli.output_fps,
            device=sim.device,
            frame_range=args_cli.frame_range,
            motion_data=preloaded_csv.get(csv_path),
        )
        timing["load_motion"] += time.perf_counter() - t0
        timing["motions"] += 1
        num_frames = motion.output_frames
        num_joints = robot.data.joint_pos.shape[1]
        num_bodies = robot.data.body_pos_w.shape[1]
        bufs = {
            "joint_pos": torch.empty(
                (num_frames, num_joints), device=sim.device, dtype=robot.data.joint_pos.dtype
            ),
            "joint_vel": torch.empty(
                (num_frames, num_joints), device=sim.device, dtype=robot.data.joint_vel.dtype
            ),
            "body_pos_w": torch.empty(
                (num_frames, num_bodies, 3),
                device=sim.device,
                dtype=robot.data.body_pos_w.dtype,
            ),
            "body_quat_w": torch.empty(
                (num_frames, num_bodies, 4),
                device=sim.device,
                dtype=robot.data.body_quat_w.dtype,
            ),
            "body_lin_vel_w": torch.empty(
                (num_frames, num_bodies, 3),
                device=sim.device,
                dtype=robot.data.body_lin_vel_w.dtype,
            ),
            "body_ang_vel_w": torch.empty(
                (num_frames, num_bodies, 3),
                device=sim.device,
                dtype=robot.data.body_ang_vel_w.dtype,
            ),
        }
        return {
            "env_id": env_id,
            "csv_path": csv_path,
            "npz_path": npz_path,
            "motion": motion,
            "bufs": bufs,
            "frame_idx": 0,
            "num_frames": num_frames,
        }

    env_slots = [start_next_motion(i) for i in range(args_cli.num_envs)]

    sim.reset()
    scene.reset()

    def any_active():
        return any(slot is not None for slot in env_slots)

    while any_active() and simulation_app.is_running():
        t_step_start = time.perf_counter()
        root_states = robot.data.default_root_state.clone()
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()

        for slot in env_slots:
            if slot is None:
                continue
            env_id = slot["env_id"]
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
            ) = slot["motion"].get_next_state()

            root_states[env_id, :3] = motion_base_pos[0]
            root_states[env_id, :2] += scene.env_origins[env_id, :2]
            root_states[env_id, 3:7] = motion_base_rot[0]
            root_states[env_id, 7:10] = motion_base_lin_vel[0]
            root_states[env_id, 10:] = motion_base_ang_vel[0]

            joint_pos[env_id, robot_joint_indexes] = motion_dof_pos[0]
            joint_vel[env_id, robot_joint_indexes] = motion_dof_vel[0]

            slot["reset_flag"] = reset_flag

        t_write = time.perf_counter()
        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        timing["write_state"] += time.perf_counter() - t_write

        t_render = time.perf_counter()
        sim.render()  # We don't want physics (sim.step())
        scene.update(sim.get_physics_dt())
        timing["render_update"] += time.perf_counter() - t_render

        first_active = next((s for s in env_slots if s is not None), None)
        if first_active is not None:
            pos_lookat = root_states[first_active["env_id"], :3].cpu().numpy()
            sim.set_camera_view(pos_lookat + np.array([3.0, 3.0, 0.5]), pos_lookat)

        t_log = time.perf_counter()
        for i, slot in enumerate(env_slots):
            if slot is None:
                continue
            env_id = slot["env_id"]
            frame_idx = slot["frame_idx"]
            if frame_idx < slot["num_frames"]:
                slot["bufs"]["joint_pos"][frame_idx].copy_(
                    robot.data.joint_pos[env_id, :]
                )
                slot["bufs"]["joint_vel"][frame_idx].copy_(
                    robot.data.joint_vel[env_id, :]
                )
                slot["bufs"]["body_pos_w"][frame_idx].copy_(
                    robot.data.body_pos_w[env_id, :]
                )
                slot["bufs"]["body_quat_w"][frame_idx].copy_(
                    robot.data.body_quat_w[env_id, :]
                )
                slot["bufs"]["body_lin_vel_w"][frame_idx].copy_(
                    robot.data.body_lin_vel_w[env_id, :]
                )
                slot["bufs"]["body_ang_vel_w"][frame_idx].copy_(
                    robot.data.body_ang_vel_w[env_id, :]
                )
                slot["frame_idx"] += 1
            if slot["reset_flag"]:
                t_save = time.perf_counter()
                frames = slot["frame_idx"]
                os.makedirs(os.path.dirname(slot["npz_path"]), exist_ok=True)
                traj_module = build_traj_module(
                    fps=args_cli.output_fps,
                    joint_pos=slot["bufs"]["joint_pos"][:frames].contiguous(),
                    joint_vel=slot["bufs"]["joint_vel"][:frames].contiguous(),
                    body_pos_w=slot["bufs"]["body_pos_w"][:frames].contiguous(),
                    body_quat_w=slot["bufs"]["body_quat_w"][:frames].contiguous(),
                    body_lin_vel_w=slot["bufs"]["body_lin_vel_w"][:frames].contiguous(),
                    body_ang_vel_w=slot["bufs"]["body_ang_vel_w"][:frames].contiguous(),
                )
                trajectory_modules.append(traj_module)
                traj_cpu = trajectory_modules[-1].to("cpu")
                npz_payload = traj_module_to_numpy_dict(traj_cpu)
                if save_executor is not None:
                    save_futures.append(
                        save_executor.submit(np.savez, slot["npz_path"], **npz_payload)
                    )
                    print(f"[INFO]: Motion enqueued to {slot['npz_path']}")
                else:
                    np.savez(slot["npz_path"], **npz_payload)
                    print(f"[INFO]: Motion saved to {slot['npz_path']}")
                del trajectory_modules[-1]
                elapsed = time.perf_counter() - t_save
                timing["save_submit"] += elapsed
                timing["save_npz"] += elapsed
                env_slots[i] = start_next_motion(env_id)
        timing["log_copy"] += time.perf_counter() - t_log

        timing["steps"] += 1
        timing["step_total"] += time.perf_counter() - t_step_start

    print("[INFO]: All motions processed.!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    if timing["steps"] > 0:
        avg_step_ms = 1000.0 * timing["step_total"] / timing["steps"]
    else:
        avg_step_ms = 0.0
    print(
        "[TIMING]: motions="
        f"{timing['motions']} "
        f"steps={timing['steps']} "
        f"avg_step_ms={avg_step_ms:.3f} "
        f"load_motion_s={timing['load_motion']:.3f} "
        f"write_state_s={timing['write_state']:.3f} "
        f"render_update_s={timing['render_update']:.3f} "
        f"log_copy_s={timing['log_copy']:.3f} "
        f"save_npz_s={timing['save_npz']:.3f}"
    )
    if timing["save_submit"] > 0:
        print(f"[TIMING]: save_submit_s={timing['save_submit']:.3f}")
    if timing["preload_csv"] > 0:
        print(f"[TIMING]: preload_csv_s={timing['preload_csv']:.3f}")

    if save_executor is not None:
        print("[INFO]: Waiting for async .npz saves to finish...")
        for fut in concurrent.futures.as_completed(save_futures):
            fut.result()
        save_executor.shutdown(wait=True)
        print(f"[INFO]: Async .npz saves finished: {len(save_futures)} files.")

    # Close sim app


if __name__ == "__main__":
    # Run the main function
    main()
    simulation_app.close()
