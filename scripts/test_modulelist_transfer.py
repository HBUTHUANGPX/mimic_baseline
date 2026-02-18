import argparse
import time

import torch
import torch.nn as nn


def build_traj_module(device: torch.device, frames: int, num_joints: int, num_bodies: int):
    mod = nn.Module()
    mod.register_buffer("joint_pos", torch.empty((frames, num_joints), device=device))
    mod.register_buffer("joint_vel", torch.empty((frames, num_joints), device=device))
    mod.register_buffer("body_pos_w", torch.empty((frames, num_bodies, 3), device=device))
    mod.register_buffer("body_quat_w", torch.empty((frames, num_bodies, 4), device=device))
    mod.register_buffer("body_lin_vel_w", torch.empty((frames, num_bodies, 3), device=device))
    mod.register_buffer("body_ang_vel_w", torch.empty((frames, num_bodies, 3), device=device))
    return mod


def build_pinned_buffers(frames: int, num_joints: int, num_bodies: int):
    return {
        "joint_pos": torch.empty((frames, num_joints), device="cpu", pin_memory=True),
        "joint_vel": torch.empty((frames, num_joints), device="cpu", pin_memory=True),
        "body_pos_w": torch.empty((frames, num_bodies, 3), device="cpu", pin_memory=True),
        "body_quat_w": torch.empty((frames, num_bodies, 4), device="cpu", pin_memory=True),
        "body_lin_vel_w": torch.empty((frames, num_bodies, 3), device="cpu", pin_memory=True),
        "body_ang_vel_w": torch.empty((frames, num_bodies, 3), device="cpu", pin_memory=True),
    }


def sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_block(fn, device: torch.device, repeats: int = 3):
    sync_if_cuda(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync_if_cuda(device)
    return (time.perf_counter() - t0) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_traj", type=int, default=32)
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--num_joints", type=int, default=30)
    parser.add_argument("--num_bodies", type=int, default=18)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    num_traj = args.num_traj

    # 1D ModuleList
    ml_1d = nn.ModuleList(
        [build_traj_module(device, args.frames, args.num_joints, args.num_bodies) for _ in range(num_traj)]
    )

    # 2D ModuleList: rows x cols
    rows = args.rows
    cols = (num_traj + rows - 1) // rows
    ml_2d = nn.ModuleList()
    for r in range(rows):
        row_list = nn.ModuleList()
        for c in range(cols):
            idx = r * cols + c
            if idx >= num_traj:
                break
            row_list.append(build_traj_module(device, args.frames, args.num_joints, args.num_bodies))
        ml_2d.append(row_list)

    t_1d_all = time_block(lambda: ml_1d.cpu(), device, args.repeats)
    t_1d_each = time_block(lambda: [m.cpu() for m in ml_1d], device, args.repeats)
    t_2d_all = time_block(lambda: ml_2d.cpu(), device, args.repeats)
    t_2d_each = time_block(
        lambda: [[m.cpu() for m in row] for row in ml_2d], device, args.repeats
    )

    pinned_1d = [build_pinned_buffers(args.frames, args.num_joints, args.num_bodies) for _ in range(num_traj)]

    def copy_to_pinned_1d():
        for mod, pin in zip(ml_1d, pinned_1d):
            pin["joint_pos"].copy_(mod.joint_pos, non_blocking=True)
            pin["joint_vel"].copy_(mod.joint_vel, non_blocking=True)
            pin["body_pos_w"].copy_(mod.body_pos_w, non_blocking=True)
            pin["body_quat_w"].copy_(mod.body_quat_w, non_blocking=True)
            pin["body_lin_vel_w"].copy_(mod.body_lin_vel_w, non_blocking=True)
            pin["body_ang_vel_w"].copy_(mod.body_ang_vel_w, non_blocking=True)

    t_1d_each_pinned = time_block(copy_to_pinned_1d, device, args.repeats)

    print(f"[TIMING] 1d_all_cpu_s={t_1d_all:.4f}")
    print(f"[TIMING] 1d_each_cpu_s={t_1d_each:.4f}")
    print(f"[TIMING] 1d_each_pinned_nonblock_s={t_1d_each_pinned:.4f}")
    print(f"[TIMING] 2d_all_cpu_s={t_2d_all:.4f}")
    print(f"[TIMING] 2d_each_cpu_s={t_2d_each:.4f}")


if __name__ == "__main__":
    main()
