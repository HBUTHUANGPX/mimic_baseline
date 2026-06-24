#!/usr/bin/env python3
import argparse
import xml.etree.ElementTree as ET
from collections import namedtuple
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MUJOCO_DEPLOY_ROOT = SCRIPT_DIR.parent
DEFAULT_LOG_PATH = MUJOCO_DEPLOY_ROOT / "tmp/motion.npz"
DEFAULT_URDF_PATH = (
    MUJOCO_DEPLOY_ROOT / "deploy/assets/rx_27dof/rx_custom_collision_27dof.urdf"
)
DEFAULT_OUTPUT_PATH = MUJOCO_DEPLOY_ROOT / "tmp/joint_torque_speed.png"

JointLog = namedtuple("JointLog", ["joint_names", "velocities", "torques"])
JointLayoutRow = namedtuple("JointLayoutRow", ["group_name", "joint_indices"])

JOINT_GROUPS = {
    "l_leg": [
        "l_hip_pitch_joint",
        "l_hip_roll_joint",
        "l_hip_yaw_joint",
        "l_knee_joint",
        "l_ankle_pitch_joint",
        "l_ankle_roll_joint",
    ],
    "r_leg": [
        "r_hip_pitch_joint",
        "r_hip_roll_joint",
        "r_hip_yaw_joint",
        "r_knee_joint",
        "r_ankle_pitch_joint",
        "r_ankle_roll_joint",
    ],
    "l_arm": [
        "l_shoulder_pitch_joint",
        "l_shoulder_roll_joint",
        "l_shoulder_yaw_joint",
        "l_elbow_joint",
        "l_wrist_yaw_joint",
        "l_wrist_roll_joint",
    ],
    "r_arm": [
        "r_shoulder_pitch_joint",
        "r_shoulder_roll_joint",
        "r_shoulder_yaw_joint",
        "r_elbow_joint",
        "r_wrist_yaw_joint",
        "r_wrist_roll_joint",
    ],
    "waist": [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ],
}


def _grouped_joint_names():
    joint_names = []
    for names in JOINT_GROUPS.values():
        joint_names.extend(names)
    return joint_names


def load_joint_log(log):
    if "dof_velocities" not in log or "dof_torque" not in log or "dof_names" not in log:
        raise KeyError(
            "Log must contain dof_names, dof_velocities, and dof_torque arrays."
        )

    source_joint_names = [str(name) for name in log["dof_names"].tolist()]
    velocities = np.asarray(log["dof_velocities"], dtype=np.float64)
    torques = np.asarray(log["dof_torque"], dtype=np.float64)

    if velocities.ndim != 2 or torques.ndim != 2:
        raise ValueError("dof_velocities and dof_torque must be 2-D arrays.")
    if velocities.shape != torques.shape:
        raise ValueError(
            f"dof_velocities shape {velocities.shape} does not match "
            f"dof_torque shape {torques.shape}."
        )
    if velocities.shape[1] != len(source_joint_names):
        raise ValueError(
            f"Found {len(source_joint_names)} joint names, but velocity has "
            f"{velocities.shape[1]} columns."
        )

    desired_joint_names = _grouped_joint_names()
    missing = [name for name in desired_joint_names if name not in source_joint_names]
    if missing:
        raise ValueError(f"Log is missing expected joints: {missing}")

    indices = [source_joint_names.index(name) for name in desired_joint_names]
    return JointLog(
        joint_names=desired_joint_names,
        velocities=velocities[:, indices],
        torques=torques[:, indices],
    )


def build_joint_layout(joint_names):
    rows = []
    for group_name, group_joint_names in JOINT_GROUPS.items():
        rows.append(
            JointLayoutRow(
                group_name=group_name,
                joint_indices=[joint_names.index(name) for name in group_joint_names],
            )
        )
    return rows


def read_urdf_limits(urdf_path):
    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file does not exist: {urdf_path}")

    root = ET.parse(urdf_path).getroot()
    limits = {}
    for joint in root.findall("joint"):
        joint_name = joint.attrib.get("name")
        limit = joint.find("limit")
        if joint_name is None or limit is None:
            continue
        effort = limit.attrib.get("effort")
        velocity = limit.attrib.get("velocity")
        limits[joint_name] = {
            "torque": None if effort is None else abs(float(effort)),
            "speed": None if velocity is None else abs(float(velocity)),
        }
    return limits


def _max_abs(values):
    if values.size == 0:
        return 1.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return float(np.max(np.abs(finite)))


def _axis_bound(values, limit):
    data_bound = _max_abs(values)
    limit_bound = 0.0 if limit is None else abs(float(limit))
    bound = max(data_bound, limit_bound, 1e-6)
    return bound * 1.12


def _plot_limit_lines(ax, torque_limit, speed_limit, show_negative_limits):
    if torque_limit is not None:
        ax.axhline(torque_limit, color="#d62728", linewidth=1.0, linestyle="--")
        if show_negative_limits:
            ax.axhline(-torque_limit, color="#d62728", linewidth=1.0, linestyle=":")
    if speed_limit is not None:
        ax.axvline(speed_limit, color="#1f77b4", linewidth=1.0, linestyle="--")
        if show_negative_limits:
            ax.axvline(-speed_limit, color="#1f77b4", linewidth=1.0, linestyle=":")


def plot_joint_torque_speed(
    joint_log,
    limits,
    output_path,
    alpha=0.25,
    point_size=5.0,
    dpi=180,
    show_negative_limits=False,
):
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to render the plot. Install it in the "
            "current environment or run this script from an environment that "
            "already provides matplotlib."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1.")

    layout = build_joint_layout(joint_log.joint_names)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(24, 15), constrained_layout=True)
    width_ratios = []
    for _ in range(6):
        width_ratios.extend([5.0, 1.0])
    grid = fig.add_gridspec(
        nrows=5,
        ncols=12,
        width_ratios=width_ratios,
        wspace=0.08,
        hspace=0.32,
    )

    scatter_axes = []
    for row_idx, row in enumerate(layout):
        for col_idx in range(6):
            scatter_ax = fig.add_subplot(grid[row_idx, col_idx * 2])
            box_ax = fig.add_subplot(grid[row_idx, col_idx * 2 + 1], sharey=scatter_ax)

            if col_idx >= len(row.joint_indices):
                scatter_ax.axis("off")
                box_ax.axis("off")
                continue

            joint_idx = row.joint_indices[col_idx]
            joint_name = joint_log.joint_names[joint_idx]
            speeds = joint_log.velocities[:, joint_idx]
            torques = joint_log.torques[:, joint_idx]
            joint_limits = limits.get(joint_name, {})
            torque_limit = joint_limits.get("torque")
            speed_limit = joint_limits.get("speed")

            x_bound = _axis_bound(speeds, speed_limit)
            y_bound = _axis_bound(torques, torque_limit)
            scatter_ax.scatter(
                speeds,
                torques,
                s=point_size,
                alpha=alpha,
                c="#2f6f9f",
                edgecolors="none",
                rasterized=True,
            )
            _plot_limit_lines(
                scatter_ax, torque_limit, speed_limit, show_negative_limits
            )
            scatter_ax.set_xlim(-x_bound, x_bound)
            scatter_ax.set_ylim(-y_bound, y_bound)
            scatter_ax.grid(True, linewidth=0.35, alpha=0.35)
            scatter_ax.set_title(joint_name.replace("_joint", ""), fontsize=9)
            if row_idx == 4:
                scatter_ax.set_xlabel("speed [rad/s]", fontsize=8)
            if col_idx == 0:
                scatter_ax.set_ylabel(f"{row.group_name}\ntorque [Nm]", fontsize=8)
            scatter_ax.tick_params(labelsize=7)

            box = box_ax.boxplot(
                torques,
                vert=True,
                widths=0.55,
                showfliers=False,
                patch_artist=True,
            )
            for patch in box["boxes"]:
                patch.set(facecolor="#9ecae1", alpha=0.75, edgecolor="#2f6f9f")
            for median in box["medians"]:
                median.set(color="#d62728", linewidth=1.1)
            box_ax.set_ylim(scatter_ax.get_ylim())
            box_ax.set_xticks([])
            box_ax.yaxis.tick_right()
            box_ax.tick_params(axis="y", labelsize=6)
            box_ax.grid(True, axis="y", linewidth=0.3, alpha=0.25)
            scatter_axes.append(scatter_ax)

    fig.suptitle("Joint Torque-Speed Scatter", fontsize=16)
    fig.text(
        0.5,
        0.01,
        "Dashed horizontal line: max torque. Dashed vertical line: max speed.",
        ha="center",
        fontsize=10,
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot per-joint torque-speed scatter plots from deploy_mujoco motion.npz."
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"Path to deploy_mujoco log npz. Default: {DEFAULT_LOG_PATH}",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF_PATH,
        help=f"URDF used for effort/velocity limits. Default: {DEFAULT_URDF_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output image path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Scatter point alpha in [0, 1]. Default: 0.25",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=5.0,
        help="Scatter point size. Default: 5.0",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output image DPI. Default: 180",
    )
    parser.add_argument(
        "--show-negative-limits",
        action="store_true",
        help="Also draw mirrored negative max torque and max speed lines.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with np.load(args.log, allow_pickle=True) as log:
        joint_log = load_joint_log(log)
    limits = read_urdf_limits(args.urdf)
    output_path = plot_joint_torque_speed(
        joint_log,
        limits,
        args.output,
        alpha=args.alpha,
        point_size=args.point_size,
        dpi=args.dpi,
        show_negative_limits=args.show_negative_limits,
    )
    print(f"Saved torque-speed plot to: {output_path}")


if __name__ == "__main__":
    main()
