#!/usr/bin/env python3
"""Load a MuJoCo XML and compute whole-body center of mass (COM)."""

from __future__ import annotations

import argparse
import sys

import mujoco
import numpy as np


def compute_whole_body_com(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[np.ndarray, float]:
    """Return COM in world frame and total mass.

    COM = sum_i(m_i * x_i) / sum_i(m_i), where x_i is each body's COM position
    in world frame (`data.xipos[i]`).
    """
    body_mass = model.body_mass

    # Exclude world body (id 0, mass=0 in normal models).
    masses = body_mass[1:]
    positions = data.xipos[1:]

    total_mass = float(np.sum(masses))
    if total_mass <= 0:
        raise ValueError("Total body mass is non-positive; cannot compute COM.")

    com = np.sum(positions * masses[:, None], axis=0) / total_mass
    return com, total_mass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute whole-body COM from a MuJoCo XML model."
    )
    parser.add_argument(
        "--xml",
        default="general_motion_tracker_whole_body_teleoperation/general_motion_tracker_whole_body_teleoperation/assets/h2_description/H2.xml",
        required=False,
        help="Path to MuJoCo XML file",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        model = mujoco.MjModel.from_xml_path(args.xml)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to load XML: {args.xml}\n{exc}", file=sys.stderr)
        return 1

    data = mujoco.MjData(model)
    pos = [0.0, 0.0, 0.9968]  # Default position for floating base (pelvis)
    quat = [1.0, 0.0, 0.0, 0.0]  # Default orientation (no rotation)
    leg = [0.0] * 6 * 2
    waist = [0.0] * 3
    arm = [0.0, 0.0, 0.0, 1.57, 0.0 ,0.0,0.0]  * 2
    head = [0.0] * 2
    qpos = np.asarray(pos + quat + leg + waist + arm + head, dtype=np.float64)
    qpos = np.asarray(pos + quat + waist + head + arm + leg, dtype=np.float64)
    data.qpos[:] = qpos

    mujoco.mj_forward(model, data)

    try:
        com, total_mass = compute_whole_body_com(model, data)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"xml: {args.xml}")
    print(f"total_mass: {total_mass:.6f} kg")
    print(f"com_world_xyz: [{com[0]:.6f}, {com[1]:.6f}, {com[2]:.6f}] m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
