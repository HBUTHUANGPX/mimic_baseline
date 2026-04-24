from __future__ import annotations

"""Compare exported SOMA BVH against a human motion npz.

This script intentionally follows the same human parsing flow as
`soma-retargeter/app/play_npz_mujoco.py` for BVH:

BVH -> soma_retargeter.assets.bvh.load_bvh
    -> local transforms
    -> compute_global_joint_transforms
    -> apply_visualization_frame

The NPZ side uses the same local->global->visualization-frame path so both
sources are compared in exactly the same coordinate convention.
"""

import argparse
import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = Path(__file__).resolve().parent
SOMA_RETARGETER_ROOT = REPO_ROOT / "soma-retargeter"
SOMA_RETARGETER_APP = SOMA_RETARGETER_ROOT / "app"
REFERENCE_PLAYER_COMMON = SOMA_RETARGETER_APP / "motion_npz_player_common.py"

DEFAULT_INPUT_BVH = Path(
    "hdf5_parse/out/soma_bvh/annotation_83581004785937_83582554784896.bvh"
)
DEFAULT_FOCUS_JOINTS = (
    "Root",
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Neck2",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
)

LOGGER = logging.getLogger(__name__)


@dataclass
class JointDiffStats:
    joint_name: str
    position_max_abs_diff: float
    position_mean_abs_diff: float
    quaternion_max_abs_diff: float
    quaternion_mean_abs_diff: float


@dataclass
class AlignmentReport:
    npz_path: Path
    bvh_path: Path
    frame_count: int
    npz_fps: float
    bvh_fps: float
    bvh_resampled_to_npz_fps: bool
    compared_joint_names: list[str]
    overall_position_max_abs_diff: float
    overall_position_mean_abs_diff: float
    overall_quaternion_max_abs_diff: float
    overall_quaternion_mean_abs_diff: float
    joint_stats: dict[str, JointDiffStats]


def _ensure_import_paths() -> None:
    for path in (MODULE_DIR, REPO_ROOT, SOMA_RETARGETER_ROOT, SOMA_RETARGETER_APP):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _reference_player_common():
    _ensure_import_paths()
    return _load_module("motion_npz_player_common", REFERENCE_PLAYER_COMMON)


def _reference_bvh_utils():
    _ensure_import_paths()
    import soma_retargeter.assets.bvh as bvh_utils

    return bvh_utils


def _read_npz_names(payload: np.lib.npyio.NpzFile, *keys: str) -> list[str]:
    for key in keys:
        if key in payload:
            return [str(value) for value in np.asarray(payload[key]).tolist()]
    raise KeyError(f"Missing joint-name key. Expected one of: {keys}")


def _read_npz_scalar_first(payload: np.lib.npyio.NpzFile) -> bool:
    if "scalar_first" not in payload:
        return False
    value = np.asarray(payload["scalar_first"])
    return bool(value.item() if value.shape == () else value.reshape(-1)[0])


def _read_npz_fps(payload: np.lib.npyio.NpzFile) -> float:
    if "fps" not in payload:
        raise KeyError("Missing fps in npz payload.")
    return float(np.asarray(payload["fps"]).item())


def _to_xyzw(quat: np.ndarray, *, scalar_first: bool) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    if quat.shape[-1] != 4:
        raise ValueError(f"Quaternion last dim must be 4, got {quat.shape}")
    if scalar_first:
        return quat[..., [1, 2, 3, 0]]
    return quat


def load_npz_visualized_globals(
    npz_path: str | Path,
) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    npz_path = Path(npz_path)
    reference = _reference_player_common()
    with np.load(npz_path, allow_pickle=False) as payload:
        fps = _read_npz_fps(payload)
        scalar_first = _read_npz_scalar_first(payload)
        joint_names = _read_npz_names(payload, "human_joint_names", "human_body_names")
        if "human_global_pos" in payload and "human_global_quat" in payload:
            positions = np.asarray(payload["human_global_pos"], dtype=np.float32)
            rotations = _to_xyzw(
                np.asarray(payload["human_global_quat"], dtype=np.float32),
                scalar_first=scalar_first,
            )
            return joint_names, positions.astype(np.float32), rotations.astype(np.float32), fps

        parent_indices = np.asarray(payload["human_parent_indices"], dtype=np.int32)
        local_transforms = np.asarray(payload["human_local_transforms"], dtype=np.float32)
        local_transforms = np.array(local_transforms, copy=True)
        local_transforms[..., 3:7] = _to_xyzw(
            local_transforms[..., 3:7],
            scalar_first=scalar_first,
        )

    positions, rotations = reference.compute_global_joint_transforms(
        local_transforms,
        parent_indices,
    )
    positions, rotations = reference.apply_visualization_frame(positions, rotations)
    return joint_names, positions.astype(np.float32), rotations.astype(np.float32), fps


def _compute_sample_times(sample_rate: float, num_frames: int, output_fps: float) -> np.ndarray:
    if num_frames <= 0:
        return np.zeros((0,), dtype=np.float32)
    if num_frames == 1:
        return np.zeros((1,), dtype=np.float32)

    duration = (num_frames - 1) / float(sample_rate)
    times = np.arange(0.0, duration, 1.0 / float(output_fps), dtype=np.float32)
    if times.size == 0:
        return np.zeros((1,), dtype=np.float32)
    return times


def load_bvh_visualized_globals(
    bvh_path: str | Path,
    *,
    target_fps: float | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, float, bool]:
    bvh_path = Path(bvh_path)
    bvh_utils = _reference_bvh_utils()
    reference = _reference_player_common()
    skeleton, animation = bvh_utils.load_bvh(str(bvh_path))
    bvh_fps = float(animation.sample_rate)
    resampled = False
    if target_fps is not None and not np.isclose(float(target_fps), bvh_fps):
        sample_times = _compute_sample_times(bvh_fps, int(animation.num_frames), float(target_fps))
        local_transforms = np.asarray([animation.sample(float(t)) for t in sample_times], dtype=np.float32)
        resampled = True
    else:
        local_transforms = np.asarray(animation.local_transforms, dtype=np.float32)
    parent_indices = np.asarray(skeleton.parent_indices, dtype=np.int32)
    positions, rotations = reference.compute_global_joint_transforms(
        local_transforms,
        parent_indices,
    )
    positions, rotations = reference.apply_visualization_frame(positions, rotations)
    return (
        list(skeleton.joint_names),
        positions.astype(np.float32),
        rotations.astype(np.float32),
        bvh_fps,
        resampled,
    )


def _aligned_joint_names(
    npz_joint_names: Iterable[str],
    bvh_joint_names: Iterable[str],
    focus_joint_names: Iterable[str] | None = None,
) -> list[str]:
    npz_names = list(npz_joint_names)
    bvh_set = set(bvh_joint_names)
    if focus_joint_names is None:
        return [name for name in npz_names if name in bvh_set]
    focus = []
    for name in focus_joint_names:
        if name in npz_names and name in bvh_set and name not in focus:
            focus.append(name)
    return focus


def _quat_component_abs_diff(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    direct = np.abs(lhs - rhs)
    flipped = np.abs(lhs + rhs)
    return np.minimum(direct, flipped)


def compare_annotation_npz_against_bvh(
    *,
    npz_path: str | Path,
    bvh_path: str | Path,
    focus_joint_names: Iterable[str] | None = DEFAULT_FOCUS_JOINTS,
) -> AlignmentReport:
    npz_path = Path(npz_path)
    bvh_path = Path(bvh_path)
    npz_joint_names, npz_positions, npz_rotations, npz_fps = load_npz_visualized_globals(npz_path)
    bvh_joint_names, bvh_positions, bvh_rotations, bvh_fps, bvh_resampled = load_bvh_visualized_globals(
        bvh_path,
        target_fps=npz_fps,
    )
    if bvh_resampled:
        LOGGER.info(
            "BVH fps (%.3f) differs from NPZ fps (%.3f); resampling BVH to match NPZ fps before comparison.",
            bvh_fps,
            npz_fps,
        )

    compared_joint_names = _aligned_joint_names(
        npz_joint_names,
        bvh_joint_names,
        focus_joint_names,
    )
    if not compared_joint_names:
        raise ValueError("No overlapping joints found between NPZ and BVH.")

    npz_index = {name: idx for idx, name in enumerate(npz_joint_names)}
    bvh_index = {name: idx for idx, name in enumerate(bvh_joint_names)}
    frame_count = min(int(npz_positions.shape[0]), int(bvh_positions.shape[0]))
    if frame_count <= 0:
        raise ValueError("No frames available for comparison.")

    npz_pos = np.stack(
        [npz_positions[:frame_count, npz_index[name]] for name in compared_joint_names],
        axis=1,
    )
    bvh_pos = np.stack(
        [bvh_positions[:frame_count, bvh_index[name]] for name in compared_joint_names],
        axis=1,
    )
    npz_quat = np.stack(
        [npz_rotations[:frame_count, npz_index[name]] for name in compared_joint_names],
        axis=1,
    )
    bvh_quat = np.stack(
        [bvh_rotations[:frame_count, bvh_index[name]] for name in compared_joint_names],
        axis=1,
    )

    pos_abs_diff = np.abs(npz_pos - bvh_pos)
    quat_abs_diff = _quat_component_abs_diff(npz_quat, bvh_quat)

    joint_stats: dict[str, JointDiffStats] = {}
    for joint_offset, joint_name in enumerate(compared_joint_names):
        joint_stats[joint_name] = JointDiffStats(
            joint_name=joint_name,
            position_max_abs_diff=float(pos_abs_diff[:, joint_offset].max()),
            position_mean_abs_diff=float(pos_abs_diff[:, joint_offset].mean()),
            quaternion_max_abs_diff=float(quat_abs_diff[:, joint_offset].max()),
            quaternion_mean_abs_diff=float(quat_abs_diff[:, joint_offset].mean()),
        )

    return AlignmentReport(
        npz_path=npz_path,
        bvh_path=bvh_path,
        frame_count=frame_count,
        npz_fps=npz_fps,
        bvh_fps=bvh_fps,
        bvh_resampled_to_npz_fps=bvh_resampled,
        compared_joint_names=compared_joint_names,
        overall_position_max_abs_diff=float(pos_abs_diff.max()),
        overall_position_mean_abs_diff=float(pos_abs_diff.mean()),
        overall_quaternion_max_abs_diff=float(quat_abs_diff.max()),
        overall_quaternion_mean_abs_diff=float(quat_abs_diff.mean()),
        joint_stats=joint_stats,
    )


def format_alignment_report(report: AlignmentReport, *, top_k: int = 10) -> str:
    lines = [
        f"NPZ: {report.npz_path}",
        f"BVH: {report.bvh_path}",
        (
            f"FPS: npz={report.npz_fps:.3f}, bvh={report.bvh_fps:.3f}, "
            f"bvh_resampled_to_npz_fps={report.bvh_resampled_to_npz_fps}"
        ),
        f"Compared frames: {report.frame_count}",
        f"Compared joints ({len(report.compared_joint_names)}): {', '.join(report.compared_joint_names)}",
        (
            "Overall diffs: "
            f"pos max={report.overall_position_max_abs_diff:.6f}, "
            f"pos mean={report.overall_position_mean_abs_diff:.6f}, "
            f"quat max={report.overall_quaternion_max_abs_diff:.6f}, "
            f"quat mean={report.overall_quaternion_mean_abs_diff:.6f}"
        ),
        "Worst joints by quaternion max abs diff:",
    ]
    ranked = sorted(
        report.joint_stats.values(),
        key=lambda item: (item.quaternion_max_abs_diff, item.position_max_abs_diff),
        reverse=True,
    )
    for joint in ranked[: max(1, int(top_k))]:
        lines.append(
            (
                f"- {joint.joint_name}: "
                f"pos max={joint.position_max_abs_diff:.6f}, "
                f"pos mean={joint.position_mean_abs_diff:.6f}, "
                f"quat max={joint.quaternion_max_abs_diff:.6f}, "
                f"quat mean={joint.quaternion_mean_abs_diff:.6f}"
            )
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare exported SOMA BVH against a human motion npz that follows soma-retargeter semantics."
    )
    parser.add_argument(
        "--npz",
        type=Path,
        required=True,
        help="Path to a human motion npz, typically produced by bvh_to_csv_converter.py.",
    )
    parser.add_argument(
        "--bvh",
        type=Path,
        default=DEFAULT_INPUT_BVH,
        help="Path to exported SOMA BVH.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many worst joints to print.",
    )
    parser.add_argument(
        "--all-joints",
        action="store_true",
        help="Compare all overlapping joints instead of the default torso/limb focus set.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = build_arg_parser().parse_args()
    report = compare_annotation_npz_against_bvh(
        npz_path=args.npz,
        bvh_path=args.bvh,
        focus_joint_names=None if args.all_joints else DEFAULT_FOCUS_JOINTS,
    )
    print(format_alignment_report(report, top_k=args.top_k))


if __name__ == "__main__":
    main()
