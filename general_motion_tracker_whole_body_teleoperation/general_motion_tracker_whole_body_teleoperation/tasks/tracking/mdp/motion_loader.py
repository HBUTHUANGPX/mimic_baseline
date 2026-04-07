from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch


def extract_part(path: str) -> str | None:
    """Extract an artifact-relative motion path."""
    if path.startswith("artifacts/"):
        relative_path = path[len("artifacts/") :]
        if relative_path.endswith(".npz"):
            return relative_path
    return None


def _normalize_paths(paths: list[str] | str) -> list[str]:
    """Normalize a path field so the loader can treat single and multi-file groups uniformly."""
    if isinstance(paths, str):
        return [paths]
    return list(paths)


def _validate_motion_file(motion_path: str) -> None:
    """Fail fast when a configured motion path does not exist."""
    assert os.path.isfile(motion_path), f"Invalid file path: {motion_path}"


def _quat_mul_xyzw(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiply for tensors stored in xyzw order."""
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def _quat_conjugate_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate for xyzw tensors."""
    result = quat.clone()
    result[..., :3] *= -1.0
    return result


def _normalize_quat_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """Keep quaternion-derived finite differences numerically stable."""
    return quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)


def _finite_difference(values: torch.Tensor, dt: float) -> torch.Tensor:
    """Simple first-order finite difference with first-frame replication."""
    if values.shape[0] <= 1:
        return torch.zeros_like(values)
    diff = (values[1:] - values[:-1]) / dt
    return torch.cat((diff[:1], diff), dim=0)


def _quat_to_angular_velocity(quat_xyzw: torch.Tensor, dt: float) -> torch.Tensor:
    """Estimate angular velocity from consecutive xyzw quaternions.

    The returned tensor has the same leading dimensions as the input and stores
    an angular velocity vector in rad/s on the last axis.
    """
    if quat_xyzw.shape[0] <= 1:
        return torch.zeros((*quat_xyzw.shape[:-1], 3), dtype=quat_xyzw.dtype, device=quat_xyzw.device)

    quat_xyzw = _normalize_quat_xyzw(quat_xyzw)
    rel = _quat_mul_xyzw(quat_xyzw[1:], _quat_conjugate_xyzw(quat_xyzw[:-1]))
    rel = _normalize_quat_xyzw(rel)

    neg_mask = rel[..., 3] < 0.0
    rel = torch.where(neg_mask.unsqueeze(-1), -rel, rel)

    xyz = rel[..., :3]
    w = rel[..., 3].clamp(-1.0, 1.0)
    sin_half = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w.unsqueeze(-1))
    axis = xyz / sin_half.clamp_min(1e-8)
    angvel = axis * (angle / dt)
    angvel = torch.where(sin_half > 1e-8, angvel, torch.zeros_like(angvel))
    return torch.cat((angvel[:1], angvel), dim=0)


class MotionLoader:
    """Load human+robot motions and organize them on one shared timeline."""

    def __init__(
        self,
        motion_file_group: dict[str, list[str] | str],
        body_indexes: Sequence[int],
        history_frames: int,
        future_frames: int,
        device: str = "cpu",
    ) -> None:
        self.group_names: list[str] = []
        self.extracted_list: list[str] = []
        self.motion_lengths: list[int] = []
        self.num_motions = 0
        self.fps: int | None = None
        self._body_indexes = list(body_indexes)
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.window_size = history_frames + future_frames + 1
        self.device = device

        self.robot_joint_names: list[str] | None = None
        self.robot_body_names: list[str] | None = None
        self.human_joint_names: list[str] | None = None

        robot_joint_pos_list: list[torch.Tensor] = []
        robot_joint_vel_list: list[torch.Tensor] = []
        robot_body_pos_list: list[torch.Tensor] = []
        robot_body_vel_list: list[torch.Tensor] = []
        robot_body_quat_list: list[torch.Tensor] = []
        robot_body_angvel_list: list[torch.Tensor] = []

        human_joint_pos_list: list[torch.Tensor] = []
        human_joint_vel_list: list[torch.Tensor] = []
        human_body_pos_list: list[torch.Tensor] = []
        human_body_vel_list: list[torch.Tensor] = []
        human_body_quat_list: list[torch.Tensor] = []
        human_body_angvel_list: list[torch.Tensor] = []

        motion_id_list: list[torch.Tensor] = []
        motion_group_list: list[torch.Tensor] = []

        motion_group_index = 0
        for group_name, paths in motion_file_group.items():
            normalized_paths = _normalize_paths(paths)
            print(f"\nGroup: {group_name}")
            print(f"[INFO] Loading {len(normalized_paths)} motion files.")

            extracted_list = [
                extract_part(path)
                for path in normalized_paths
                if extract_part(path) is not None
            ]

            for local_motion_id, motion_path in enumerate(normalized_paths):
                _validate_motion_file(motion_path)
                data = np.load(motion_path, allow_pickle=False)
                self._validate_schema(data, motion_path)
                self._validate_metadata(data)

                # Robot and human tensors stay separate, but every frame in this
                # file is assumed to refer to the same shared timeline instant.
                robot_joint_pos = torch.tensor(
                    data["robot_joint_pos"], dtype=torch.float32, device=device
                )
                robot_body_pos = torch.tensor(
                    data["robot_body_pos"], dtype=torch.float32, device=device
                )
                robot_body_quat = torch.tensor(
                    data["robot_body_quat"], dtype=torch.float32, device=device
                )
                human_local_transforms = torch.tensor(
                    data["human_local_transforms"], dtype=torch.float32, device=device
                )
                human_global_pos = torch.tensor(
                    data["human_global_pos"], dtype=torch.float32, device=device
                )
                human_global_quat = torch.tensor(
                    data["human_global_quat"], dtype=torch.float32, device=device
                )

                num_frames = int(robot_joint_pos.shape[0])
                self._validate_frame_alignment(
                    num_frames,
                    robot_body_pos,
                    robot_body_quat,
                    human_local_transforms,
                    human_global_pos,
                    human_global_quat,
                    motion_path,
                )

                dt = 1.0 / float(self.fps)
                # Missing velocity-like quantities are reconstructed on load so
                # existing exported NPZ files can be used directly for training.
                robot_joint_vel = _finite_difference(robot_joint_pos, dt)
                robot_body_vel = _finite_difference(robot_body_pos, dt)
                robot_body_angvel = _quat_to_angular_velocity(robot_body_quat, dt)

                # Human joints are 3-DoF ball joints, so the joint state is kept
                # as a local quaternion instead of being projected onto scalar DoFs.
                human_joint_pos = human_local_transforms[..., 3:7]
                human_joint_vel = _quat_to_angular_velocity(human_joint_pos, dt)
                human_body_vel = _finite_difference(human_global_pos, dt)
                human_body_angvel = _quat_to_angular_velocity(human_global_quat, dt)

                robot_joint_pos_list.append(robot_joint_pos)
                robot_joint_vel_list.append(robot_joint_vel)
                robot_body_pos_list.append(robot_body_pos)
                robot_body_vel_list.append(robot_body_vel)
                robot_body_quat_list.append(robot_body_quat)
                robot_body_angvel_list.append(robot_body_angvel)

                human_joint_pos_list.append(human_joint_pos)
                human_joint_vel_list.append(human_joint_vel)
                human_body_pos_list.append(human_global_pos)
                human_body_vel_list.append(human_body_vel)
                human_body_quat_list.append(human_global_quat)
                human_body_angvel_list.append(human_body_angvel)

                motion_group_list.append(
                    torch.full((num_frames, 1), motion_group_index, dtype=torch.long, device=device)
                )
                motion_id_list.append(
                    torch.full(
                        (num_frames, 1),
                        self.num_motions + local_motion_id,
                        dtype=torch.long,
                        device=device,
                    )
                )
                self.motion_lengths.append(num_frames)

            self.extracted_list.extend(extracted_list)
            self.group_names.append(group_name)
            self.num_motions += len(normalized_paths)
            motion_group_index += 1

        assert self.num_motions > 0, "At least one motion file is required."

        self.robot_joint_pos = torch.cat(robot_joint_pos_list, dim=0)
        self.robot_joint_vel = torch.cat(robot_joint_vel_list, dim=0)
        self._robot_body_pos = torch.cat(robot_body_pos_list, dim=0)
        self._robot_body_vel = torch.cat(robot_body_vel_list, dim=0)
        self._robot_body_quat = torch.cat(robot_body_quat_list, dim=0)
        self._robot_body_angvel = torch.cat(robot_body_angvel_list, dim=0)

        self.human_joint_pos = torch.cat(human_joint_pos_list, dim=0)
        self.human_joint_vel = torch.cat(human_joint_vel_list, dim=0)
        self.human_body_pos = torch.cat(human_body_pos_list, dim=0)
        self.human_body_vel = torch.cat(human_body_vel_list, dim=0)
        self.human_body_quat = torch.cat(human_body_quat_list, dim=0)
        self.human_body_angvel = torch.cat(human_body_angvel_list, dim=0)

        self._motion_id = torch.cat(motion_id_list, dim=0)
        self._motion_group = torch.cat(motion_group_list, dim=0)

        # Only the robot body tensors are reduced by body_indexes. Human body
        # tensors stay full-width because their topology is tied to
        # human_joint_names rather than the robot body subset.
        body_index_tensor = torch.as_tensor(self._body_indexes, dtype=torch.long, device=device)
        assert int(body_index_tensor.max().item()) < self._robot_body_pos.shape[1]
        self.robot_body_pos = self._robot_body_pos[:, body_index_tensor]
        self.robot_body_vel = self._robot_body_vel[:, body_index_tensor]
        self.robot_body_quat = self._robot_body_quat[:, body_index_tensor]
        self.robot_body_angvel = self._robot_body_angvel[:, body_index_tensor]

        # Legacy robot-only aliases keep commands.py readable while the rest of
        # the stack is migrated to explicit robot_* / human_* names.
        self.joint_pos = self.robot_joint_pos
        self.joint_vel = self.robot_joint_vel
        self.body_pos_w = self.robot_body_pos
        self.body_quat_w = self.robot_body_quat
        self.body_lin_vel_w = self.robot_body_vel
        self.body_ang_vel_w = self.robot_body_angvel

        self.time_step_total = self.robot_joint_pos.shape[0]
        self.motion_indices = self._build_motion_indices(device)
        self.new_data_flag = self._build_new_data_flag(device)
        self.window_offsets = torch.arange(
            -self.history_frames,
            self.future_frames + 1,
            dtype=torch.long,
            device=device,
        )
        self.valid_center_mask = self._build_valid_center_mask(device)
        self.valid_center_indices = torch.nonzero(
            self.valid_center_mask, as_tuple=False
        ).squeeze(-1)
        assert self.valid_center_indices.numel() > 0, "No valid center frames found."

    def _validate_schema(self, data: np.lib.npyio.NpzFile, motion_path: str) -> None:
        """Require the current unified NPZ schema rather than supporting legacy variants."""
        required = (
            "fps",
            "robot_joint_names",
            "robot_body_names",
            "human_joint_names",
            "robot_joint_pos",
            "robot_body_pos",
            "robot_body_quat",
            "human_local_transforms",
            "human_global_pos",
            "human_global_quat",
        )
        missing = [key for key in required if key not in data.files]
        assert not missing, f"Motion file {motion_path} missing fields: {missing}"

    def _validate_metadata(self, data: np.lib.npyio.NpzFile) -> None:
        """All motions in one loader must agree on names and sampling rate."""
        fps = int(data["fps"])
        if self.fps is None:
            self.fps = fps
            self.robot_joint_names = data["robot_joint_names"].tolist()
            self.robot_body_names = data["robot_body_names"].tolist()
            self.human_joint_names = data["human_joint_names"].tolist()
            return

        assert self.fps == fps, "All motion files must have the same fps."
        assert self.robot_joint_names == data["robot_joint_names"].tolist()
        assert self.robot_body_names == data["robot_body_names"].tolist()
        assert self.human_joint_names == data["human_joint_names"].tolist()

    def _validate_frame_alignment(
        self,
        num_frames: int,
        robot_body_pos: torch.Tensor,
        robot_body_quat: torch.Tensor,
        human_local_transforms: torch.Tensor,
        human_global_pos: torch.Tensor,
        human_global_quat: torch.Tensor,
        motion_path: str,
    ) -> None:
        """Human and robot branches must have the same frame count inside one NPZ."""
        aligned_lengths = (
            robot_body_pos.shape[0],
            robot_body_quat.shape[0],
            human_local_transforms.shape[0],
            human_global_pos.shape[0],
            human_global_quat.shape[0],
        )
        assert all(length == num_frames for length in aligned_lengths), (
            f"Frame mismatch in motion file {motion_path}: "
            f"{(num_frames, *aligned_lengths)}"
        )

    def _build_motion_indices(self, device: str) -> torch.Tensor:
        """Store each motion segment as a global [start, end) range."""
        motion_indices = torch.zeros(self.num_motions, 2, dtype=torch.long, device=device)
        start = 0
        for motion_id, length in enumerate(self.motion_lengths):
            end = start + length
            motion_indices[motion_id, 0] = start
            motion_indices[motion_id, 1] = end
            start = end
        return motion_indices

    def _build_new_data_flag(self, device: str) -> torch.Tensor:
        """Mark the first frame of every motion after concatenation."""
        new_data_flag = torch.zeros(self.time_step_total, dtype=torch.bool, device=device)
        cumulative_length = 0
        for motion_id, length in enumerate(self.motion_lengths):
            if motion_id > 0:
                new_data_flag[cumulative_length] = True
            cumulative_length += length
        return new_data_flag

    def _build_valid_center_mask(self, device: str) -> torch.Tensor:
        """Centers are valid only when the full history/future window stays inside one motion."""
        valid_center_mask = torch.zeros(self.time_step_total, dtype=torch.bool, device=device)
        for motion_id in range(self.num_motions):
            start, end = self.motion_indices[motion_id]
            valid_start = start + self.history_frames
            valid_end = end - self.future_frames
            if valid_start < valid_end:
                valid_center_mask[valid_start:valid_end] = True
        return valid_center_mask
