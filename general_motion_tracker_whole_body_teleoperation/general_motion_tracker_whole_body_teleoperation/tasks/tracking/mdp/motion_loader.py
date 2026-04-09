from collections.abc import Sequence
import torch
import os
import numpy as np

class MotionLoader:
    def __init__(
        self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"
    ):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(
            data["joint_pos"], dtype=torch.float32, device=device
        )
        self.joint_vel = torch.tensor(
            data["joint_vel"], dtype=torch.float32, device=device
        )
        self._body_pos_w = torch.tensor(
            data["body_pos_w"], dtype=torch.float32, device=device
        )
        self._body_quat_w = torch.tensor(
            data["body_quat_w"], dtype=torch.float32, device=device
        )
        self._body_lin_vel_w = torch.tensor(
            data["body_lin_vel_w"], dtype=torch.float32, device=device
        )
        self._body_ang_vel_w = torch.tensor(
            data["body_ang_vel_w"], dtype=torch.float32, device=device
        )
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]
        self.body_pos_w = self._body_pos_w[:, self._body_indexes]
        self.body_quat_w = self._body_quat_w[:, self._body_indexes]
        self.body_lin_vel_w = self._body_lin_vel_w[:, self._body_indexes]   
        self.body_ang_vel_w = self._body_ang_vel_w[:, self._body_indexes]


class MotionLoader_robot:
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
        self.fps = None
        self._body_indexes = body_indexes
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.window_size = history_frames + future_frames + 1

        joint_pos_list: list[torch.Tensor] = []
        joint_vel_list: list[torch.Tensor] = []
        body_pos_w_list: list[torch.Tensor] = []
        body_quat_w_list: list[torch.Tensor] = []
        body_lin_vel_w_list: list[torch.Tensor] = []
        body_ang_vel_w_list: list[torch.Tensor] = []
        motion_id_list: list[torch.Tensor] = []
        motion_group_list: list[torch.Tensor] = []

        motion_group_index = 0
        for group_name, paths in motion_file_group.items():
            normalized_paths = self._normalize_paths(paths)
            print(f"\nGroup: {group_name}")
            print(f"[INFO] Loading {len(normalized_paths)} motion files for training.")
            # print(f"[INFO] load motion file: {normalized_paths}")

            extracted_list = [
                self.extract_part(path)
                for path in normalized_paths
                if self.extract_part(path) is not None
            ]

            for local_motion_id, motion_path in enumerate(normalized_paths):
                self._validate_motion_file(motion_path)
                data = np.load(motion_path)
                self._validate_fps(data)

                joint_pos_tensor = torch.tensor(
                    data["joint_pos"], dtype=torch.float32, device=device
                )
                num_frames = joint_pos_tensor.shape[0]

                joint_pos_list.append(joint_pos_tensor)
                joint_vel_list.append(
                    torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
                )
                body_pos_w_list.append(
                    torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
                )
                body_quat_w_list.append(
                    torch.tensor(
                        data["body_quat_w"], dtype=torch.float32, device=device
                    )
                )
                body_lin_vel_w_list.append(
                    torch.tensor(
                        data["body_lin_vel_w"], dtype=torch.float32, device=device
                    )
                )
                body_ang_vel_w_list.append(
                    torch.tensor(
                        data["body_ang_vel_w"], dtype=torch.float32, device=device
                    )
                )
                motion_group_list.append(
                    torch.full(
                        (num_frames, 1),
                        motion_group_index,
                        dtype=torch.float32,
                        device=device,
                    )
                )
                motion_id_list.append(
                    torch.full(
                        (num_frames, 1),
                        self.num_motions + local_motion_id,
                        dtype=torch.float32,
                        device=device,
                    )
                )
                self.motion_lengths.append(num_frames)

            self.extracted_list.extend(extracted_list)
            self.group_names.append(group_name)
            self.num_motions += len(normalized_paths)
            motion_group_index += 1

        assert self.num_motions > 0, "At least one motion file is required."

        self.joint_pos = torch.cat(joint_pos_list, dim=0)
        self.joint_vel = torch.cat(joint_vel_list, dim=0)
        self._body_pos_w = torch.cat(body_pos_w_list, dim=0)
        self._body_quat_w = torch.cat(body_quat_w_list, dim=0)
        self._body_lin_vel_w = torch.cat(body_lin_vel_w_list, dim=0)
        self._body_ang_vel_w = torch.cat(body_ang_vel_w_list, dim=0)
        self._motion_id = torch.cat(motion_id_list, dim=0)
        self._motion_group = torch.cat(motion_group_list, dim=0)

        self.body_pos_w = self._body_pos_w[:, self._body_indexes]
        self.body_quat_w = self._body_quat_w[:, self._body_indexes]
        self.body_lin_vel_w = self._body_lin_vel_w[:, self._body_indexes]
        self.body_ang_vel_w = self._body_ang_vel_w[:, self._body_indexes]

        self.time_step_total = self.joint_pos.shape[0]
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
        assert (
            self.valid_center_indices.numel() > 0
        ), "No valid center frames found for the configured window size."

    def extract_part(self,path: str) -> str | None:
        """Extract an artifact-relative motion path."""
        if path.startswith("artifacts/"):
            relative_path = path[len("artifacts/") :]
            if relative_path.endswith(".npz"):
                return relative_path
        return None
    
    def _normalize_paths(self, paths: list[str] | str) -> list[str]:
        """Convert a path input to a normalized list."""
        if isinstance(paths, str):
            return [paths]
        return list(paths)

    def _validate_motion_file(self, motion_path: str) -> None:
        """Ensure the referenced motion file exists."""
        assert os.path.isfile(motion_path), f"Invalid file path: {motion_path}"

    def _validate_fps(self, data: np.lib.npyio.NpzFile) -> None:
        """Ensure all loaded motions share the same fps."""
        if self.fps is None:
            self.fps = data["fps"]
        else:
            assert self.fps == data["fps"], "All motion files must have the same fps."

    def _build_motion_indices(self, device: str) -> torch.Tensor:
        """Build `[start, end)` index ranges for each motion segment."""
        motion_indices = torch.zeros(
            self.num_motions, 2, dtype=torch.long, device=device
        )
        start = 0
        for motion_id, length in enumerate(self.motion_lengths):
            end = start + length
            motion_indices[motion_id] = torch.tensor(
                [start, end], dtype=torch.long, device=device
            )
            start = end
        return motion_indices

    def _build_new_data_flag(self, device: str) -> torch.Tensor:
        """Mark the first frame of every trajectory except the first one."""
        new_data_flag = torch.zeros(
            self.time_step_total, dtype=torch.bool, device=device
        )
        cumulative_length = 0
        for motion_id, length in enumerate(self.motion_lengths):
            if motion_id > 0:
                new_data_flag[cumulative_length] = True
            cumulative_length += length
        return new_data_flag

    def _build_valid_center_mask(self, device: str) -> torch.Tensor:
        """Mark frame indices that can serve as valid window centers.

        A frame is valid when the full temporal window `[t - n, ..., t + m]`
        stays inside the same trajectory.

        Args:
            device: Device used for the output tensor.

        Returns:
            Boolean tensor over the concatenated global timeline.
        """
        valid_center_mask = torch.zeros(
            self.time_step_total, dtype=torch.bool, device=device
        )
        for motion_id in range(self.num_motions):
            start, end = self.motion_indices[motion_id]
            valid_start = start + self.history_frames
            valid_end = end - self.future_frames
            if valid_start < valid_end:
                valid_center_mask[valid_start:valid_end] = True
        return valid_center_mask

