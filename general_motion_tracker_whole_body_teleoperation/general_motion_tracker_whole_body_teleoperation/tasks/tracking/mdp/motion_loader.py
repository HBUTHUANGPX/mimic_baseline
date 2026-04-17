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
        motion_file_group: dict[str, list[str] | str] | str,
        body_indexes: Sequence[int] | None = None,
        body_names: Sequence[int] | None = None,
        history_frames: int = 0,
        future_frames: int = 0,
        device: str = "cpu",
    ) -> None:
        if isinstance(motion_file_group, str):
            motion_file_group = {"default": motion_file_group}
        
        self.group_names: list[str] = []
        self.extracted_list: list[str] = []
        self.motion_lengths: list[int] = []
        self.num_motions = 0
        self.fps = None
        self._body_indexes = body_indexes
        self._body_names = body_names
        # TODO:
        if body_indexes is None and body_names is None:
            ...# 不能都为None，否则无法确定要加载哪些身体部位的数据
        elif body_indexes is not None and body_names is not None:
            ...# 不能同时指定索引和名称，否则可能会出现冲突或不一致的情况，导致加载错误的数据或引发混淆,
            ...# 此时抛出warn,建议用户指定一种方式来选择身体部位的数据
            ...# 此时实际的self._body_indexes由 body_indexes 确定
        elif body_indexes is not None and body_names is None:
            ...# 此时实际的self._body_indexes由 body_indexes 确定
        elif body_indexes is None and body_names is not None:
            ...# 此时实际的self._body_indexes由 body_names 确定,但需要确定后续的motion文件中含有相应的key
        
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.window_size = history_frames + future_frames + 1

        self._prepare_np_list()

        motion_group_list: list[int] = []
        motion_id_list: list[int] = []
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
                self._append_motion_data(data)

                num_frames = self.np_joint_pos_list[-1].shape[0]
                self.motion_lengths.append(num_frames)

                motion_group_list.extend([motion_group_index] * num_frames)
                motion_id_list.extend([self.num_motions + local_motion_id] * num_frames)

            self.extracted_list.extend(extracted_list)
            self.group_names.append(group_name)
            self.num_motions += len(normalized_paths)
            motion_group_index += 1

        assert self.num_motions > 0, "At least one motion file is required."

        self._motion_data_np_list_to_tensor(device)

        self._motion_id = torch.tensor(
            motion_id_list, dtype=torch.long, device=device
        ).unsqueeze(1)
        self._motion_group = torch.tensor(
            motion_group_list, dtype=torch.long, device=device
        ).unsqueeze(1)

        self.time_step_total = self.joint_pos.shape[0]
        self.motion_indices = self._build_motion_indices(device)
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

    def _prepare_np_list(self):
        self.np_joint_pos_list: list[np.ndarray] = []
        self.np_joint_vel_list: list[np.ndarray] = []
        self.np_body_pos_w_list: list[np.ndarray] = []
        self.np_body_quat_w_list: list[np.ndarray] = []
        self.np_body_lin_vel_w_list: list[np.ndarray] = []
        self.np_body_ang_vel_w_list: list[np.ndarray] = []

    def _append_motion_data(self, data: np.lib.npyio.NpzFile) -> None:
        self.np_joint_pos_list.append(data["joint_pos"].astype(np.float32))
        self.np_joint_vel_list.append(data["joint_vel"].astype(np.float32))
        self.np_body_pos_w_list.append(data["body_pos_w"].astype(np.float32))
        self.np_body_quat_w_list.append(data["body_quat_w"].astype(np.float32))
        self.np_body_lin_vel_w_list.append(data["body_lin_vel_w"].astype(np.float32))
        self.np_body_ang_vel_w_list.append(data["body_ang_vel_w"].astype(np.float32))

    def _motion_data_np_list_to_tensor(self, device: str) -> None:
        self.joint_pos = self.np_list_to_tensor(self.np_joint_pos_list, device)
        self.joint_vel = self.np_list_to_tensor(self.np_joint_vel_list, device)
        self.body_pos_w = self.np_list_to_tensor(self.np_body_pos_w_list, device)[
            :, self._body_indexes
        ]
        self.body_quat_w = self.np_list_to_tensor(self.np_body_quat_w_list, device)[
            :, self._body_indexes
        ]
        self.body_lin_vel_w = self.np_list_to_tensor(
            self.np_body_lin_vel_w_list, device
        )[:, self._body_indexes]
        self.body_ang_vel_w = self.np_list_to_tensor(
            self.np_body_ang_vel_w_list, device
        )[:, self._body_indexes]

    def np_list_to_tensor(self, np_list: list[np.ndarray], device: str) -> torch.Tensor:
        """Convert a NumPy array to a PyTorch tensor on the appropriate device."""
        return torch.from_numpy(np.concatenate(np_list, axis=0)).to(device)

    def extract_part(self, path: str) -> str | None:
        """Extract an artifact-relative motion path."""
        # if path.startswith("artifacts/"):
        #     relative_path = path[len("artifacts/") :]
        if path.endswith(".npz"):
            return path
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

class MotionLoader_human:
    def __init__(
        self,
        motion_file_group: dict[str, list[str] | str] | str,
        robot_body_names: Sequence[int] | None = None,
        robot_joint_names: Sequence[int] | None = None,
        body_indexes: Sequence[int] | None = None,
        desire_human_joint_names: Sequence[int] | None = None,
        history_frames: int = 0,
        future_frames: int = 0,
        device: str = "cpu",
    ) -> None:
        if isinstance(motion_file_group, str):
            motion_file_group = {"default": motion_file_group}
        self.device = device
        self.group_names: list[str] = []
        self.extracted_list: list[str] = []
        self.motion_lengths: list[int] = []
        self.num_motions = 0
        self.fps = None
        self.file_joint_names = None
        self.file_body_names = None
        self.human_joint_names = None
        self.human_joint_indexes = None
        # 传入仿真器的机器人模型中身体部位的名称和关节名称
        self._robot_body_names = robot_body_names
        self._robot_joint_names = robot_joint_names
        # 传入仿真器中需要加载的机器人身体部位的索引
        self._body_indexes = body_indexes
        self.desire_human_joint_names = desire_human_joint_names
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.window_size = history_frames + future_frames + 1

        self._prepare_np_list()

        motion_group_list: list[int] = []
        motion_id_list: list[int] = []
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
                self._validate_joint_names(data)
                self._validate_link_names(data)
                self._validate_human_joint_names(data)
                self._append_motion_data(data)

                num_frames = self.np_joint_pos_list[-1].shape[0]
                self.motion_lengths.append(num_frames)

                motion_group_list.extend([motion_group_index] * num_frames)
                motion_id_list.extend([self.num_motions + local_motion_id] * num_frames)

            self.extracted_list.extend(extracted_list)
            self.group_names.append(group_name)
            self.num_motions += len(normalized_paths)
            motion_group_index += 1

        assert self.num_motions > 0, "At least one motion file is required."
        
        self._motion_data_np_list_to_tensor()

        self._motion_id = torch.tensor(
            motion_id_list, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        self._motion_group = torch.tensor(
            motion_group_list, dtype=torch.long, device=self.device
        ).unsqueeze(1)

        self.time_step_total = self.joint_pos.shape[0]
        self.motion_indices = self._build_motion_indices()
        self.window_offsets = torch.arange(
            -self.history_frames,
            self.future_frames + 1,
            dtype=torch.long,
            device=self.device,
        )
        self.valid_center_mask = self._build_valid_center_mask()
        self.valid_center_indices = torch.nonzero(
            self.valid_center_mask, as_tuple=False
        ).squeeze(-1)
        assert (
            self.valid_center_indices.numel() > 0
        ), "No valid center frames found for the configured window size."

    def _prepare_np_list(self):
        self.np_joint_pos_list: list[np.ndarray] = []
        self.np_joint_vel_list: list[np.ndarray] = []
        self.np_body_pos_w_list: list[np.ndarray] = []
        self.np_body_quat_w_list: list[np.ndarray] = []
        self.np_body_lin_vel_w_list: list[np.ndarray] = []
        self.np_body_ang_vel_w_list: list[np.ndarray] = []
        self.np_human_body_pos_w_list: list[np.ndarray] = []
        self.np_human_body_quat_w_list: list[np.ndarray] = []

    def _append_motion_data(self, data: np.lib.npyio.NpzFile) -> None:
        self.np_joint_pos_list.append(data["robot_joint_pos"].astype(np.float32)[:, self._robot_joint_indexes])
        self.np_joint_vel_list.append(data["robot_joint_vel"].astype(np.float32)[:, self._robot_joint_indexes])
        self.np_body_pos_w_list.append(data["robot_body_pos"].astype(np.float32)[:, self._robot_body_indexes])
        self.np_body_quat_w_list.append(data["robot_body_quat"].astype(np.float32)[:, self._robot_body_indexes][..., [3,0,1,2]])
        self.np_body_lin_vel_w_list.append(data["robot_body_lin_vel"].astype(np.float32)[:, self._robot_body_indexes])
        self.np_body_ang_vel_w_list.append(data["robot_body_ang_vel"].astype(np.float32)[:, self._robot_body_indexes])
        self.np_human_body_pos_w_list.append(data["human_global_pos"].astype(np.float32)[:,self.human_joint_indexes])
        self.np_human_body_quat_w_list.append(data["human_global_quat"].astype(np.float32)[:,self.human_joint_indexes][..., [3,0,1,2]])

    def _motion_data_np_list_to_tensor(self,) -> None:
        self.joint_pos = self.np_list_to_tensor(self.np_joint_pos_list)
        self.joint_vel = self.np_list_to_tensor(self.np_joint_vel_list)

        self.body_pos_w = self.np_list_to_tensor(self.np_body_pos_w_list)[:, self._body_indexes]

        self.body_quat_w = self.np_list_to_tensor(self.np_body_quat_w_list)[:, self._body_indexes]

        self.body_lin_vel_w = self.np_list_to_tensor(
            self.np_body_lin_vel_w_list
        )[:, self._body_indexes]

        self.body_ang_vel_w = self.np_list_to_tensor(
            self.np_body_ang_vel_w_list
        )[:, self._body_indexes]

        self.human_body_pos_w = self.np_list_to_tensor(self.np_human_body_pos_w_list)
        self.human_body_quat_w = self.np_list_to_tensor(self.np_human_body_quat_w_list)

    def np_list_to_tensor(self, np_list: list[np.ndarray]) -> torch.Tensor:
        """Convert a NumPy array to a PyTorch tensor on the appropriate device."""
        return torch.from_numpy(np.concatenate(np_list, axis=0)).to(self.device)

    def extract_part(self, path: str) -> str | None:
        """Extract an artifact-relative motion path."""
        # if path.startswith("artifacts/"):
        #     relative_path = path[len("artifacts/") :]
        if path.endswith(".npz"):
            return path
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

    def _validate_human_joint_names(self, data: np.lib.npyio.NpzFile) -> None:
        if self.human_joint_names is None:
            self.human_joint_names = data["human_joint_names"].tolist()
            self.human_joint_indexes = [self.human_joint_names.index(name) for name in self.desire_human_joint_names]
            # print("human_joint_names:\r\n",self.human_joint_names)
        else:
            human_joint_names = data["human_joint_names"].tolist()
            assert self.human_joint_names == human_joint_names, (
                f"Motion file human joint names {human_joint_names} do not match expected {self.human_joint_names}."
            )

    def _validate_joint_names(self, data: np.lib.npyio.NpzFile) -> None:
        """Ensure the motion file contains the required joint names."""
        if self.file_joint_names is None:
            self.file_joint_names = data["robot_joint_names"].tolist()
            # 将file中的关节数据转换为仿真器的关节顺序,先获得索引
            self._robot_joint_indexes = [self.file_joint_names.index(name) for name in self._robot_joint_names]
        else:
            file_joint_names = data["robot_joint_names"].tolist()
            assert self.file_joint_names == file_joint_names, (
                f"Motion file joint names {file_joint_names} do not match expected {self.file_joint_names}."
            )

    def _validate_link_names(self, data: np.lib.npyio.NpzFile) -> None:
        """Ensure the motion file contains the required link names."""
        if self.file_body_names is None:
            self.file_body_names = data["robot_body_names"].tolist()
            print("robot_body_names",self._robot_body_names)
            print("file_body_names",self.file_body_names)
            self._robot_body_indexes = [self.file_body_names.index(name) for name in self._robot_body_names]
        else:
            file_body_names = data["robot_body_names"].tolist()
            assert self.file_body_names == file_body_names, (
                f"Motion file body names {file_body_names} do not match expected {self.file_body_names}."
            )

    def _build_motion_indices(self) -> torch.Tensor:
        """Build `[start, end)` index ranges for each motion segment."""
        motion_indices = torch.zeros(
            self.num_motions, 2, dtype=torch.long, device=self.device
        )
        start = 0
        for motion_id, length in enumerate(self.motion_lengths):
            end = start + length
            motion_indices[motion_id] = torch.tensor(
                [start, end], dtype=torch.long, device=self.device
            )
            start = end
        return motion_indices

    def _build_valid_center_mask(self) -> torch.Tensor:
        """Mark frame indices that can serve as valid window centers.

        A frame is valid when the full temporal window `[t - n, ..., t + m]`
        stays inside the same trajectory.

        Returns:
            Boolean tensor over the concatenated global timeline.
        """
        valid_center_mask = torch.zeros(
            self.time_step_total, dtype=torch.bool, device=self.device
        )
        for motion_id in range(self.num_motions):
            start, end = self.motion_indices[motion_id]
            valid_start = start + self.history_frames
            valid_end = end - self.future_frames
            if valid_start < valid_end:
                valid_center_mask[valid_start:valid_end] = True
        return valid_center_mask

# Example usage:
if __name__ == "__main__":
    try:
        from scripts.rsl_rl.load_motion_file import collect_npz_paths
    except ModuleNotFoundError:
        import importlib.util
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[5]
        module_path = repo_root / "scripts" / "rsl_rl" / "load_motion_file.py"
        spec = importlib.util.spec_from_file_location("load_motion_file", module_path)
        if spec is None or spec.loader is None:
            raise ModuleNotFoundError(f"Unable to load module from {module_path}")
        load_motion_file = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(load_motion_file)
        collect_npz_paths = load_motion_file.collect_npz_paths

    motion_file_path = "scripts/rsl_rl/motion_file.yaml"
    motion_file_group = collect_npz_paths(motion_file_path)
    for group_name, paths in motion_file_group.items():
        print(f"\nGroup: {group_name}")
        print(f"[INFO] Collected {len(paths)} motion files for training.")
    robot_body_names = ["pelvis"]
    robot_joint_names = ["left_hip_pitch_joint"]
    ml_r = MotionLoader_human(
        motion_file_group=motion_file_group,
        robot_body_names=robot_body_names,
        robot_joint_names=robot_joint_names,
        body_indexes=[0, 1, 2],
        history_frames=2,
        future_frames=2,
        device="cuda:0",
    )
