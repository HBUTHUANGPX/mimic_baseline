import os
import numpy as np
from pathlib import Path
from collections.abc import Sequence


class MotionLoader:
    def __init__(
        self, 
        motion_file: str, 
        robot_body_names: Sequence[int] | None = None,
        robot_joint_names: Sequence[int] | None = None,
        body_indexes: Sequence[int] | None = None, 
        desire_human_joint_names: Sequence[int] | None = None,
        history_frames: int = 0,
        future_frames: int = 0,
        device: str = "cpu",
        use_token: bool = False,
    ):

        if isinstance(motion_file, str):
            self.motion_file = [motion_file]
        else:
            self.motion_file = motion_file
        for file in self.motion_file:
            assert os.path.isfile(file), f"Invalid file path: {file}"
        self.motion_lengths = []  # Length of each motion segment
        self.fps = None  # Assume all files have the same fps
        self.file_joint_names = None
        self.file_body_names = None
        self.human_joint_names = None
        self.human_joint_indexes = None
        self._robot_body_names = robot_body_names
        self._robot_joint_names = robot_joint_names
        self.desire_human_joint_names = desire_human_joint_names
        self._body_indexes = body_indexes
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.window_size = history_frames + future_frames + 1

        self._prepare_np_list()
        for _file in self.motion_file:
            data = np.load(_file)
            p = Path(_file)
            if self.fps is None:
                self.fps = data["fps"]
            else:
                assert (
                    self.fps == data["fps"]
                ), "All motion files must have the same fps."
            self._validate_joint_names(data)
            self._validate_link_names(data)
            self._validate_human_joint_names(data)
            self._append_motion_data(data)

            num_frames = self.np_joint_pos_list[-1].shape[0]
            if use_token:
                token = np.load(p.with_name(p.stem + "_token.tknpz"), allow_pickle=True)
                self._validate_token_data(token, _file, num_frames)
                self._append_token_data(token)
            
        # Concatenate along time dimension (dim=0)
        self.joint_pos = np.concatenate(self.np_joint_pos_list, axis=0)
        self.joint_vel = np.concatenate(self.np_joint_vel_list, axis=0)
        self.body_pos_w = np.concatenate(self.np_body_pos_w_list, axis=0)[:, self._body_indexes]
        self.body_quat_w = np.concatenate(self.np_body_quat_w_list, axis=0)[
            :, self._body_indexes
        ]
        self.body_lin_vel_w = np.concatenate(self.np_body_lin_vel_w_list, axis=0)[
            :, self._body_indexes
        ]
        self.body_ang_vel_w = np.concatenate(self.np_body_ang_vel_w_list, axis=0)[
            :, self._body_indexes
        ]
        self.human_body_pos_w = np.concatenate(self.np_human_body_pos_w_list, axis=0)
        self.human_body_quat_w = np.concatenate(self.np_human_body_quat_w_list, axis=0)
        self.human_joint_quat = np.concatenate(self.np_human_joint_quat_list, axis=0)
        if use_token:
            self.actor_q_human = np.concatenate(self.np_actor_q_human_list, axis=0)
            self.actor_q_robot = np.concatenate(self.np_actor_q_robot_list, axis=0)
        print("motion clips:")
        print("self.joint_pos.shape: ", self.joint_pos.shape)
        print("self.joint_vel.shape: ", self.joint_vel.shape)
        print("self.body_pos_w.shape: ", self.body_pos_w.shape)
        print("self.body_quat_w.shape: ", self.body_quat_w.shape)
        print("self.body_lin_vel_w.shape: ", self.body_lin_vel_w.shape)
        print("self.body_ang_vel_w.shape: ", self.body_ang_vel_w.shape)
        print("self.human_body_pos_w.shape: ", self.human_body_pos_w.shape)
        print("self.human_body_quat_w.shape: ", self.human_body_quat_w.shape)
        print("self.human_joint_quat.shape: ", self.human_joint_quat.shape)
        self.time_step_total = self.joint_pos.shape[0]

    def _prepare_np_list(self):
        self.np_joint_pos_list: list[np.ndarray] = []
        self.np_joint_vel_list: list[np.ndarray] = []
        self.np_body_pos_w_list: list[np.ndarray] = []
        self.np_body_quat_w_list: list[np.ndarray] = []
        self.np_body_lin_vel_w_list: list[np.ndarray] = []
        self.np_body_ang_vel_w_list: list[np.ndarray] = []
        self.np_human_body_pos_w_list: list[np.ndarray] = []
        self.np_human_body_quat_w_list: list[np.ndarray] = []
        self.np_human_joint_quat_list: list[np.ndarray] = []
        self.np_actor_q_human_list: list[np.ndarray] = []
        self.np_actor_q_robot_list: list[np.ndarray] = []

    def _append_motion_data(self, data: np.lib.npyio.NpzFile) -> None:
        self.np_joint_pos_list.append(
            np.asarray(data["robot_joint_pos"], dtype=np.float32)[
                :, self._robot_joint_indexes
            ]
        )
        self.np_joint_vel_list.append(
            np.asarray(data["robot_joint_vel"], dtype=np.float32)[
                :, self._robot_joint_indexes
            ]
        )
        self.np_body_pos_w_list.append(
            np.asarray(data["robot_body_pos"], dtype=np.float32)[
                :, self._robot_body_indexes
            ]
        )
        self.np_body_quat_w_list.append(
            self._quat_to_wxyz(
                np.asarray(data["robot_body_quat"], dtype=np.float32)[
                    :, self._robot_body_indexes
                ],
                data,
            )
        )
        self.np_body_lin_vel_w_list.append(
            np.asarray(data["robot_body_lin_vel"], dtype=np.float32)[
                :, self._robot_body_indexes
            ]
        )
        self.np_body_ang_vel_w_list.append(
            np.asarray(data["robot_body_ang_vel"], dtype=np.float32)[
                :, self._robot_body_indexes
            ]
        )
        self.np_human_body_pos_w_list.append(
            np.asarray(data["human_global_pos"], dtype=np.float32)[
                :, self.human_joint_indexes
            ]
        )
        self.np_human_body_quat_w_list.append(
            self._quat_to_wxyz(
                np.asarray(data["human_global_quat"], dtype=np.float32)[
                    :, self.human_joint_indexes
                ],
                data,
            )
        )
        self.np_human_joint_quat_list.append(
            self._quat_to_wxyz(
                np.asarray(data["human_local_transforms"], dtype=np.float32)[
                    :, self.human_joint_indexes, 3:7
                ],
                data,
            )
        )

    def _quat_to_wxyz(self, quat: np.ndarray, data: np.lib.npyio.NpzFile) -> np.ndarray:
        scalar_first = bool(data["scalar_first"]) if "scalar_first" in data else False
        return quat if scalar_first else quat[..., [3, 0, 1, 2]]

    def _validate_human_joint_names(self, data: np.lib.npyio.NpzFile) -> None:
        if self.human_joint_names is None:
            self.human_joint_names = data["human_joint_names"].tolist()
            self.human_joint_indexes = [
                self.human_joint_names.index(name)
                for name in self.desire_human_joint_names
            ]
            # print("human_joint_names:\r\n",self.human_joint_names)
        else:
            human_joint_names = data["human_joint_names"].tolist()
            assert (
                self.human_joint_names == human_joint_names
            ), f"Motion file human joint names {human_joint_names} do not match expected {self.human_joint_names}."

    def _validate_joint_names(self, data: np.lib.npyio.NpzFile) -> None:
        """Ensure the motion file contains the required joint names."""
        if self.file_joint_names is None:
            self.file_joint_names = data["robot_joint_names"].tolist()
            # 将file中的关节数据转换为仿真器的关节顺序,先获得索引
            self._robot_joint_indexes = [
                self.file_joint_names.index(name) for name in self._robot_joint_names
            ]
        else:
            file_joint_names = data["robot_joint_names"].tolist()
            assert (
                self.file_joint_names == file_joint_names
            ), f"Motion file joint names {file_joint_names} do not match expected {self.file_joint_names}."

    def _validate_link_names(self, data: np.lib.npyio.NpzFile) -> None:
        """Ensure the motion file contains the required link names."""
        if self.file_body_names is None:
            self.file_body_names = data["robot_body_names"].tolist()
            print("robot_body_names", self._robot_body_names)
            print("file_body_names", self.file_body_names)
            self._robot_body_indexes = [
                self.file_body_names.index(name) for name in self._robot_body_names
            ]
        else:
            file_body_names = data["robot_body_names"].tolist()
            assert (
                self.file_body_names == file_body_names
            ), f"Motion file body names {file_body_names} do not match expected {self.file_body_names}."

    def _validate_token_data(
        self,
        token: np.lib.npyio.NpzFile,
        motion_path: str,
        num_frames: int,
    ) -> None:
        required_fields = (
            "actor_q_human",
            "actor_q_robot",
            "critic_q_human",
            "critic_q_robot",
        )
        for field in required_fields:
            assert field in token, f"{motion_path} 对应 token 文件缺少字段 {field}."
            assert (
                token[field].ndim == 2
            ), f"{motion_path} 对应 token 字段 {field} 必须是 [num_frames, latent_dim]."
            assert (
                token[field].shape[0] == num_frames
            ), (
                f"{motion_path} 对应 token 字段 {field} 帧数为 "
                f"{token[field].shape[0]}，motion 帧数为 {num_frames}."
            )

    def _append_token_data(self, token: np.lib.npyio.NpzFile) -> None:
        self.np_actor_q_human_list.append(token["actor_q_human"].astype(np.float32))
        self.np_actor_q_robot_list.append(token["actor_q_robot"].astype(np.float32))