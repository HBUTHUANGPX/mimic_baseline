import os
import numpy as np
from collections.abc import Sequence
import torch

class MotionLoader:
    def __init__(
        self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"
    ):
        

        if isinstance(motion_file, str):
            self.motion_file = [motion_file]
        else:
            self.motion_file = motion_file
        for file in self.motion_file:
            assert os.path.isfile(file), f"Invalid file path: {file}"
        # Load and concatenate data from all files
        joint_pos_list = []
        joint_vel_list = []
        body_pos_w_list = []
        body_quat_w_list = []
        body_lin_vel_w_list = []
        body_ang_vel_w_list = []
        self.motion_lengths = []  # Length of each motion segment
        self.fps = None  # Assume all files have the same fps

        for _file in self.motion_file:
            data = np.load(_file)
            if self.fps is None:
                self.fps = data["fps"]
            else:
                assert (
                    self.fps == data["fps"]
                ), "All motion files must have the same fps."

            joint_pos_list.append(
                torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
            )
            joint_vel_list.append(
                torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
            )
            body_pos_w_list.append(
                torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
            )
            body_quat_w_list.append(
                torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
            )
            body_lin_vel_w_list.append(
                torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
            )
            body_ang_vel_w_list.append(
                torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
            )
        # Concatenate along time dimension (dim=0)
        self.joint_pos = torch.cat(joint_pos_list, dim=0)
        self.joint_vel = torch.cat(joint_vel_list, dim=0)
        self._body_pos_w = torch.cat(body_pos_w_list, dim=0)
        self._body_quat_w = torch.cat(body_quat_w_list, dim=0)
        self._body_lin_vel_w = torch.cat(body_lin_vel_w_list, dim=0)
        self._body_ang_vel_w = torch.cat(body_ang_vel_w_list, dim=0)

        # assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        # data = np.load(motion_file)
        # self.fps = data["fps"]
        # self.joint_pos = torch.tensor(
        #     data["joint_pos"], dtype=torch.float32, device=device
        # )
        # self.joint_vel = torch.tensor(
        #     data["joint_vel"], dtype=torch.float32, device=device
        # )
        # self._body_pos_w = torch.tensor(
        #     data["body_pos_w"], dtype=torch.float32, device=device
        # )
        # self._body_quat_w = torch.tensor(
        #     data["body_quat_w"], dtype=torch.float32, device=device
        # )
        # self._body_lin_vel_w = torch.tensor(
        #     data["body_lin_vel_w"], dtype=torch.float32, device=device
        # )
        # self._body_ang_vel_w = torch.tensor(
        #     data["body_ang_vel_w"], dtype=torch.float32, device=device
        # )
        print("self.joint_pos.shape: ",self.joint_pos.shape)
        print("self.joint_vel.shape: ",self.joint_vel.shape)
        print("self._body_pos_w.shape: ",self._body_pos_w.shape)
        print("self._body_quat_w.shape: ",self._body_quat_w.shape)
        print("self._body_lin_vel_w.shape: ",self._body_lin_vel_w.shape)
        print("self._body_ang_vel_w.shape: ",self._body_ang_vel_w.shape)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

