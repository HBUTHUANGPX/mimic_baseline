import os
import numpy as np
from collections.abc import Sequence

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
                np.asarray(data["joint_pos"], dtype=np.float32)
            )
            joint_vel_list.append(
                np.asarray(data["joint_vel"], dtype=np.float32)
            )
            body_pos_w_list.append(
                np.asarray(data["body_pos_w"], dtype=np.float32)
            )
            body_quat_w_list.append(
                np.asarray(data["body_quat_w"], dtype=np.float32)
            )
            body_lin_vel_w_list.append(
                np.asarray(data["body_lin_vel_w"], dtype=np.float32)
            )
            body_ang_vel_w_list.append(
                np.asarray(data["body_ang_vel_w"], dtype=np.float32)
            )
        # Concatenate along time dimension (dim=0)
        self.joint_pos = np.concatenate(joint_pos_list, axis=0)
        self.joint_vel = np.concatenate(joint_vel_list, axis=0)
        self._body_pos_w = np.concatenate(body_pos_w_list, axis=0)
        self._body_quat_w = np.concatenate(body_quat_w_list, axis=0)
        self._body_lin_vel_w = np.concatenate(body_lin_vel_w_list, axis=0)
        self._body_ang_vel_w = np.concatenate(body_ang_vel_w_list, axis=0)
        print("motion clips:")
        print("self.joint_pos.shape: ",self.joint_pos.shape)
        print("self.joint_vel.shape: ",self.joint_vel.shape)
        print("self._body_pos_w.shape: ",self._body_pos_w.shape)
        print("self._body_quat_w.shape: ",self._body_quat_w.shape)
        print("self._body_lin_vel_w.shape: ",self._body_lin_vel_w.shape)
        print("self._body_ang_vel_w.shape: ",self._body_ang_vel_w.shape)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> np.ndarray:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> np.ndarray:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> np.ndarray:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> np.ndarray:
        return self._body_ang_vel_w[:, self._body_indexes]
