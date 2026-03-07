import numpy as np
import mujoco
import mujoco.viewer
import time
from typing import Union, List, Dict, Tuple
import pickle
import os
import csv

from scipy.spatial.transform import Rotation as R

class cfg:
    only_leg_flag = False  # True, False
    with_wrist_flag = True  # True, False

class pkl_load_and_csv_save:
    def __init__(self,_rtg_data_file_path):
        with open(_rtg_data_file_path, "rb") as f:
            self.data_collection = pickle.load(f)

            self.frame_data = np.asarray(self.data_collection["frames"],dtype=np.float32)
            # [root position (3D), root rotation (3D), joint rotations]
            self.pos = self.frame_data[:, 0:3]
            self.rot = self.frame_data[:, 3:6]
            self.quat = R.from_rotvec(self.rot, degrees=False).as_quat(scalar_first=True) # w,x,y,z
            self.qj = self.frame_data[:, 6:]

    def save_as_csv(self,_csv_file):
        csv_file = _csv_file
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)

        if os.path.exists(csv_file):
            os.remove(csv_file)
        frames_len = len(self.data_collection["frames"])
        with open(csv_file, "w", newline="") as file:
            writer = csv.writer(file)
            result = np.concatenate(
                (
                    self.pos,
                    self.quat[:,[1,2,3,0]],#w,x,y,z to x,y,z,w
                    self.qj,
                ),
                axis=1,
            )
            for idx in range(frames_len):
                row = result[idx, :]
                writer.writerow(row)

        print(f"数据已写入 {csv_file}")
        return True
    
class mujoco_displayanimanim(pkl_load_and_csv_save):
    def __init__(self, _robot_file, _rtg_data_file_path):
        super().__init__(_rtg_data_file_path)
        self.robot_file = _robot_file
        self.spec = mujoco.MjSpec.from_file(self.robot_file)
        # self._rehandle_xml()
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        a = 1

    def compensate_displacements(self,quaternions, displacements):
        """
        计算第一个四元数的 z 轴转角，并对所有位移值应用补偿旋转（反向旋转）。
        参数：
            quaternions: n x 4 的 NumPy 数组，每行是一个四元数 [w, x, y, z]
            displacements: n x 3 的 NumPy 数组，每行是一个位移向量 [x, y, z]
        返回：
            新的 n x 3 数组，所有位移值已被补偿旋转
        """
        # 确保输入形状正确
        quaternions = np.array(quaternions, dtype=float)
        displacements = np.array(displacements, dtype=float)
        if quaternions.shape[1] != 4 or displacements.shape[1] != 3 or quaternions.shape[0] != displacements.shape[0]:
            raise ValueError("输入数组必须是 n x 4 的四元数和 n x 3 的位移")

        # 获取第一个四元数并归一化
        q1 = quaternions[0] / np.linalg.norm(quaternions[0])
        w, x, y, z = q1

        # 计算 z 轴转角 θ（考虑符号和象限）
        theta = 2 * np.arctan2(z, w)

        # 补偿旋转角度为 -theta
        cos_neg_theta = np.cos(-theta)
        sin_neg_theta = np.sin(-theta)

        # 构造绕 z 轴的旋转矩阵 R(-theta)
        rotation_matrix = np.array([
            [cos_neg_theta, -sin_neg_theta, 0],
            [sin_neg_theta, cos_neg_theta, 0],
            [0, 0, 1]
        ])

        # 对所有位移向量应用旋转（矢量化操作，更高效）
        compensated_displacements = np.dot(displacements, rotation_matrix.T)  # 使用 .T 因为矩阵是行向量形式

        return compensated_displacements
    
    def compensate_z_rotation(self, quaternions):
        """
        补偿 n x 4 四元数数组中所有四元数的 z 轴转角，使其等于第一个四元数的 z 轴转角归零。
        参数：
            quaternions: n x 4 的 NumPy 数组，每行是一个四元数 [w, x, y, z]
        返回：
            新的 n x 4 数组，所有四元数的 z 轴转角已被补偿
        """
        # 确保输入是 n x 4 的 NumPy 数组
        quaternions = np.array(quaternions, dtype=np.float32)
        if quaternions.shape[1] != 4:
            raise ValueError("输入数组必须是 n x 4 的形状")

        # 获取第一个四元数并归一化
        q1 = quaternions[0] / np.linalg.norm(quaternions[0])
        w, x, y, z = q1

        # 计算 z 轴转角
        # 四元数绕 z 轴旋转的表示为 [cos(θ/2), 0, 0, sin(θ/2)]
        # 比较 q1 和 z 轴旋转四元数，提取 θ
        theta = 2 * np.arctan2(z, w)  # z 轴旋转角度

        # 构造反向旋转四元数（绕 z 轴旋转 -θ）
        cos_half_theta = np.cos(-theta / 2)
        sin_half_theta = np.sin(-theta / 2)
        q_comp = np.array([cos_half_theta, 0.0, 0.0, sin_half_theta])

        # 四元数乘法函数
        def quaternion_multiply(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            return np.array([w, x, y, z])

        # 对每个四元数应用补偿旋转
        result = np.zeros_like(quaternions)
        for i in range(quaternions.shape[0]):
            # 归一化当前四元数
            q = quaternions[i] / np.linalg.norm(quaternions[i])
            # 应用补偿旋转：q_new = q_comp * q
            result[i] = quaternion_multiply(q_comp, q)
            # 确保结果四元数是单位四元数
            result[i] = result[i] / np.linalg.norm(result[i])

        return result

    def animate_bvh(self):
        # 动画播放
        start_bias = 0
        frames_len = len(self.data_collection["frames"])-start_bias
        # frames_len = 3700
        frame_time = 1 / self.data_collection["fps"]
        # print(np.mean(self.data_collection["dof_pos"],axis=1))
        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            # show_left_ui=False,
            # show_right_ui=False
        ) as self.viewer:
            frame_idx = start_bias
            while self.viewer.is_running() and frame_idx < frames_len:
                self.data.qpos[0:3] = self.pos[frame_idx, :]
                # self.data.qpos[3:7] = self.data_collection["root_rot"][frame_idx, :]
                self.data.qpos[3:7] = self.quat[frame_idx, :]
                # print(R.from_quat(self.data_collection["root_rot"][frame_idx, :],scalar_first=True).as_rotvec(degrees=True))
                self.data.qpos[7:] = self.qj[frame_idx, :]
                self.data.qvel[:] = 0
                time.sleep(frame_time)
                mujoco.mj_forward(self.model, self.data)
                self.viewer.sync()
                frame_idx += 1
                if frame_idx % frames_len == 0:
                    break

import argparse
import multiprocessing as mp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkl_file",
        help="BVH motion file to load.",
        default="",
        type=str,
    )

    parser.add_argument(
        "--retargeting_data_folder",
        help="BVH motion file to load.",
        default="",
        type=str,
    )

    parser.add_argument(
        "--csv_folder",
        help="BVH motion file to load.",
        default="",
        type=str,
    )
    parser.add_argument("--num_cpus", default=4, type=int)
    args = parser.parse_args()


    print(f"Total CPUs: {mp.cpu_count()}")
    print(f"Using {args.num_cpus} CPUs.")

    robot_xml_file_name = (
        # "/home/hpx/HPX_LOCO_2/GMR/assets/unitree_h1_2/h1_2_handless.xml"
        # "/home/hpx/HPX_LOCO_2/GMR/assets/rotaku_xs3/mjcf/rotaku_xs3_rl.xml"
        # "/home/hpx/HPX_LOCO_2/GMR/assets/Q1/mjcf/Q1_wo_hand.xml"
        "/home/hpx/HPX_LOCO_2/mimic_baseline/general_motion_tracker_whole_body_teleoperation/general_motion_tracker_whole_body_teleoperation/assets/unitree_g1/g1_29dof_rev_1_0.xml"
        # "/home/hpx/HPX_LOCO_2/GMR/assets/h1_2/h1_2_wo_hand.xml"
    )
    motion = "g1_walk"
    retargeting_data_file_name = (
        # "/home/hpx/HPX_LOCO_2/mimic_baseline/scripts/mimickit_data_read/MimicKit_Data/motions/g1/g1_cartwheel.pkl"
        # "scripts/mimickit_data_read/MimicKit_Data/motions/g1/g1_double_kong.pkl"
        "scripts/mimickit_data_read/MimicKit_Data/motions/g1/"+motion+".pkl"
    )
    # 7+27
    d = mujoco_displayanimanim(robot_xml_file_name, retargeting_data_file_name)
    d.save_as_csv("/home/hpx/HPX_LOCO_2/mimic_baseline/scripts/mimickit_data_read/MimicKit_Data/motions/g1/"+motion+".csv")
    d.animate_bvh()

# python scripts/mimickit_data_read/g1_pkl_mj_view.py
# python scripts/csvs_to_npzs_2.py --input_folder scripts/mimickit_data_read/MimicKit_Data/motions/g1/ --output_folder artifacts/g1/mimic_kit/ --input_fps 60 --output_fps 50 --headless --preload_csv --async_save_npz --robot g1