import numpy as np
import matplotlib.pyplot as plt
import math

# 定义NPZ文件路径
file_path = "/home/hpx/HPX_LOCO_2/mimic_baseline/deploy_mujoco/motion.npz"
target_joints = [
    "L_hip_roll_joint",
    "L_hip_yaw_joint",
    "L_hip_pitch_joint",
    "L_knee_joint",
    "L_ankle_pitch_joint",
    "L_ankle_roll_joint",
    # "pelvis_joint",
    # "L_shoulder_pitch_joint",
    # "L_shoulder_roll_joint",
    # "L_shoulder_yaw_joint",
    # "L_elbow_joint",
    # "L_forearm_yaw_joint",
    # "L_wrist_roll_joint",
    # "L_wrist_pitch_joint",
]
try:
    # 加载NPZ文件
    data = np.load(file_path)

    # 提取qvel数据
    qvel = data["dof_velocities"]

    # 提取qfrc_actuator数据
    qfrc_actuator = data["dof_torque"]
    dof_names = data["dof_names"]
    # 示例：打印形状以验证数据
    print("qvel shape:", qvel.shape)
    print("qfrc_actuator shape:", qfrc_actuator.shape)
    joint_length = len(target_joints)

    # 计算行数和列数：找到最小cols使得rows = cols - 1，且rows * cols >= num_joints
    cols = math.ceil(
        (1 + math.sqrt(1 + 4 * joint_length)) / 2
    )  # 基于二次方程求解最小cols
    rows = cols - 1
    while rows * cols < joint_length:
        cols += 1
        rows = cols - 1

    # 创建subplots
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axs = axs.flatten() if joint_length > 1 else [axs]  # 展平以便索引

    print("enumerate")
    i = 0
    for name in dof_names:
        if name not in target_joints:
            continue
        velocities = qvel[:, i]
        efforts = qfrc_actuator[:, i]

        axs[i].scatter(velocities, efforts, color="blue", label="data dot", alpha=0.04)
        # axs[i].plot(velocities, efforts, color="red", linestyle="--", label="line")
        axs[i].set_xlabel("velocity, rad/s")
        axs[i].set_ylabel("effort, Nm")
        axs[i].set_title(f"{name} effort-velocity")
        axs[i].grid(True)
        axs[i].legend()
        i += 1

    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("文件未找到，请检查路径是否正确。")
except KeyError as e:
    print(f"键 {e} 不存在于NPZ文件中。")
except Exception as e:
    print(f"加载文件时发生错误: {e}")
