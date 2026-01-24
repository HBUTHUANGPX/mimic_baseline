import mujoco
import numpy as np

# ==================== 加载模型 ====================
model = mujoco.MjModel.from_xml_path("deploy_mujoco/assets/Q1/mjcf/Q1_wo_hand.xml")
data = mujoco.MjData(model)

# ==================== 计算复合刚体惯性 ====================
mujoco.mj_crb(model, data)  # 填充 data.crb

# ==================== 遍历并计算每个 hinge joint 的转动惯量 ====================
print("每个旋转关节下游子树关于其轴的转动惯量（kg·m²）：\n")
for j in range(model.njnt):
    if model.jnt_type[j] == 3:  # 3 == mjJNT_HINGE (旋转关节)
        # 关节名称
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if not joint_name:
            joint_name = f"joint_{j}"

        # 该关节对应的 body（子树根）
        body_id = model.jnt_bodyid[j]

        # 旋转轴（已在模型中归一化）
        axis = model.jnt_axis[3*j:3*j+3]

        # 复合惯性数据（10 个元素/ body）
        offset = 10 * body_id
        I_flat = data.crb[offset + 4: offset + 10]  # [Ixx, Ixy, Ixz, Iyy, Iyz, Izz]

        # 重构对称惯性矩阵
        I = np.array([
            [I_flat[0], I_flat[1], I_flat[2]],
            [I_flat[1], I_flat[3], I_flat[4]],
            [I_flat[2], I_flat[4], I_flat[5]]
        ])

        # 计算标量转动惯量
        inertia = float(np.dot(axis, np.dot(I, axis)))

        print(f"{joint_name:25s} : {inertia:.6f}")