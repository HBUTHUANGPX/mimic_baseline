import pinocchio as pin
from pinocchio.utils import zero
from pinocchio.robot_wrapper import RobotWrapper
"""
conda install pinocchio -c conda-forge
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

"""
export PYTHONPATH=""
"""
class pin_mj:
    def __init__(self, _cfg):
        # ========== 1. 准备Pinocchio模型 ==========
        self.robot: RobotWrapper = RobotWrapper.BuildFromURDF(
            _cfg.urdf_path, _cfg.asset_path, pin.JointModelFreeFlyer()
        )

        self.base_pos_world = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.base_quat_world = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def mujoco_to_pinocchio(
        self,
        joint_angles,
        base_pos=np.array([0.0, 0.0, 0.0], dtype=np.double),
        base_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.double),
    ):
        """
        将从Mujoco获取的机器人状态(基座位置、姿态、关节角)赋值到Pinocchio中。
        base_pos: np.array([x, y, z]) 基座在世界坐标系的位置
        base_quat: np.array([x, y, z, w]) 基座在世界坐标系的四元数 (Pinocchio默认的四元数顺序同为 [x,y,z,w])
        joint_angles: np.array([...]) 机器人关节角，长度为model.nq - 7(若有浮动基), 或 model.nq(若固定基)
        model, data: Pinocchio的model和data
        """

        q: np.ndarray = zero(
            self.robot.model.nq
        )  # 广义坐标 [7 + nJoints] (若 free-flyer)

        # 如果是浮动基模式，则前7维为 [x, y, z, q_x, q_y, q_z, q_w]
        # 注意：Pinocchio中free-flyer的顺序约定是 [xyz, qwxyz]
        # 若是固定基，则model.nq == 机器人关节数，无需设置基座
        if self.robot.model.joints[1].shortname() == "JointModelFreeFlyer":
            q[0:3] = base_pos
            q[3:7] = base_quat  # [x, y, z, w]
            # 后面是机器人关节
            q[7:] = joint_angles
        else:
            # 如果是固定基模型，则整段q都是关节
            q[:] = joint_angles

        # pin.forwardKinematics(self.model, self.data, q.astype(np.double).reshape(self.model.nq))
        self.robot.framesForwardKinematics(q)
        """ros的python环境在bash中会被引入，产生冲突
        unset PYTHONPATH
        unset LD_LIBRARY_PATH
        """
        # forwardGeometry(或 updateFramePlacements) 通常可以帮助更新 frame 的位姿
        # pin.updateFramePlacements(self.robot.model, self.robot.data)

        return q

    def get_link_quaternion(self, link_name=""):
        self._link_id = self.robot.model.getFrameId(link_name)
        _rot_world: np.ndarray = self.robot.data.oMf[self._link_id].rotation
        return R.from_matrix(_rot_world).as_quat(scalar_first=True)
