"""Pinocchio helpers that mirror MuJoCo robot state into kinematic queries."""

import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import zero
from scipy.spatial.transform import Rotation as R


class pin_mj:
    """Convenience wrapper around a Pinocchio robot built from the deploy URDF."""

    def __init__(self, _cfg):
        """Builds a Pinocchio robot using the active deployment config.

        Args:
            _cfg: Robot configuration providing URDF and asset paths.
        """
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
        """Copies MuJoCo state into Pinocchio generalized coordinates.

        Args:
            joint_angles: Joint angle vector in URDF joint order.
            base_pos: World-frame base position ``[x, y, z]``.
            base_quat: World-frame base quaternion in ``[x, y, z, w]`` order.

        Returns:
            Generalized coordinate vector passed to Pinocchio.
        """
        q: np.ndarray = zero(self.robot.model.nq)
        if self.robot.model.joints[1].shortname() == "JointModelFreeFlyer":
            q[0:3] = base_pos
            q[3:7] = base_quat
            q[7:] = joint_angles
        else:
            q[:] = joint_angles
        # Update frame placements immediately so callers can query link poses
        # without performing an extra forward pass.
        self.robot.framesForwardKinematics(q)
        return q

    def get_link_quaternion(self, link_name=""):
        """Returns the world-frame quaternion of a named link.

        Args:
            link_name: Pinocchio frame name to query.

        Returns:
            Quaternion in scalar-first ``[w, x, y, z]`` order.
        """
        self._link_id = self.robot.model.getFrameId(link_name)
        _rot_world: np.ndarray = self.robot.data.oMf[self._link_id].rotation
        return R.from_matrix(_rot_world).as_quat(scalar_first=True)
