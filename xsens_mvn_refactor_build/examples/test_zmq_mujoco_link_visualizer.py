import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from zmq_mujoco_link_visualizer import quaternion_xyzw_to_matrix


class QuaternionMatrixTest(unittest.TestCase):
    def test_identity_quaternion_returns_identity_matrix(self):
        matrix = quaternion_xyzw_to_matrix(0.0, 0.0, 0.0, 1.0)
        self.assertTrue(np.allclose(matrix, np.eye(3)))

    def test_half_turn_about_z_axis_rotates_x_axis_to_negative_x(self):
        matrix = quaternion_xyzw_to_matrix(0.0, 0.0, 1.0, 0.0)
        vector = matrix @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.assertTrue(np.allclose(vector, np.array([-1.0, 0.0, 0.0], dtype=np.float64)))

    def test_quaternion_is_normalized_before_conversion(self):
        scale = math.sqrt(2.0)
        matrix = quaternion_xyzw_to_matrix(0.0, 0.0, 0.0, scale)
        self.assertTrue(np.allclose(matrix, np.eye(3)))


if __name__ == "__main__":
    unittest.main()
