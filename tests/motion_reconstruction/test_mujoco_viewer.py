from pathlib import Path

import numpy as np
import pytest

from motion_reconstruction.visualization.mujoco_viewer import _load_robot_kinematics, _robot_positions


pytest.importorskip("mujoco")


def test_robot_positions_places_model_root_from_anchor_body_pose(tmp_path: Path):
    xml_path = tmp_path / "robot.xml"
    xml_path.write_text(
        """
<mujoco model="anchor_test">
  <worldbody>
    <body name="base" pos="0 0 0">
      <freejoint/>
      <geom type="sphere" size="0.05"/>
      <body name="torso_link" pos="0 0 1">
        <geom type="sphere" size="0.05"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    robot = _load_robot_kinematics(xml_path, "torso_link")
    feature = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    anchor_pos_w = np.array([2.0, -1.0, 3.0], dtype=np.float32)

    positions, _ = _robot_positions(robot, feature, anchor_pos_w)

    assert np.allclose(positions[robot.anchor_body_id], anchor_pos_w, atol=1e-5)
    assert np.allclose(positions[1], np.array([2.0, -1.0, 2.0], dtype=np.float32), atol=1e-5)
