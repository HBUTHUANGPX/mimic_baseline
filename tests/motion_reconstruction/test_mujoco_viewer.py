from pathlib import Path

import numpy as np
import pytest

from motion_reconstruction.evaluation import ReconstructionResult
from motion_reconstruction.visualization import mujoco_viewer
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


def test_play_reconstruction_uses_robot_xml_as_viewer_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    xml_path = tmp_path / "robot.xml"
    xml_path.write_text(
        """
<mujoco model="viewer_robot">
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
    captured: dict[str, object] = {}

    class _FakeScene:
        def __init__(self) -> None:
            self.ngeom = 0

    class _FakeCamera:
        def __init__(self) -> None:
            self.azimuth = 0.0
            self.elevation = 0.0
            self.distance = 0.0
            self.lookat = np.zeros(3, dtype=np.float32)

    class _FakeViewer:
        def __init__(self) -> None:
            self.user_scn = _FakeScene()
            self.cam = _FakeCamera()
            self._first = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def is_running(self) -> bool:
            if self._first:
                self._first = False
                return True
            return False

        def sync(self) -> None:
            return None

    def _fake_launch_passive(model, data):
        captured["model"] = model
        captured["data"] = data
        return _FakeViewer()

    monkeypatch.setattr(
        mujoco_viewer,
        "_import_mujoco_viewer",
        lambda: type("_ViewerModule", (), {"launch_passive": staticmethod(_fake_launch_passive)}),
    )
    monkeypatch.setattr(mujoco_viewer.time, "sleep", lambda *_args, **_kwargs: None)

    result = ReconstructionResult(
        fps=30,
        center_indices=np.array([0], dtype=np.int64),
        original_robot_feature=None,
        recon_from_robot_feature=None,
        recon_from_human_feature=np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
        robot_anchor_pos_w=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        human_body_pos_w=np.array([[[0.0, 0.0, 0.9], [0.0, 0.0, 1.6]]], dtype=np.float32),
        robot_joint_names=[],
        robot_body_names=["base", "torso_link"],
        human_body_names=["Hips", "Head"],
        robot_anchor_body="torso_link",
        human_anchor_body="Hips",
    )

    mujoco_viewer.play_reconstruction(
        result=result,
        xml_path=xml_path,
        pair="human",
        loop=False,
    )

    model = captured["model"]
    assert int(model.nbody) == 3
