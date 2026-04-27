from pathlib import Path

import numpy as np

from nymeria_parse.utils.mvnx import DEFAULT_SEQUENCE_DIR, load_mvnx_motion


def test_load_mvnx_motion_reads_normal_frames_with_metadata():
    motion = load_mvnx_motion(DEFAULT_SEQUENCE_DIR / "body_xdata_mvnx", end_frame=5)

    assert motion.fps == 240.0
    assert motion.segment_count == 23
    assert motion.frame_indices.tolist() == [0, 1, 2, 3, 4]
    assert motion.frame_timestamps.shape == (5,)
    assert motion.segment_quat_wxyz.shape == (5, 23, 4)
    assert motion.segment_pos_xyz.shape == (5, 23, 3)
    np.testing.assert_allclose(
        motion.segment_pos_xyz[0, 0],
        np.array([0.000004, -0.000012, 0.890444], dtype=np.float32),
        atol=1e-6,
    )


def test_load_mvnx_motion_uses_millisecond_timestamps():
    motion = load_mvnx_motion(DEFAULT_SEQUENCE_DIR / "body_xdata_mvnx", end_frame=2)

    assert motion.frame_timestamps.tolist() == [5674564, 5674608]
