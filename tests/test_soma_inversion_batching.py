import numpy as np

from hdf5_parse.motion_export.smpl_soma import (
    SMPLBodyMotion,
    configure_warp_quiet,
    iter_batch_slices,
    project_rotation_matrices,
)


def test_iter_batch_slices_uses_full_motion_when_batch_size_is_none():
    motion = SMPLBodyMotion(
        global_orient=np.zeros((5, 3), dtype=np.float32),
        body_pose=np.zeros((5, 69), dtype=np.float32),
        transl=np.zeros((5, 3), dtype=np.float32),
        betas=np.zeros((5, 10), dtype=np.float32),
        frame_nums=np.arange(5, dtype=np.int32),
        frame_timestamps=np.arange(5, dtype=np.int64),
        fps=240.0,
    )

    assert list(iter_batch_slices(motion, batch_size=None)) == [(0, 5)]


def test_iter_batch_slices_splits_motion_into_safe_chunks():
    motion = SMPLBodyMotion(
        global_orient=np.zeros((5, 3), dtype=np.float32),
        body_pose=np.zeros((5, 69), dtype=np.float32),
        transl=np.zeros((5, 3), dtype=np.float32),
        betas=np.zeros((5, 10), dtype=np.float32),
        frame_nums=np.arange(5, dtype=np.int32),
        frame_timestamps=np.arange(5, dtype=np.int64),
        fps=240.0,
    )

    assert list(iter_batch_slices(motion, batch_size=2)) == [(0, 2), (2, 4), (4, 5)]


def test_configure_warp_quiet_suppresses_module_load_logs():
    import warp as wp

    original = wp.config.quiet
    try:
        wp.config.quiet = False
        configure_warp_quiet(True)
        assert wp.config.quiet is True
    finally:
        wp.config.quiet = original


def test_project_rotation_matrices_repairs_non_positive_determinant_matrix():
    bad_matrix = np.array(
        [
            [0.99842989, -0.00479428, 0.02178393],
            [-0.00479425, 0.9644292, 0.17610992],
            [0.02178393, 0.17611, 0.03267289],
        ],
        dtype=np.float64,
    )

    projected = project_rotation_matrices(bad_matrix)

    assert projected.shape == (3, 3)
    assert np.linalg.det(projected) > 0.0
    np.testing.assert_allclose(projected.T @ projected, np.eye(3), atol=1e-6)
