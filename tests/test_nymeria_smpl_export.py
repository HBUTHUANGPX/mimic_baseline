import numpy as np

from nymeria_parse.motion_export.smpl import (
    DEFAULT_SEQUENCE_DIR,
    build_smpl_motion_payload,
    save_smpl_motion_npz,
)


def test_build_smpl_motion_payload_contains_standard_smpl_fields_only():
    payload = build_smpl_motion_payload(DEFAULT_SEQUENCE_DIR, start_frame=0, end_frame=8)

    assert set(payload) == {"global_orient", "body_pose", "transl", "betas"}
    assert payload["global_orient"].shape == (8, 3)
    assert payload["body_pose"].shape == (8, 69)
    assert payload["transl"].shape == (8, 3)
    assert payload["betas"].shape == (8, 10)
    np.testing.assert_allclose(payload["betas"], 0.0)
    assert np.any(np.abs(payload["body_pose"]) > 1e-6)


def test_save_smpl_motion_npz_round_trips_standard_fields(tmp_path):
    payload = build_smpl_motion_payload(DEFAULT_SEQUENCE_DIR, start_frame=0, end_frame=3)
    output_path = save_smpl_motion_npz(payload, tmp_path / "motion.npz")

    loaded = np.load(output_path)
    assert set(loaded.files) == set(payload)
    assert loaded["body_pose"].shape == (3, 69)
