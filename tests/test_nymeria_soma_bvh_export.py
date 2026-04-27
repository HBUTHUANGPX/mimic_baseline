import numpy as np

from nymeria_parse.motion_export.soma_bvh import DEFAULT_SOMA_BATCH_SIZE, export_nymeria_to_soma_bvh


def test_export_nymeria_to_soma_bvh_writes_root_zero_channels(monkeypatch, tmp_path):
    captured_kwargs = {}

    def fake_run_soma_inversion(motion, **kwargs):
        captured_kwargs.update(kwargs)
        local = np.zeros((motion.num_frames, 2, 7), dtype=np.float32)
        local[:, :, 3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        local[:, 0, :3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        local[:, 1, :3] = np.array([0.0, 101.0, 0.0], dtype=np.float32)
        return {
            "joint_names": ["Root", "Hips"],
            "parent_indices": np.array([-1, 0], dtype=np.int32),
            "reference_local_transforms": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 101.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            "local_transforms": local,
        }

    monkeypatch.setattr("nymeria_parse.motion_export.soma_bvh.run_soma_inversion", fake_run_soma_inversion)
    output_path = export_nymeria_to_soma_bvh(
        output_path=tmp_path / "motion.bvh",
        start_frame=0,
        end_frame=2,
        device="cuda",
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    motion_start = lines.index("MOTION")
    first_motion_values = [float(value) for value in lines[motion_start + 3].split()]
    assert first_motion_values[:6] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert first_motion_values[6:9] == [1.0, 103.0, 3.0]
    assert captured_kwargs["batch_size"] == DEFAULT_SOMA_BATCH_SIZE


def test_export_nymeria_to_soma_bvh_respects_explicit_batch_size(monkeypatch, tmp_path):
    captured_kwargs = {}

    def fake_run_soma_inversion(motion, **kwargs):
        captured_kwargs.update(kwargs)
        local = np.zeros((motion.num_frames, 2, 7), dtype=np.float32)
        local[:, :, 3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return {
            "joint_names": ["Root", "Hips"],
            "parent_indices": np.array([-1, 0], dtype=np.int32),
            "reference_local_transforms": local[0],
            "local_transforms": local,
        }

    monkeypatch.setattr("nymeria_parse.motion_export.soma_bvh.run_soma_inversion", fake_run_soma_inversion)
    export_nymeria_to_soma_bvh(
        output_path=tmp_path / "motion.bvh",
        start_frame=0,
        end_frame=2,
        device="cuda",
        batch_size=7,
    )

    assert captured_kwargs["batch_size"] == 7
