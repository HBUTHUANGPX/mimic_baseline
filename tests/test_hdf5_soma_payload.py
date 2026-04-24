from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    module = importlib.import_module("hdf5_parse.motion_export.core")
    return importlib.reload(module)


def test_drop_soma_virtual_root_reparents_hips() -> None:
    module = load_module()
    joint_names = ["Root", "Hips", "Spine1", "LeftHand"]
    parent_indices = np.array([0, 0, 1, 2], dtype=np.int32)
    reference_local_transforms = np.array(
        [
            [9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    local_transforms = np.arange(2 * 4 * 7, dtype=np.float32).reshape(2, 4, 7)

    dropped_names, dropped_parents, dropped_reference, dropped_local = module.drop_soma_virtual_root(
        joint_names=joint_names,
        parent_indices=parent_indices,
        reference_local_transforms=reference_local_transforms,
        local_transforms=local_transforms,
    )

    assert dropped_names == ["Hips", "Spine1", "LeftHand"]
    np.testing.assert_array_equal(dropped_parents, np.array([-1, 0, 1], dtype=np.int32))
    np.testing.assert_allclose(dropped_reference, reference_local_transforms[1:])
    np.testing.assert_allclose(dropped_local, local_transforms[:, 1:])


def test_mask_joint_data_zeroes_only_non_selected_joints() -> None:
    module = load_module()
    joint_names = ["Hips", "Spine1", "Spine2", "LeftHand"]
    local_transforms = np.ones((2, 4, 7), dtype=np.float32)
    global_pos = np.full((2, 4, 3), 3.0, dtype=np.float32)
    global_quat = np.full((2, 4, 4), 4.0, dtype=np.float32)

    masked_local, masked_global_pos, masked_global_quat = module.mask_joint_data(
        joint_names=joint_names,
        human_local_transforms=local_transforms,
        human_global_pos=global_pos,
        human_global_quat=global_quat,
        selected_joint_names={"Hips", "Spine1"},
    )

    np.testing.assert_allclose(masked_local[:, 0], 1.0)
    np.testing.assert_allclose(masked_local[:, 1], 1.0)
    np.testing.assert_allclose(masked_local[:, 2:], 0.0)
    np.testing.assert_allclose(masked_global_pos[:, 2:], 0.0)
    np.testing.assert_allclose(masked_global_quat[:, 2:], 0.0)


def test_build_human_export_payload_preserves_human_fields_and_extras() -> None:
    module = load_module()
    joint_names = ["Hips", "Spine1"]
    parent_indices = np.array([-1, 0], dtype=np.int32)
    reference_local_transforms = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    human_local_transforms = np.array(
        [
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            [
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
        ],
        dtype=np.float32,
    )
    extras = {
        "timeline_frame_indices": np.array([10, 20], dtype=np.int32),
        "main_task_texts": np.array(["UNKNOWN", "Task"], dtype=object),
        "main_task_text_indices": np.array([1, 1], dtype=np.int32),
        "smpl_transl": np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32),
    }

    payload = module.build_human_export_payload(
        fps=30,
        joint_names=joint_names,
        parent_indices=parent_indices,
        reference_local_transforms=reference_local_transforms,
        human_local_transforms=human_local_transforms,
        extra_payload=extras,
    )

    assert payload["fps"].item() == 30
    assert payload["num_frames"].item() == 2
    assert payload["scalar_first"].item() is False
    assert payload["human_joint_names"].tolist() == joint_names
    np.testing.assert_array_equal(payload["human_parent_indices"], parent_indices)
    np.testing.assert_allclose(payload["human_reference_local_transforms"], reference_local_transforms)
    np.testing.assert_allclose(
        payload["human_global_pos"][0],
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        payload["human_global_pos"][1],
        np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 1.0]], dtype=np.float32),
        atol=1e-6,
    )
    assert "robot_name" not in payload
    assert "robot_joint_names" not in payload
    np.testing.assert_array_equal(payload["timeline_frame_indices"], np.array([10, 20], dtype=np.int32))


def test_build_human_export_payload_matches_reference_player_semantics() -> None:
    module = load_module()
    joint_names = ["Hips", "Head"]
    parent_indices = np.array([-1, 0], dtype=np.int32)
    root_inv_vis = np.array([-np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32)
    reference_local_transforms = np.array(
        [
            [0.0, 0.0, 0.0, *root_inv_vis],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    human_local_transforms = np.array(
        [
            [
                [0.0, 0.0, 0.0, *root_inv_vis],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=np.float32,
    )

    payload = module.build_human_export_payload(
        fps=20,
        joint_names=joint_names,
        parent_indices=parent_indices,
        reference_local_transforms=reference_local_transforms,
        human_local_transforms=human_local_transforms,
    )

    hips_to_head = payload["human_global_pos"][0, 1] - payload["human_global_pos"][0, 0]
    np.testing.assert_allclose(hips_to_head, np.array([0.0, 0.0, 1.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_array_equal(payload["human_up_axis"], np.array([0.0, 0.0, 1.0], dtype=np.float32))
