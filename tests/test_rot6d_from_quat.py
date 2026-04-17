from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MATH_PATH = Path(
    "IsaacLab_v230/source/isaaclab/isaaclab/utils/math.py"
)


def _load_math_module():
    module_name = "test_math_utils_module"
    spec = importlib.util.spec_from_file_location(module_name, MATH_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rot6d_from_quat_matches_matrix_first_two_columns():
    math_utils = _load_math_module()
    quaternions = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9238795, 0.0, 0.3826834, 0.0],
            [0.7071068, 0.0, 0.0, 0.7071068],
        ],
        dtype=torch.float32,
    )

    rot6d = math_utils.rot6d_from_quat(quaternions)
    expected = math_utils.matrix_from_quat(quaternions)[..., :2].reshape(
        quaternions.shape[0], -1
    )

    torch.testing.assert_close(rot6d, expected)
