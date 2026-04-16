from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


COMMANDS_PATH = Path(
    "general_motion_tracker_whole_body_teleoperation/"
    "general_motion_tracker_whole_body_teleoperation/"
    "tasks/tracking/mdp/commands.py"
)


def _load_commands_module():
    module_name = "test_mdp_commands_module"
    for name in [
        module_name,
        "isaaclab",
        "isaaclab.assets",
        "isaaclab.managers",
        "isaaclab.markers",
        "isaaclab.markers.config",
        "isaaclab.utils",
        "isaaclab.utils.math",
        "general_motion_tracker_whole_body_teleoperation",
        "general_motion_tracker_whole_body_teleoperation.tasks",
        "general_motion_tracker_whole_body_teleoperation.tasks.tracking",
        "general_motion_tracker_whole_body_teleoperation.tasks.tracking.mdp",
        "general_motion_tracker_whole_body_teleoperation.tasks.tracking.mdp.motion_loader",
    ]:
        sys.modules.pop(name, None)

    isaaclab = types.ModuleType("isaaclab")
    isaaclab_assets = types.ModuleType("isaaclab.assets")
    isaaclab_managers = types.ModuleType("isaaclab.managers")
    isaaclab_markers = types.ModuleType("isaaclab.markers")
    isaaclab_markers_config = types.ModuleType("isaaclab.markers.config")
    isaaclab_utils = types.ModuleType("isaaclab.utils")
    isaaclab_utils_math = types.ModuleType("isaaclab.utils.math")

    class CommandTerm:
        pass

    class CommandTermCfg:
        pass

    class VisualizationMarkers:
        def __init__(self, cfg):
            self.cfg = cfg

        def set_visibility(self, visible: bool):
            self.visible = visible

        def visualize(self, *args, **kwargs):
            del args, kwargs

    class VisualizationMarkersCfg:
        def __init__(self):
            self.markers = {"frame": SimpleNamespace(scale=None)}

        def replace(self, prim_path: str):
            del prim_path
            return self

    isaaclab_assets.Articulation = object
    isaaclab_managers.CommandTerm = CommandTerm
    isaaclab_managers.CommandTermCfg = CommandTermCfg
    isaaclab_markers.VisualizationMarkers = VisualizationMarkers
    isaaclab_markers.VisualizationMarkersCfg = VisualizationMarkersCfg
    isaaclab_markers_config.FRAME_MARKER_CFG = VisualizationMarkersCfg()
    isaaclab_utils.configclass = lambda cls: cls

    zero_quat = lambda *shape: torch.zeros(*shape, 4)
    isaaclab_utils_math.matrix_from_quat = lambda quat: torch.zeros(
        *quat.shape[:-1], 3, 3, dtype=quat.dtype, device=quat.device
    )
    isaaclab_utils_math.quat_apply = lambda quat, vec: vec
    isaaclab_utils_math.quat_apply_inverse = lambda quat, vec: vec
    isaaclab_utils_math.quat_error_magnitude = lambda q1, q2: torch.zeros(
        q1.shape[:-1], dtype=q1.dtype, device=q1.device
    )
    isaaclab_utils_math.quat_from_euler_xyz = lambda roll, pitch, yaw: zero_quat(
        *roll.shape
    )
    isaaclab_utils_math.quat_inv = lambda quat: quat
    isaaclab_utils_math.quat_mul = lambda q1, q2: q1
    isaaclab_utils_math.sample_uniform = (
        lambda low, high, shape, device=None: torch.zeros(shape, device=device)
    )
    isaaclab_utils_math.subtract_frame_transforms = (
        lambda p1, q1, p2, q2: (p2, q2)
    )
    isaaclab_utils_math.yaw_quat = lambda quat: quat

    sys.modules["isaaclab"] = isaaclab
    sys.modules["isaaclab.assets"] = isaaclab_assets
    sys.modules["isaaclab.managers"] = isaaclab_managers
    sys.modules["isaaclab.markers"] = isaaclab_markers
    sys.modules["isaaclab.markers.config"] = isaaclab_markers_config
    sys.modules["isaaclab.utils"] = isaaclab_utils
    sys.modules["isaaclab.utils.math"] = isaaclab_utils_math

    root_pkg = types.ModuleType("general_motion_tracker_whole_body_teleoperation")
    tasks_pkg = types.ModuleType("general_motion_tracker_whole_body_teleoperation.tasks")
    tracking_pkg = types.ModuleType(
        "general_motion_tracker_whole_body_teleoperation.tasks.tracking"
    )
    mdp_pkg = types.ModuleType(
        "general_motion_tracker_whole_body_teleoperation.tasks.tracking.mdp"
    )
    motion_loader_pkg = types.ModuleType(
        "general_motion_tracker_whole_body_teleoperation.tasks.tracking.mdp.motion_loader"
    )
    motion_loader_pkg.MotionLoader_human = object

    sys.modules[root_pkg.__name__] = root_pkg
    sys.modules[tasks_pkg.__name__] = tasks_pkg
    sys.modules[tracking_pkg.__name__] = tracking_pkg
    sys.modules[mdp_pkg.__name__] = mdp_pkg
    sys.modules[motion_loader_pkg.__name__] = motion_loader_pkg

    spec = importlib.util.spec_from_file_location(module_name, COMMANDS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_get_env_ids_to_resample_uses_valid_center_mask():
    module = _load_commands_module()
    command = object.__new__(module.MotionCommand)
    command.num_envs = 4
    command.device = "cpu"
    command.time_steps = torch.tensor([1, 2, 4, 6], dtype=torch.long)
    command.motion = SimpleNamespace(
        time_step_total=6,
        valid_center_mask=torch.tensor(
            [True, True, False, True, True, False], dtype=torch.bool
        ),
    )

    env_ids = module.MotionCommand._get_env_ids_to_resample(command)

    assert torch.equal(env_ids, torch.tensor([1, 3], dtype=torch.long))


def test_update_command_resamples_at_subtrajectory_boundary():
    module = _load_commands_module()
    command = object.__new__(module.MotionCommand)
    command.num_envs = 2
    command.device = "cpu"
    command.time_steps = torch.tensor([1, 3], dtype=torch.long)
    command.motion = SimpleNamespace(
        time_step_total=6,
        valid_center_mask=torch.tensor(
            [True, True, False, True, True, False], dtype=torch.bool
        ),
    )
    command.cfg = SimpleNamespace(adaptive_alpha=0.25)
    command.bin_failed_count = torch.zeros(3, dtype=torch.float32)
    command._current_bin_failed = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32)

    observed = {"resampled_env_ids": None, "call_order": []}

    def _resample_command(env_ids):
        observed["resampled_env_ids"] = env_ids.clone()
        observed["call_order"].append("resample")

    def _update_motion_cache():
        observed["call_order"].append("motion")

    def _update_robot_state_cache():
        observed["call_order"].append("robot")

    def _make_calculate():
        observed["call_order"].append("calculate")

    command._resample_command = _resample_command
    command._update_motion_cache = _update_motion_cache
    command._update_robot_state_cache = _update_robot_state_cache
    command._make_calculate = _make_calculate

    module.MotionCommand._update_command(command)

    assert torch.equal(
        observed["resampled_env_ids"], torch.tensor([0], dtype=torch.long)
    )
    assert torch.equal(command.time_steps, torch.tensor([2, 4], dtype=torch.long))
    assert observed["call_order"] == ["resample", "motion", "robot", "calculate"]
    assert torch.allclose(
        command.bin_failed_count, torch.tensor([0.0, 0.5, 0.0], dtype=torch.float32)
    )
    assert torch.equal(command._current_bin_failed, torch.zeros(3, dtype=torch.float32))
