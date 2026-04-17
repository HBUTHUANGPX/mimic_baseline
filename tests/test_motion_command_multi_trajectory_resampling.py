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
ADAPTIVE_SAMPLE_PATH = Path(
    "general_motion_tracker_whole_body_teleoperation/"
    "general_motion_tracker_whole_body_teleoperation/"
    "tasks/tracking/mdp/adaptive_sample.py"
)


def _install_stub_dependencies():
    for name in [
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
        "general_motion_tracker_whole_body_teleoperation.tasks.tracking.mdp.adaptive_sample",
        "general_motion_tracker_whole_body_teleoperation.utils",
        "general_motion_tracker_whole_body_teleoperation.utils.motion_loader",
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
    adaptive_sample_pkg = types.ModuleType(
        "general_motion_tracker_whole_body_teleoperation.tasks.tracking.mdp.adaptive_sample"
    )
    utils_pkg = types.ModuleType("general_motion_tracker_whole_body_teleoperation.utils")
    motion_loader_pkg = types.ModuleType(
        "general_motion_tracker_whole_body_teleoperation.utils.motion_loader"
    )

    class AdaptiveSamplingModule:
        def __init__(self, command, cfg=None):
            self.command = command
            self.cfg = cfg

        def on_resample_start(self, env_ids, update_failure_statistics):
            del env_ids, update_failure_statistics

        def build_sampling_probabilities(self):
            return torch.tensor([1.0])

        def on_resample_complete(
            self, env_ids, sampled_bins, update_failure_statistics
        ):
            del env_ids, sampled_bins, update_failure_statistics

        def on_step_end(self):
            return

    class LegacyBinAdaptiveSampling(AdaptiveSamplingModule):
        pass

    class SonicBinAdaptiveSampling(AdaptiveSamplingModule):
        pass

    class AdaptiveSamplingModuleCfg:
        def __init__(self, class_type=None):
            self.class_type = class_type

    class LegacyBinAdaptiveSamplingCfg(AdaptiveSamplingModuleCfg):
        def __init__(self):
            super().__init__(LegacyBinAdaptiveSampling)

    class SonicBinAdaptiveSamplingCfg(AdaptiveSamplingModuleCfg):
        def __init__(self):
            super().__init__(SonicBinAdaptiveSampling)

    adaptive_sample_pkg.AdaptiveSamplingModule = AdaptiveSamplingModule
    adaptive_sample_pkg.AdaptiveSamplingModuleCfg = AdaptiveSamplingModuleCfg
    adaptive_sample_pkg.LegacyBinAdaptiveSampling = LegacyBinAdaptiveSampling
    adaptive_sample_pkg.LegacyBinAdaptiveSamplingCfg = LegacyBinAdaptiveSamplingCfg
    adaptive_sample_pkg.SonicBinAdaptiveSampling = SonicBinAdaptiveSampling
    adaptive_sample_pkg.SonicBinAdaptiveSamplingCfg = SonicBinAdaptiveSamplingCfg
    motion_loader_pkg.MotionLoader_human = object

    sys.modules[root_pkg.__name__] = root_pkg
    sys.modules[tasks_pkg.__name__] = tasks_pkg
    sys.modules[tracking_pkg.__name__] = tracking_pkg
    sys.modules[mdp_pkg.__name__] = mdp_pkg
    sys.modules[adaptive_sample_pkg.__name__] = adaptive_sample_pkg
    sys.modules[utils_pkg.__name__] = utils_pkg
    sys.modules[motion_loader_pkg.__name__] = motion_loader_pkg


def _load_commands_module():
    module_name = "test_mdp_commands_module"
    _install_stub_dependencies()

    spec = importlib.util.spec_from_file_location(module_name, COMMANDS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_adaptive_sample_module():
    module_name = "test_mdp_adaptive_sample_module"
    _install_stub_dependencies()
    sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(module_name, ADAPTIVE_SAMPLE_PATH)
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
    observed = {"resampled_env_ids": None, "call_order": []}
    command.adaptive_sampler = SimpleNamespace(
        on_step_end=lambda: observed["call_order"].append("step_end")
    )

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
    assert observed["call_order"] == [
        "resample",
        "motion",
        "robot",
        "calculate",
        "step_end",
    ]


def test_legacy_sampler_attributes_failures_to_previous_time_step():
    module = _load_adaptive_sample_module()

    command = SimpleNamespace()
    command.device = "cpu"
    command.bin_count = 4
    command.bin_frame_count = 2
    command.time_steps = torch.tensor([2, 5], dtype=torch.long)
    command._previous_time_steps = torch.tensor([1, 4], dtype=torch.long)
    command._env = SimpleNamespace(
        termination_manager=SimpleNamespace(
            terminated=torch.tensor([True, False], dtype=torch.bool)
        )
    )
    command.cfg = SimpleNamespace(
        adaptive_sampler=module.LegacyBinAdaptiveSamplingCfg()
    )
    cfg = module.LegacyBinAdaptiveSamplingCfg()

    sampler = module.LegacyBinAdaptiveSampling(command, cfg)
    sampler.on_resample_start(torch.tensor([0, 1], dtype=torch.long), True)

    assert torch.equal(
        sampler.current_bin_failed,
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
    )
