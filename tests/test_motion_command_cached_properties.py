from pathlib import Path


COMMANDS_PATH = Path(
    "general_motion_tracker_whole_body_teleoperation/"
    "general_motion_tracker_whole_body_teleoperation/"
    "tasks/tracking/mdp/commands.py"
)


def _get_property_block(source: str, property_name: str) -> str:
    marker = f"    def {property_name}(self) -> torch.Tensor:\n"
    start = source.index(marker)
    next_property = source.find("\n    @property\n", start + len(marker))
    if next_property == -1:
        next_property = len(source)
    return source[start:next_property]


def test_properties_return_precomputed_caches():
    source = COMMANDS_PATH.read_text()

    expected_returns = {
        "motion_id": "return self._motion_id",
        "motion_group": "return self._motion_group",
        "command": "return self._command",
        "joint_pos": "return self._joint_pos",
        "joint_vel": "return self._joint_vel",
        "ref_lin_vel_w": "return self._ref_lin_vel_w",
        "ref_ang_vel_w": "return self._ref_ang_vel_w",
    }

    forbidden_fragments = ("[self.time_steps]", "[:, self.center_frame_index]", "torch.cat(")

    for property_name, return_line in expected_returns.items():
        block = _get_property_block(source, property_name)
        assert return_line in block
        for fragment in forbidden_fragments:
            assert fragment not in block
