from pathlib import Path


COMMANDS_PATH = Path(
    "general_motion_tracker_whole_body_teleoperation/"
    "general_motion_tracker_whole_body_teleoperation/"
    "tasks/tracking/mdp/commands.py"
)


def test_sample_time_steps_from_bins_has_no_python_bin_loop():
    source = COMMANDS_PATH.read_text()
    marker = "    def _sample_time_steps_from_bins(self, sampled_bins: torch.Tensor) -> torch.Tensor:\n"
    start = source.index(marker)
    end = source.find("\n    def _resample_time_steps(", start)
    block = source[start:end]

    assert "torch.searchsorted(" in block
    assert "torch.unique(sampled_bins).tolist()" not in block
    assert "for bin_id in" not in block
