from pathlib import Path


COMMANDS_PATH = Path(
    "general_motion_tracker_whole_body_teleoperation/"
    "general_motion_tracker_whole_body_teleoperation/"
    "tasks/tracking_q1/mdp/commands.py"
)


def test_motion_command_installs_automatic_timing_wrappers():
    source = COMMANDS_PATH.read_text()

    assert "def _install_motion_command_timing_wrappers" in source
    assert "_install_motion_command_timing_wrappers(MotionCommand)" in source
    assert '_timing_metric_name("property",' in source
    assert '_timing_metric_name("func",' in source


def test_motion_command_has_step_level_timing_metrics():
    source = COMMANDS_PATH.read_text()

    expected_metric_fragments = [
        "time_step_resample_on_start_ms",
        "time_step_resample_build_probs_ms",
        "time_step_resample_multinomial_ms",
        "time_step_resample_sample_steps_ms",
        "time_step_resample_update_motion_ids_ms",
        "time_step_resample_on_complete_ms",
        "time_step_resample_update_metrics_ms",
        "time_step_motion_window_gather_ms",
        "time_step_motion_center_cache_ms",
        "time_step_motion_body_cache_ms",
        "time_step_motion_ref_cache_ms",
        "time_step_state_current_targets_ms",
        "time_step_state_robot_state_clone_ms",
        "time_step_state_relative_body_ms",
        "time_step_state_motion_ref_ms",
        "time_step_state_window_ms",
        "time_step_window_motion_ref_ms",
        "time_step_window_body_state_ms",
        "time_step_window_joint_delta_ms",
        "time_step_update_command_post_update_ms",
        "time_step_update_command_resample_ms",
        "time_step_update_command_motion_cache_ms",
        "time_step_update_command_state_cache_ms",
    ]

    for fragment in expected_metric_fragments:
        assert fragment in source
