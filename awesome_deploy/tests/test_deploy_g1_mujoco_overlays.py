from types import SimpleNamespace

from awesome_deploy.scripts import deploy_g1_mujoco as deploy_module


def test_draw_viewer_overlays_draws_realtime_xsens_frames(monkeypatch):
    calls = {}

    def fake_draw_link_frames(**kwargs):
        calls["kwargs"] = kwargs

    monkeypatch.setattr(deploy_module, "draw_link_frames", fake_draw_link_frames)
    monkeypatch.setattr(deploy_module.cfg, "realtime_draw_xsens_frames", True)
    monkeypatch.setattr(deploy_module.cfg, "realtime_draw_xsens_labels", False)
    monkeypatch.setattr(deploy_module.cfg, "realtime_xsens_frame_axis_length", 0.12)
    monkeypatch.setattr(deploy_module.cfg, "realtime_xsens_frame_shaft_width", 0.01)

    runner = deploy_module.simulator.__new__(deploy_module.simulator)
    runner.viewer = SimpleNamespace(user_scn=SimpleNamespace(ngeom=7))
    runner.motion = SimpleNamespace(
        is_realtime=True,
        get_latest_xsens_human_frame=lambda: "human_frame",
    )
    runner.update_vel_geom = lambda: calls.setdefault("label_updated", True)

    runner._draw_viewer_overlays()

    assert runner.viewer.user_scn.ngeom == 0
    assert calls["kwargs"]["human_frame"] == "human_frame"
    assert calls["kwargs"]["axis_length"] == 0.12
    assert calls["kwargs"]["shaft_width"] == 0.01
    assert calls["kwargs"]["show_labels"] is False
    assert calls["kwargs"]["clear_existing"] is False
    assert calls["label_updated"] is True
