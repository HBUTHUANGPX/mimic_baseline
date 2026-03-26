from types import SimpleNamespace

from awesome_deploy.scripts import deploy_g1_mujoco as deploy_module


class _FakeBuffers:
    def __init__(self, time_step=0):
        self.values = {"time_step": time_step}

    def get(self, name, default=None):
        return self.values.get(name, default)

    def set(self, name, value):
        self.values[name] = value


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


def test_prepare_motion_play_step_advances_realtime_source_to_latest_frame():
    calls = []
    runner = deploy_module.simulator.__new__(deploy_module.simulator)
    runner.inference_engine = SimpleNamespace(buffers=_FakeBuffers(time_step=0))
    runner.motion = SimpleNamespace(
        is_realtime=True,
        time_step_total=5,
        advance=lambda: calls.append("advance"),
    )

    runner._prepare_motion_play_step()

    assert calls == ["advance"]
    assert runner.time_step == 4


def test_finalize_motion_play_step_advances_offline_frame_index():
    runner = deploy_module.simulator.__new__(deploy_module.simulator)
    runner.inference_engine = SimpleNamespace(buffers=_FakeBuffers(time_step=1))
    runner.motion = SimpleNamespace(
        is_realtime=False,
        time_step_total=3,
    )

    runner._finalize_motion_play_step()

    assert runner.time_step == 2
