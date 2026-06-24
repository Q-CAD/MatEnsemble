from __future__ import annotations

from pathlib import Path

from mcp_matensemble import dashboard


class FakeProcess:
    pid = 4242

    def poll(self):
        return None


def test_launch_dashboard_starts_background_process(monkeypatch, tmp_path: Path):
    calls = {}
    campaign = tmp_path / "campaign"
    campaign.mkdir()

    def fake_popen(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(dashboard.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(dashboard.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(dashboard.socket, "gethostname", lambda: "login001")

    result = dashboard.launch_dashboard(str(campaign), port=8123)

    assert calls["command"] == [
        "matensemble",
        "dashboard",
        str(campaign),
        "--host",
        "127.0.0.1",
        "--port",
        "8123",
    ]
    assert calls["kwargs"]["cwd"] == str(campaign)
    assert result["running"] is True
    assert result["pid"] == 4242
    assert result["node"] == "login001"
    assert (campaign / "matensemble-dashboard-8123.pid").read_text() == "4242\n"


def test_dashboard_access_command():
    result = dashboard.get_dashboard_access(
        login_host="frontier.olcf.ornl.gov",
        login_user="alice",
        remote_port=8123,
        local_port=9000,
    )

    assert result["command"] == [
        "ssh",
        "-N",
        "-L",
        "9000:127.0.0.1:8123",
        "alice@frontier.olcf.ornl.gov",
    ]
    assert result["local_url"] == "http://localhost:9000"


def test_stop_dashboard_terminates_pid(monkeypatch, tmp_path: Path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    pid_path = campaign / "matensemble-dashboard-8000.pid"
    pid_path.write_text("4242\n", encoding="utf-8")
    killed = []

    def fake_kill(pid, sig):
        killed.append((pid, sig))

    monkeypatch.setattr(dashboard.os, "kill", fake_kill)

    result = dashboard.stop_dashboard(str(campaign))

    assert result["stopped"] is True
    assert killed == [(4242, 0), (4242, dashboard.signal.SIGTERM)]
    assert not pid_path.exists()
