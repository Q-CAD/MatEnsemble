from __future__ import annotations

from mcp_matensemble.dashboard import plan_dashboard_access, start_dashboard


def test_plan_dashboard_access_returns_login_node_tunnel(tmp_path):
    campaign = tmp_path / "campaign"
    workflow = campaign / "matensemble_workflow-20260101_000000"
    workflow.mkdir(parents=True)
    (workflow / "status.json").write_text('{"state": "running"}', encoding="utf-8")

    plan = plan_dashboard_access(
        str(workflow),
        login_host="frontier.olcf.ornl.gov",
        login_user="alice",
        cwd=tmp_path,
    )

    assert plan["dashboard_root"] == str(campaign)
    assert plan["url"] == "http://localhost:8000"
    assert plan["forward_command"] == [
        "ssh",
        "-N",
        "-L",
        "8000:127.0.0.1:8000",
        "alice@frontier.olcf.ornl.gov",
    ]
    assert "matensemble dashboard" in plan["background_start_command"]
    assert str(campaign) in plan["background_start_command"]


def test_start_dashboard_defaults_to_dry_run(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()

    result = start_dashboard(str(campaign), cwd=tmp_path)

    assert result["executed"] is False
    assert result["command"][-5:] == [
        str(campaign),
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    assert result["pid_path"] == str(campaign / "matensemble-dashboard-8000.pid")
