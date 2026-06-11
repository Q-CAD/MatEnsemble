from __future__ import annotations

import json

from mcp_matensemble.contracts import wrap
from mcp_matensemble import v1


def test_unsupported_system_returns_structured_error():
    result = wrap(v1.get_system_context, "summit")

    assert result["ok"] is False
    assert result["error_code"] == "UNSUPPORTED_SYSTEM"
    assert result["message"] == "Unsupported system: summit"
    assert result["details"]["supported_systems"] == [
        "frontier",
        "perlmutter",
        "pathfinder",
        "linux",
    ]


def test_campaign_batch_and_launch_plan_contract(tmp_path):
    created = v1.create_campaign("valid_name_123", "frontier", cwd=tmp_path)
    campaign = created["result"]["campaign"]

    workflow = v1.write_workflow(campaign, "Run a smoke calculation.", cwd=tmp_path)
    missing_account = v1.write_batch_script(
        campaign,
        "frontier",
        account=None,
        cwd=tmp_path,
    )
    batch = v1.write_batch_script(
        campaign,
        "frontier",
        account="MAT269",
        cwd=tmp_path,
    )
    plan = v1.prepare_launch_plan(
        campaign,
        "frontier",
        account="MAT269",
        cwd=tmp_path,
    )

    assert workflow["ok"] is True
    assert workflow["result"]["path"].endswith("/workflow.py")
    assert missing_account["ok"] is False
    assert missing_account["error_code"] == "ACCOUNT_REQUIRED"
    assert batch["ok"] is True
    assert batch["result"]["path"].endswith("/submit.slurm")
    assert plan["ok"] is True
    assert plan["commands_not_run"] == [["sbatch", "submit.slurm"]]
    assert plan["result"]["launch_plan"]["requires_confirmation"] is True


def test_launch_confirm_rejects_modified_plan(tmp_path):
    created = v1.create_campaign("launch_hash", "frontier", cwd=tmp_path)
    campaign = created["result"]["campaign"]
    v1.write_workflow(campaign, "Run a smoke calculation.", cwd=tmp_path)
    v1.write_batch_script(campaign, "frontier", account="MAT269", cwd=tmp_path)
    plan_result = v1.prepare_launch_plan(campaign, "frontier", account="MAT269", cwd=tmp_path)
    plan = plan_result["result"]["launch_plan"]

    path = tmp_path / campaign / "launch_plan.json"
    modified = json.loads(path.read_text(encoding="utf-8"))
    modified["command"] = ["sbatch", "other.slurm"]
    path.write_text(json.dumps(modified, indent=2) + "\n", encoding="utf-8")

    confirmed = v1.confirm_launch(plan["launch_plan_id"], cwd=tmp_path)

    assert confirmed["ok"] is False
    assert confirmed["error_code"] == "PLAN_MODIFIED"


def test_path_safety_rejects_campaign_traversal(tmp_path):
    result = wrap(v1.create_campaign, "../evil", "frontier", cwd=tmp_path)

    assert result["ok"] is False
    assert result["error_code"] == "VALIDATION_ERROR"
