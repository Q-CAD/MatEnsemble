from __future__ import annotations

import json

from mcp_matensemble.environment import get_local_matensemble_version
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


def test_v1_returns_live_example_and_container_file_bundles():
    examples = v1.get_examples("perlmutter")
    containers = v1.get_container_build_info("perlmutter")

    assert examples["ok"] is True
    assert {
        file["path"] for file in examples["result"]["files"]
    } >= {
        "example_workflows/generic/dependencies/workflow.py",
        "example_workflows/perlmutter/lammps_smoke/workflow.py",
        "example_workflows/perlmutter/lammps_mace/submit.slurm",
    }
    assert containers["ok"] is True
    assert {
        file["path"] for file in containers["result"]["files"]
    } == {
        "containers/perlmutter/Dockerfile.base",
        "containers/perlmutter/Dockerfile.base.lammps",
        "containers/perlmutter/Dockerfile.matensemble",
    }


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


def test_perlmutter_pull_plan_uses_local_version_without_registry_lookup(tmp_path):
    result = v1.prepare_container_pull_plan("perlmutter", cwd=tmp_path)
    local_version = get_local_matensemble_version()

    assert result["ok"] is True
    assert result["result"]["image"] == (
        f"ghcr.io/freddude2004/matensemble:perlmutter-v{local_version}"
    )
    assert result["result"]["registry_lookup_performed"] is False
    assert result["commands_not_run"] == [
        [
            "podman-hpc",
            "pull",
            f"ghcr.io/freddude2004/matensemble:perlmutter-v{local_version}",
        ]
    ]
    serialized = json.dumps(result)
    assert "tags/list" not in serialized
    assert "ghcr.io/token" not in serialized
    assert "curl" not in serialized


def test_pull_plan_accepts_explicit_version(tmp_path):
    result = v1.prepare_container_pull_plan(
        "perlmutter",
        version="0.3.12",
        cwd=tmp_path,
    )

    assert result["ok"] is True
    assert result["result"]["image"] == (
        "ghcr.io/freddude2004/matensemble:perlmutter-v0.3.12"
    )
    assert result["commands_not_run"][0] == [
        "podman-hpc",
        "pull",
        "ghcr.io/freddude2004/matensemble:perlmutter-v0.3.12",
    ]


def test_perlmutter_batch_defaults_to_two_nodes_and_matensemble_cli(tmp_path):
    created = v1.create_campaign("pm_batch", "perlmutter", cwd=tmp_path)
    campaign = created["result"]["campaign"]
    v1.write_workflow(campaign, "Run a smoke calculation.", cwd=tmp_path)

    result = v1.write_batch_script(
        campaign,
        "perlmutter",
        account="m5014",
        cwd=tmp_path,
    )
    script = (tmp_path / campaign / "submit.slurm").read_text(encoding="utf-8")

    assert result["ok"] is True
    assert result["result"]["resources"]["nodes"] == 2
    assert "#SBATCH -N 2" in script
    assert "ghcr.io/freddude2004/matensemble:perlmutter-v" in script
    assert "matensemble set-image ghcr.io/freddude2004/matensemble:perlmutter-v" in script
    assert "matensemble run workflow.py" in script
    assert "podman-hpc run" not in script
    assert "flux start python workflow.py" not in script
    assert 'cd "$(dirname' not in script


def test_hpc_nodes_bump_to_two_with_warning(tmp_path):
    created = v1.create_campaign("node_bump", "frontier", cwd=tmp_path)
    campaign = created["result"]["campaign"]
    v1.write_workflow(campaign, "Run a smoke calculation.", cwd=tmp_path)

    batch = v1.write_batch_script(
        campaign,
        "frontier",
        account="MAT269",
        nodes=1,
        cwd=tmp_path,
    )
    plan = v1.prepare_launch_plan(
        campaign,
        "frontier",
        account="MAT269",
        nodes=1,
        cwd=tmp_path,
    )
    script = (tmp_path / campaign / "submit.slurm").read_text(encoding="utf-8")

    assert batch["ok"] is True
    assert batch["result"]["resources"]["nodes"] == 2
    assert batch["warnings"]
    assert "broker" in batch["warnings"][0]
    assert "#SBATCH -N 2" in script
    assert "module load olcf-container-tools" in script
    assert "matensemble set-image ../containers/frontier/matensemble.sif" in script
    assert 'cd "$(dirname' not in script
    assert plan["ok"] is True
    assert plan["result"]["launch_plan"]["resources"]["nodes"] == 2
    assert plan["result"]["launch_plan"]["warnings"]


def test_linux_nodes_are_not_bumped(tmp_path):
    created = v1.create_campaign("linux_nodes", "linux", cwd=tmp_path)
    campaign = created["result"]["campaign"]
    v1.write_workflow(campaign, "Run a smoke calculation.", cwd=tmp_path)

    result = v1.write_batch_script(
        campaign,
        "linux",
        account="local",
        nodes=1,
        cwd=tmp_path,
    )
    script = (tmp_path / campaign / "submit.slurm").read_text(encoding="utf-8")

    assert result["ok"] is True
    assert result["result"]["resources"]["nodes"] == 1
    assert result["warnings"] == []
    assert "#SBATCH -N 1" in script
    assert 'cd "$(dirname' not in script


def test_pathfinder_batch_uses_fixed_repository_skeleton(tmp_path):
    created = v1.create_campaign("pathfinder_batch", "pathfinder", cwd=tmp_path)
    campaign = created["result"]["campaign"]
    v1.write_workflow(campaign, "Run a smoke calculation.", cwd=tmp_path)

    result = v1.write_batch_script(
        campaign,
        "pathfinder",
        account="MAT269",
        cwd=tmp_path,
    )
    script = (tmp_path / campaign / "submit.slurm").read_text(encoding="utf-8")

    assert result["ok"] is True
    assert 'cd "$(dirname' not in script
    assert "mkdir -p logs" in script
    assert "matensemble set-image ../containers/pathfinder/matensemble.sif" in script
    assert "matensemble run workflow.py" in script
