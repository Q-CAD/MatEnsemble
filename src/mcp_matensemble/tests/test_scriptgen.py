from __future__ import annotations

import json

import pytest

from mcp_matensemble.scriptgen import (
    create_campaign,
    render_batch_script,
    render_interactive_script,
)


def test_create_campaign_writes_workflow_launch_and_manifest(tmp_path):
    result = create_campaign(
        "CO adsorption sweep",
        "Sweep adsorption geometries and analyze relaxed energies.",
        workflow_kind="python_dag",
        output_dir="campaigns",
        site="perlmutter",
        cwd=tmp_path,
    )

    assert result.workflow_path.exists()
    assert result.launch_path.exists()
    assert result.manifest_path.exists()
    assert result.interactive_path.exists()
    assert result.batch_path.exists()

    workflow = result.workflow_path.read_text(encoding="utf-8")
    assert "from matensemble.pipeline import Pipeline" in workflow
    assert "@pipe.chore" in workflow
    assert "pipe.submit" in workflow

    launch = result.launch_path.read_text(encoding="utf-8")
    assert "matensemble run workflow.py" in launch
    assert "Podman-HPC" in launch
    assert "This MCP server generated files only" in launch

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["campaign_name"] == "CO adsorption sweep"
    assert manifest["site"] == "perlmutter"
    assert manifest["files"]["batch"] == "run_batch.slurm"
    assert manifest["files"]["interactive"] == "run_interactive.sh"
    assert manifest["launches_executed_by_mcp"] is False

    batch = result.batch_path.read_text(encoding="utf-8")
    assert "#SBATCH" in batch
    assert "matensemble run workflow.py" in batch
    assert 'cd "$(dirname' not in batch


@pytest.mark.parametrize("system", ["frontier", "perlmutter", "pathfinder"])
def test_batch_scripts_follow_repository_examples_without_chdir(system):
    batch = render_batch_script(
        site=system,
        campaign_name="Canonical campaign",
        workflow_filename="workflow.py",
    )

    assert 'cd "$(dirname' not in batch
    assert "matensemble set-image " in batch
    assert "matensemble run workflow.py" in batch
    assert "if ! command -v matensemble" in batch


def test_frontier_batch_keeps_canonical_module_setup():
    batch = render_batch_script(
        site="frontier",
        campaign_name="Frontier campaign",
        workflow_filename="workflow.py",
    )

    assert "module reset" in batch
    assert "module load olcf-container-tools" in batch
    assert "module load apptainer-enable-mpi apptainer-enable-gpu" in batch
    assert "matensemble set-image ./matensemble.sif" in batch


@pytest.mark.parametrize("system", ["frontier", "perlmutter", "pathfinder"])
def test_interactive_scripts_follow_repository_examples(system):
    script = render_interactive_script(site=system, workflow_filename="workflow.py")

    assert 'cd "$(dirname' not in script
    assert "matensemble set-image " in script
    assert "matensemble run workflow.py" in script
    assert "if ! command -v matensemble" in script


def test_create_campaign_rejects_unknown_workflow_kind(tmp_path):
    with pytest.raises(ValueError, match="workflow_kind"):
        create_campaign(
            "bad",
            "bad",
            workflow_kind="not-a-kind",
            cwd=tmp_path,
        )


def test_create_campaign_rejects_output_dir_outside_workspace(tmp_path):
    with pytest.raises(ValueError, match="workspace"):
        create_campaign(
            "escape",
            "bad",
            output_dir="../outside",
            cwd=tmp_path,
        )


def test_create_campaign_requires_overwrite_for_existing_files(tmp_path):
    create_campaign("repeat", "first", cwd=tmp_path)

    with pytest.raises(FileExistsError):
        create_campaign("repeat", "second", cwd=tmp_path)

    result = create_campaign("repeat", "second", cwd=tmp_path, overwrite=True)
    assert "second" in result.workflow_path.read_text(encoding="utf-8")
