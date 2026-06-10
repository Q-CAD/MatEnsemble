from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_matensemble.environment import (
    get_version_info,
    plan_container_setup,
    run_container_setup,
)
from mcp_matensemble.examples import list_examples
from mcp_matensemble.resources import api_overview, read_repo_example
from mcp_matensemble.resources import (
    container_build_instructions,
    environment_setup,
    read_container_contents,
    system_profile,
    systems_overview,
)
from mcp_matensemble.scheduler import (
    cancel_job,
    get_job_status,
    inspect_outputs,
    plan_job_submission,
    submit_job,
    tail_log,
)
from mcp_matensemble.scriptgen import (
    VALID_WORKFLOW_KINDS,
    create_batch_script,
    create_campaign,
)

mcp = FastMCP("mcp-matensemble")


@mcp.resource("matensemble://api/overview")
def matensemble_api_overview() -> str:
    """Return concise guidance on the current MatEnsemble Pipeline API."""

    return api_overview()


@mcp.resource("matensemble://examples")
def matensemble_examples() -> str:
    """Return known in-repository MatEnsemble examples."""

    return json.dumps(list_examples(), indent=2)


@mcp.resource("matensemble://systems")
def matensemble_systems() -> str:
    """Return supported MatEnsemble runtime systems."""

    return json.dumps(systems_overview(), indent=2)


@mcp.tool()
def get_api_overview() -> str:
    """Get context agents should use before writing MatEnsemble scripts."""

    return api_overview()


@mcp.tool()
def list_matensemble_examples() -> list[dict[str, object]]:
    """List curated MatEnsemble examples and what each one demonstrates."""

    return list_examples()


@mcp.tool()
def get_matensemble_example(name: str) -> str:
    """Read a curated in-repository example by name."""

    return read_repo_example(name)


@mcp.tool()
def list_matensemble_systems() -> list[dict[str, object]]:
    """List supported systems and runtime environment options."""

    return systems_overview()


@mcp.tool()
def get_matensemble_system(name: str) -> dict[str, object]:
    """Get structured runtime and launch guidance for a system."""

    return system_profile(name)


@mcp.tool()
def get_matensemble_environment_setup(name: str) -> str:
    """Get install, container, CLI, allocation, and launch guidance for a system."""

    return environment_setup(name)


@mcp.tool()
def get_matensemble_container_contents(name: str) -> str:
    """Get Dockerfile-derived container contents context for a supported system."""

    return read_container_contents(name)


@mcp.tool()
def get_matensemble_container_install(name: str) -> str:
    """Get setup instructions for obtaining and configuring a MatEnsemble container."""

    return container_build_instructions(name)


@mcp.tool()
def get_matensemble_version_info(check_pypi: bool = False) -> dict[str, Any]:
    """Get local MatEnsemble version info and optionally query PyPI for the latest release."""

    return get_version_info(check_pypi=check_pypi)


@mcp.tool()
def plan_matensemble_container_setup(
    system: str,
    version: str | None = None,
    image_tag: str | None = None,
    output_path: str | None = None,
    runtime: str | None = None,
) -> dict[str, Any]:
    """Plan container/environment setup commands without executing them."""

    return plan_container_setup(
        system,
        version=version,
        image_tag=image_tag,
        output_path=output_path,
        runtime=runtime,
    )


@mcp.tool()
def run_matensemble_container_setup(
    system: str,
    action: str = "auto",
    version: str | None = None,
    image_tag: str | None = None,
    output_path: str | None = None,
    runtime: str | None = None,
    execute: bool = False,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Run one allowlisted setup command, dry-run by default."""

    return run_container_setup(
        system,
        action=action,
        version=version,
        image_tag=image_tag,
        output_path=output_path,
        runtime=runtime,
        execute=execute,
        timeout_seconds=timeout_seconds,
    )


@mcp.tool()
def create_matensemble_batch_script(
    system: str,
    campaign_dir: str,
    template: str = "smoke",
    overwrite: bool = False,
) -> dict[str, str]:
    """Create a site-specific Slurm batch script in an existing campaign directory."""

    path = create_batch_script(
        system=system,
        campaign_dir=campaign_dir,
        template=template,
        overwrite=overwrite,
    )
    return {"batch_path": str(path)}


@mcp.tool()
def plan_matensemble_job_submission(script_path: str) -> dict[str, Any]:
    """Validate and plan an sbatch submission without executing it."""

    return plan_job_submission(script_path)


@mcp.tool()
def submit_matensemble_job(
    script_path: str,
    execute: bool = False,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Submit a validated .slurm script with sbatch, dry-run by default."""

    return submit_job(
        script_path,
        execute=execute,
        timeout_seconds=timeout_seconds,
    )


@mcp.tool()
def get_matensemble_job_status(
    job_id: str, timeout_seconds: int = 30
) -> dict[str, Any]:
    """Get Slurm job status through squeue."""

    return get_job_status(job_id, timeout_seconds=timeout_seconds)


@mcp.tool()
def cancel_matensemble_job(
    job_id: str,
    execute: bool = False,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Cancel a Slurm job with scancel, dry-run by default."""

    return cancel_job(
        job_id,
        execute=execute,
        timeout_seconds=timeout_seconds,
    )


@mcp.tool()
def inspect_matensemble_outputs(workflow_dir: str) -> dict[str, Any]:
    """Inspect MatEnsemble status, log path, and per-chore output artifacts."""

    return inspect_outputs(workflow_dir)


@mcp.tool()
def tail_matensemble_log(workflow_dir: str, lines: int = 100) -> dict[str, Any]:
    """Tail the MatEnsemble workflow log from a generated workflow directory."""

    return tail_log(workflow_dir, lines=lines)


@mcp.tool()
def create_matensemble_campaign(
    campaign_name: str,
    science_goal: str,
    workflow_kind: str = "python_dag",
    output_dir: str | None = "matensemble_campaigns",
    site: str = "generic_flux",
    matensemble_version: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate workflow.py, LAUNCH.md, and manifest.json for a campaign."""

    result = create_campaign(
        campaign_name=campaign_name,
        science_goal=science_goal,
        workflow_kind=workflow_kind,
        output_dir=output_dir,
        site=site,
        matensemble_version=matensemble_version,
        overwrite=overwrite,
    )
    payload = result.to_dict()
    payload["valid_workflow_kinds"] = list(VALID_WORKFLOW_KINDS)
    payload["message"] = (
        "Generated files only. Review workflow.py and LAUNCH.md before running "
        "inside a Flux-capable environment."
    )
    return payload


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
