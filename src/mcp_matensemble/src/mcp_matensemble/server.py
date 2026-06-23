from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_matensemble import v1 as v1_tools
from mcp_matensemble.contracts import wrap
from mcp_matensemble.dashboard import (
    get_dashboard_status,
    plan_dashboard_access,
    start_dashboard,
    stop_dashboard,
)
from mcp_matensemble.environment import (
    get_version_info,
    plan_container_setup,
    run_container_setup,
)
from mcp_matensemble.examples import list_examples as legacy_list_examples
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
    get_job_status as scheduler_get_job_status,
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

    return json.dumps(legacy_list_examples(), indent=2)


@mcp.resource("matensemble://systems")
def matensemble_systems() -> str:
    """Return supported MatEnsemble runtime systems."""

    return json.dumps(systems_overview(), indent=2)


@mcp.resource("matensemble://systems/{system}/context")
def matensemble_system_context(system: str) -> str:
    """Return bundled system context for a supported system."""

    return json.dumps(wrap(v1_tools.get_system_context, system), indent=2)


@mcp.resource("matensemble://docs/{name}")
def matensemble_docs(name: str) -> str:
    """Return discovered MatEnsemble documentation by stem name."""

    root = v1_tools.repo_root()
    path = (root / "docs" / "source" / f"{name}.rst").resolve()
    docs_root = (root / "docs" / "source").resolve()
    try:
        path.relative_to(docs_root)
    except ValueError as exc:
        raise ValueError("documentation path must stay inside docs/source") from exc
    if not path.is_file():
        raise ValueError(f"documentation resource not found: {name}")
    return path.read_text(encoding="utf-8", errors="replace")


@mcp.resource("matensemble://examples/{system}/{name}")
def matensemble_example(system: str, name: str) -> str:
    """Return a repository example for a supported system."""

    return json.dumps(wrap(v1_tools.get_example, system, name), indent=2)


@mcp.resource("matensemble://containers/{system}/{filename}")
def matensemble_container(system: str, filename: str) -> str:
    """Return a repository container template for a supported system."""

    return json.dumps(wrap(v1_tools.get_container_template, system, filename), indent=2)


@mcp.resource("matensemble://source/{symbol}")
def matensemble_source(symbol: str) -> str:
    """Return source context for a MatEnsemble API symbol or module."""

    return json.dumps(wrap(v1_tools.explain_matensemble_api, symbol), indent=2)


@mcp.tool()
def list_supported_systems() -> dict[str, Any]:
    """List the strict V1 MatEnsemble MCP supported system enum."""

    return wrap(v1_tools.list_supported_systems)


@mcp.tool()
def get_system_context(system: str) -> dict[str, Any]:
    """Get bundled docs, examples, containers, and launch context for a system."""

    return wrap(v1_tools.get_system_context, system)


@mcp.tool()
def list_examples(system: str | None = None) -> dict[str, Any]:
    """List dynamically discovered examples, optionally filtered by system."""

    return wrap(v1_tools.list_examples, system)


@mcp.tool()
def get_example(system: str, name: str) -> dict[str, Any]:
    """Read a dynamically discovered example by system and name/id."""

    return wrap(v1_tools.get_example, system, name)


@mcp.tool()
def get_examples(system: str) -> dict[str, Any]:
    """Read generic workflows plus every example file for the selected system."""

    return wrap(v1_tools.get_examples, system)


@mcp.tool()
def list_container_templates(system: str | None = None) -> dict[str, Any]:
    """List dynamically discovered container templates."""

    return wrap(v1_tools.list_container_templates, system)


@mcp.tool()
def get_container_build_info(system: str) -> dict[str, Any]:
    """Read every file under containers/<system>/."""

    return wrap(v1_tools.get_container_build_info, system)


@mcp.tool()
def get_container_template(system: str, filename: str) -> dict[str, Any]:
    """Read a container template from containers/<system>/."""

    return wrap(v1_tools.get_container_template, system, filename)


@mcp.tool()
def explain_matensemble_api(symbol: str) -> dict[str, Any]:
    """Explain a MatEnsemble source symbol with snippets and related context."""

    return wrap(v1_tools.explain_matensemble_api, symbol)


@mcp.tool()
def create_campaign(
    campaign_name: str,
    system: str,
    auto_suffix: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a scratch-rooted campaign directory."""

    return wrap(
        v1_tools.create_campaign,
        campaign_name,
        system,
        auto_suffix=auto_suffix,
        overwrite=overwrite,
    )


@mcp.tool()
def list_campaigns() -> dict[str, Any]:
    """List campaigns under $SCRATCH/matensemble_campaigns."""

    return wrap(v1_tools.list_campaigns)


@mcp.tool()
def get_campaign_status(campaign: str) -> dict[str, Any]:
    """Inspect campaign files and MCP metadata."""

    return wrap(v1_tools.get_campaign_status, campaign)


@mcp.tool()
def write_workflow(
    campaign: str,
    science_goal: str,
    workflow_kind: str = "python_dag",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write workflow.py inside a campaign."""

    return wrap(
        v1_tools.write_workflow,
        campaign,
        science_goal,
        workflow_kind=workflow_kind,
        overwrite=overwrite,
    )


@mcp.tool()
def write_batch_script(
    campaign: str,
    system: str,
    account: str | None = None,
    nodes: int = 2,
    walltime: str = "00:30:00",
    queue: str = "debug",
    gpus: int = 4,
    tasks: int = 1,
    container_backend: str = "auto",
    container_path: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write submit.slurm inside a campaign. Account is required."""

    return wrap(
        v1_tools.write_batch_script,
        campaign,
        system,
        account=account,
        nodes=nodes,
        walltime=walltime,
        queue=queue,
        gpus=gpus,
        tasks=tasks,
        container_backend=container_backend,
        container_path=container_path,
        overwrite=overwrite,
    )


@mcp.tool()
def validate_campaign(campaign: str) -> dict[str, Any]:
    """Validate generated campaign workflow and launch scripts."""

    return wrap(v1_tools.validate_campaign, campaign)


@mcp.tool()
def prepare_launch_plan(
    campaign: str,
    system: str,
    mode: str = "batch",
    account: str | None = None,
    nodes: int = 2,
    walltime: str = "00:30:00",
    queue: str = "batch",
    gpus: int = 0,
    tasks: int = 1,
    container_backend: str = "auto",
    container_path: str | None = None,
) -> dict[str, Any]:
    """Prepare a launch plan without executing it."""

    return wrap(
        v1_tools.prepare_launch_plan,
        campaign,
        system,
        mode=mode,
        account=account,
        nodes=nodes,
        walltime=walltime,
        queue=queue,
        gpus=gpus,
        tasks=tasks,
        container_backend=container_backend,
        container_path=container_path,
    )


@mcp.tool()
def confirm_launch(launch_plan_id: str, timeout_seconds: int = 60) -> dict[str, Any]:
    """Execute a previously prepared launch plan by id."""

    return wrap(
        v1_tools.confirm_launch, launch_plan_id, timeout_seconds=timeout_seconds
    )


@mcp.tool()
def get_job_status(job_id: str, timeout_seconds: int = 30) -> dict[str, Any]:
    """Get Slurm job status through the V1 structured response contract."""

    return wrap(v1_tools.get_job_status, job_id, timeout_seconds=timeout_seconds)


@mcp.tool()
def prepare_container_pull_plan(
    system: str,
    force: bool = False,
    version: str | None = None,
    image_tag: str | None = None,
) -> dict[str, Any]:
    """Prepare a reusable GHCR container pull plan using the local version by default.

    This tool never queries GHCR tags. For "latest", it forms the image from
    the local MatEnsemble version: ghcr.io/freddude2004/matensemble:<system>-vX.Y.Z.
    """

    return wrap(
        v1_tools.prepare_container_pull_plan,
        system,
        force=force,
        version=version,
        image_tag=image_tag,
    )


@mcp.tool()
def confirm_container_pull(plan_id: str, timeout_seconds: int = 1800) -> dict[str, Any]:
    """Execute a prepared container pull plan by id."""

    return wrap(
        v1_tools.confirm_container_pull, plan_id, timeout_seconds=timeout_seconds
    )


@mcp.tool()
def write_container_file(
    campaign: str,
    system: str,
    apt_packages: list[str] | None = None,
    python_packages: list[str] | None = None,
    backend: str = "apptainer",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a custom container definition inside a campaign."""

    return wrap(
        v1_tools.write_container_file,
        campaign,
        system,
        apt_packages=apt_packages,
        python_packages=python_packages,
        backend=backend,
        overwrite=overwrite,
    )


@mcp.tool()
def create_container_build_plan(
    campaign: str,
    system: str,
    backend: str = "apptainer",
) -> dict[str, Any]:
    """Prepare a custom container build plan without executing it."""

    return wrap(
        v1_tools.prepare_container_build_plan, campaign, system, backend=backend
    )


@mcp.tool()
def prepare_container_build_plan(
    campaign: str,
    system: str,
    backend: str = "apptainer",
) -> dict[str, Any]:
    """Prepare a custom container build plan without executing it."""

    return wrap(
        v1_tools.prepare_container_build_plan, campaign, system, backend=backend
    )


@mcp.tool()
def confirm_container_build(
    plan_id: str, timeout_seconds: int = 3600
) -> dict[str, Any]:
    """Execute a prepared container build plan by id."""

    return wrap(
        v1_tools.confirm_container_build, plan_id, timeout_seconds=timeout_seconds
    )


@mcp.tool()
def prepare_cancel_job(job_id: str) -> dict[str, Any]:
    """Prepare a Slurm job cancellation plan without executing it."""

    return wrap(v1_tools.prepare_cancel_job, job_id)


@mcp.tool()
def confirm_cancel_job(plan_id: str, timeout_seconds: int = 30) -> dict[str, Any]:
    """Execute a prepared Slurm cancellation plan by id."""

    return wrap(v1_tools.confirm_cancel_job, plan_id, timeout_seconds=timeout_seconds)


@mcp.tool()
def delete_campaign_plan(campaign: str) -> dict[str, Any]:
    """Prepare a campaign deletion plan without deleting files."""

    return wrap(v1_tools.delete_campaign_plan, campaign)


@mcp.tool()
def confirm_delete_campaign(plan_id: str) -> dict[str, Any]:
    """Delete a campaign after plan-id confirmation."""

    return wrap(v1_tools.confirm_delete_campaign, plan_id)


@mcp.tool()
def create_matensemble_workflow_from_prompt(
    prompt: str,
    system: str,
    campaign_name: str,
    account: str | None = None,
    workflow_kind: str = "python_dag",
) -> dict[str, Any]:
    """Create and validate a campaign from a prompt without launching anything."""

    return wrap(
        v1_tools.create_matensemble_workflow_from_prompt,
        prompt,
        system,
        campaign_name,
        account=account,
        workflow_kind=workflow_kind,
    )


@mcp.tool()
def get_api_overview() -> str:
    """Get context agents should use before writing MatEnsemble scripts."""

    return api_overview()


@mcp.tool()
def list_matensemble_examples() -> list[dict[str, object]]:
    """List curated MatEnsemble examples and what each one demonstrates."""

    return legacy_list_examples()


@mcp.tool()
def get_matensemble_example(name: str) -> str:
    """Read a live in-repository example by name."""

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

    return scheduler_get_job_status(job_id, timeout_seconds=timeout_seconds)


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
def plan_matensemble_dashboard_access(
    dashboard_root: str,
    login_host: str | None = None,
    login_user: str | None = None,
    remote_port: int = 8000,
    local_port: int = 8000,
) -> dict[str, Any]:
    """Return login-node dashboard start and SSH local-forward commands."""

    return wrap(
        plan_dashboard_access,
        dashboard_root,
        login_host=login_host,
        login_user=login_user,
        remote_port=remote_port,
        local_port=local_port,
    )


@mcp.tool()
def start_matensemble_dashboard(
    dashboard_root: str,
    port: int = 8000,
    execute: bool = False,
) -> dict[str, Any]:
    """Start the dashboard on the MCP server host, dry-run by default."""

    return wrap(start_dashboard, dashboard_root, port=port, execute=execute)


@mcp.tool()
def get_matensemble_dashboard_status(
    dashboard_root: str,
    port: int = 8000,
) -> dict[str, Any]:
    """Read the local dashboard PID file and report whether it is running."""

    return wrap(get_dashboard_status, dashboard_root, port=port)


@mcp.tool()
def stop_matensemble_dashboard(
    dashboard_root: str,
    port: int = 8000,
) -> dict[str, Any]:
    """Stop a dashboard started by start_matensemble_dashboard."""

    return wrap(stop_dashboard, dashboard_root, port=port)


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
