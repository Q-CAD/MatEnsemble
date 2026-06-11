from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

from .environment import resolve_image_tag
from .security import resolve_campaign_dir, slugify
from .systems import get_system_profile


VALID_WORKFLOW_KINDS = ("python_dag", "executable", "mpi", "strategy")


@dataclass(frozen=True)
class CampaignResult:
    campaign_dir: Path
    workflow_path: Path
    launch_path: Path
    manifest_path: Path
    interactive_path: Path
    batch_path: Path
    workflow_kind: str

    def to_dict(self) -> dict[str, str]:
        return {
            "campaign_dir": str(self.campaign_dir),
            "workflow_path": str(self.workflow_path),
            "launch_path": str(self.launch_path),
            "manifest_path": str(self.manifest_path),
            "interactive_path": str(self.interactive_path),
            "batch_path": str(self.batch_path),
            "workflow_kind": self.workflow_kind,
        }


def create_campaign(
    campaign_name: str,
    science_goal: str,
    *,
    workflow_kind: str = "python_dag",
    output_dir: str | None = "matensemble_campaigns",
    site: str = "generic_flux",
    matensemble_version: str | None = None,
    overwrite: bool = False,
    cwd: Path | None = None,
) -> CampaignResult:
    """Create a MatEnsemble campaign scaffold with launch instructions."""

    if workflow_kind not in VALID_WORKFLOW_KINDS:
        raise ValueError(
            f"workflow_kind must be one of {', '.join(VALID_WORKFLOW_KINDS)}"
        )

    campaign_dir = resolve_campaign_dir(output_dir, campaign_name, cwd=cwd)
    workflow_path = campaign_dir / "workflow.py"
    launch_path = campaign_dir / "LAUNCH.md"
    manifest_path = campaign_dir / "manifest.json"
    interactive_path = campaign_dir / "run_interactive.sh"
    batch_path = campaign_dir / "run_batch.slurm"

    expected = (workflow_path, launch_path, manifest_path, interactive_path, batch_path)
    existing = [p for p in expected if p.exists()]
    if existing and not overwrite:
        paths = ", ".join(str(p) for p in existing)
        raise FileExistsError(f"campaign files already exist: {paths}")

    campaign_dir.mkdir(parents=True, exist_ok=True)
    workflow_text = render_workflow_script(
        campaign_name=campaign_name,
        science_goal=science_goal,
        workflow_kind=workflow_kind,
    )
    launch_text = render_launch_guide(
        campaign_name=campaign_name,
        science_goal=science_goal,
        workflow_kind=workflow_kind,
        site=site,
        matensemble_version=matensemble_version,
        workflow_filename=workflow_path.name,
    )
    manifest = {
        "campaign_name": campaign_name,
        "campaign_slug": slugify(campaign_name),
        "science_goal": science_goal,
        "workflow_kind": workflow_kind,
        "site": site,
        "matensemble_version": matensemble_version,
        "files": {
            "workflow": workflow_path.name,
            "launch": launch_path.name,
            "interactive": interactive_path.name,
            "batch": batch_path.name,
        },
        "launches_executed_by_mcp": False,
    }

    workflow_path.write_text(workflow_text, encoding="utf-8")
    launch_path.write_text(launch_text, encoding="utf-8")
    interactive_path.write_text(
        render_interactive_script(site=site, workflow_filename=workflow_path.name),
        encoding="utf-8",
    )
    batch_path.write_text(
        render_batch_script(
            site=site,
            campaign_name=campaign_name,
            workflow_filename=workflow_path.name,
            matensemble_version=matensemble_version,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return CampaignResult(
        campaign_dir=campaign_dir,
        workflow_path=workflow_path,
        launch_path=launch_path,
        manifest_path=manifest_path,
        interactive_path=interactive_path,
        batch_path=batch_path,
        workflow_kind=workflow_kind,
    )


def create_batch_script(
    *,
    system: str,
    campaign_dir: str,
    template: str = "smoke",
    overwrite: bool = False,
    cwd: Path | None = None,
) -> Path:
    from .security import resolve_within_workspace

    if template != "smoke":
        raise ValueError("only template='smoke' is supported")
    target_dir = resolve_within_workspace(campaign_dir, cwd=cwd)
    if not target_dir.is_dir():
        raise ValueError(f"campaign_dir must exist: {target_dir}")
    workflow_path = target_dir / "workflow.py"
    if not workflow_path.exists():
        raise ValueError(f"campaign_dir must contain workflow.py: {target_dir}")
    batch_path = target_dir / "run_batch.slurm"
    if batch_path.exists() and not overwrite:
        raise FileExistsError(f"batch script already exists: {batch_path}")
    batch_path.write_text(
        render_batch_script(
            site=system,
            campaign_name=target_dir.name,
            workflow_filename=workflow_path.name,
            matensemble_version=None,
        ),
        encoding="utf-8",
    )
    return batch_path


def render_workflow_script(
    *,
    campaign_name: str,
    science_goal: str,
    workflow_kind: str,
) -> str:
    header = f'''\
"""
MatEnsemble campaign: {campaign_name}

Science goal:
{science_goal}

Generated by the MatEnsemble MCP server MVP. Review the chore bodies,
resource requests, and input paths before launching on an allocation.
"""
'''

    if workflow_kind == "python_dag":
        body = _python_dag_template()
    elif workflow_kind == "executable":
        body = _executable_template()
    elif workflow_kind == "mpi":
        body = _mpi_template()
    elif workflow_kind == "strategy":
        body = _strategy_template()
    else:
        raise ValueError(f"unsupported workflow_kind: {workflow_kind}")

    return header + "\n" + body


def render_launch_guide(
    *,
    campaign_name: str,
    science_goal: str,
    workflow_kind: str,
    site: str,
    matensemble_version: str | None,
    workflow_filename: str,
) -> str:
    profile = get_system_profile(site)
    image = resolve_image_tag(profile.name, version=matensemble_version)
    cli_install = profile.cli_install.replace(profile.recommended_image, image)
    return dedent(
        f"""\
        # Launch Guide: {campaign_name}

        ## Science Goal

        {science_goal}

        ## Generated Workflow

        - Workflow kind: `{workflow_kind}`
        - Target site: `{profile.title}`
        - Script: `{workflow_filename}`
        - Recommended image: `{image}`
        - Runtime: `{profile.container_runtime}`

        ## Before Launch

        1. Review `workflow.py` and replace placeholder chore bodies, input paths, and executable commands.
        2. Confirm the selected container or conda environment includes the science packages your chores import.
        3. Confirm you are inside the correct allocation/runtime for `{profile.name}`.
        4. Adjust resource requests on `@pipe.chore(...)` or `pipe.exec(...)`.
        5. For MPI chores, confirm MPI and Flux PMI settings are available on the site.

        ## Environment Setup

        {profile.install_summary}

        ```bash
        {cli_install}
        ```

        ## Allocation Setup

        ```bash
        {profile.interactive_setup}
        ```

        ## Recommended Launch

        From this campaign directory, after completing the setup above:

        ```bash
        {profile.launch_command.replace("workflow.py", workflow_filename)}
        ```

        ## Batch Script Guidance

        {profile.batch_notes}

        If you write a site batch script manually, the final workflow command should be equivalent to:

        ```bash
        {profile.launch_command.replace("workflow.py", workflow_filename)}
        ```

        ## Container Contents

        {_markdown_bullets(profile.container_summary)}

        ## Outputs to Inspect

        MatEnsemble creates a timestamped directory named like:

        ```text
        matensemble_workflow-YYYYMMDD_HHMMSS/
        ```

        Inspect:

        - `status.json`
        - `matensemble_workflow.log`
        - `out/<chore-id>/stdout`
        - `out/<chore-id>/stderr`
        - `out/<chore-id>/metadata.json`
        - `out/<chore-id>/result.pickle` for Python chore results

        ## Safety Note

        This MCP server generated files only. It did not submit jobs or execute shell commands.
        """
    )


def render_interactive_script(*, site: str, workflow_filename: str) -> str:
    profile = get_system_profile(site)
    if profile.name == "frontier":
        return dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail

            if ! command -v matensemble >/dev/null 2>&1; then
              echo "Install the MatEnsemble MCP/site CLI from the cloned repo first: uv run --package mcp-matensemble matensemble-agent-install --system frontier" >&2
              exit 1
            fi

            module reset
            module load olcf-container-tools
            module load apptainer-enable-mpi apptainer-enable-gpu

            matensemble set-image ./matensemble.sif
            matensemble run {workflow_filename}
            """
        )
    if profile.name == "perlmutter":
        image = resolve_image_tag(profile.name)
        return dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail

            if ! command -v matensemble >/dev/null 2>&1; then
              echo "Install the MatEnsemble MCP/site CLI from the cloned repo first: uv run --package mcp-matensemble matensemble-agent-install --system perlmutter" >&2
              exit 1
            fi

            matensemble set-image {image}
            matensemble run {workflow_filename}
            """
        )
    if profile.name == "linux":
        return f"#!/usr/bin/env bash\nset -euo pipefail\nflux start --test-size=4 python {workflow_filename}\n"
    return f"#!/usr/bin/env bash\nset -euo pipefail\n{profile.launch_command.replace('workflow.py', workflow_filename)}\n"


def render_batch_script(
    *,
    site: str,
    campaign_name: str,
    workflow_filename: str,
    matensemble_version: str | None = None,
) -> str:
    profile = get_system_profile(site)
    image = resolve_image_tag(profile.name, version=matensemble_version)
    job_name = slugify(campaign_name)[:32] or "matensemble"
    if profile.name == "frontier":
        return dedent(
            f"""\
            #!/usr/bin/env bash
            #SBATCH -A <account>
            #SBATCH -J {job_name}
            #SBATCH -o logs/%x-%j.out
            #SBATCH -e logs/%x-%j.err
            #SBATCH -t 00:10:00
            #SBATCH -N 2
            #SBATCH -C nvme

            set -euo pipefail
            cd "$(dirname "$0")"
            mkdir -p logs

            if ! command -v matensemble >/dev/null 2>&1; then
              echo "Install the MatEnsemble MCP/site CLI from the cloned repo first: uv run --package mcp-matensemble matensemble-agent-install --system frontier" >&2
              exit 1
            fi

            module reset
            module load olcf-container-tools
            module load apptainer-enable-mpi apptainer-enable-gpu

            matensemble set-image ./matensemble.sif
            matensemble run {workflow_filename}
            """
        )
    if profile.name == "perlmutter":
        return dedent(
            f"""\
            #!/usr/bin/env bash
            #SBATCH -A <account>
            #SBATCH -C gpu
            #SBATCH --qos debug
            #SBATCH -t 00:30:00
            #SBATCH -N 2
            #SBATCH --ntasks-per-node=1
            #SBATCH --gpus-per-node=4
            #SBATCH --gpu-bind=closest
            #SBATCH -J {job_name}
            #SBATCH -o logs/%x-%j.out
            #SBATCH -e logs/%x-%j.err

            set -euo pipefail
            cd "$(dirname "$0")"
            mkdir -p logs

            if ! command -v matensemble >/dev/null 2>&1; then
              echo "Install the MatEnsemble MCP/site CLI from the cloned repo first: uv run --package mcp-matensemble matensemble-agent-install --system perlmutter" >&2
              exit 1
            fi

            matensemble set-image {image}
            matensemble run {workflow_filename}
            """
        )
    if profile.name == "pathfinder":
        return dedent(
            f"""\
            #!/usr/bin/env bash
            #SBATCH -A <account>
            #SBATCH -J {job_name}
            #SBATCH -o logs/%x-%j.out
            #SBATCH -e logs/%x-%j.err
            #SBATCH -t 00:10:00
            #SBATCH -N 2

            set -euo pipefail
            cd "$(dirname "$0")"
            mkdir -p logs
            matensemble set-image ./matensemble.sif
            matensemble run {workflow_filename}
            """
        )
    return dedent(
        f"""\
        #!/usr/bin/env bash
        #SBATCH -J {job_name}
        #SBATCH -o logs/%x-%j.out
        #SBATCH -e logs/%x-%j.err
        #SBATCH -t 00:10:00
        #SBATCH -N 1

        set -euo pipefail
        cd "$(dirname "$0")"
        mkdir -p logs
        {profile.launch_command.replace("workflow.py", workflow_filename)}
        """
    )


def _markdown_bullets(items: tuple[str, ...]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _python_dag_template() -> str:
    return dedent(
        """\
        from matensemble.pipeline import Pipeline


        pipe = Pipeline()


        @pipe.chore(name="prepare_inputs", num_tasks=1, cores_per_task=1)
        def prepare_inputs() -> dict:
            \"\"\"Prepare or discover input data for the campaign.\"\"\"

            return {
                "structure_path": "inputs/structure.ext",
                "metadata": {"replace": "with campaign-specific values"},
            }


        @pipe.chore(name="run_model", num_tasks=1, cores_per_task=1, gpus_per_task=0)
        def run_model(prepared: dict) -> dict:
            \"\"\"Run the main science calculation using prepared inputs.\"\"\"

            return {
                "structure_path": prepared["structure_path"],
                "result": "replace this placeholder with calculation output",
            }


        @pipe.chore(name="analyze_results", num_tasks=1, cores_per_task=1)
        def analyze_results(model_result: dict) -> dict:
            \"\"\"Analyze the calculation output and return summary data.\"\"\"

            return {
                "summary": "replace this placeholder with analysis",
                "source": model_result,
            }


        prepared = prepare_inputs()
        model_result = run_model(prepared)
        analysis = analyze_results(model_result)

        future = pipe.submit(log_delay=5, dashboard=False)
        print(future.result())
        """
    )


def _executable_template() -> str:
    return dedent(
        """\
        from matensemble.pipeline import Pipeline


        pipe = Pipeline()

        # Prefer argv lists over shell strings. Replace this command with the
        # executable used by your campaign.
        pipe.exec(
            command=["/bin/echo", "replace with science executable"],
            name="run_executable",
            num_tasks=1,
            cores_per_task=1,
            gpus_per_task=0,
            mpi=False,
        )

        future = pipe.submit(log_delay=5, dashboard=False)
        print(future.result())
        """
    )


def _mpi_template() -> str:
    return dedent(
        """\
        from matensemble.pipeline import Pipeline


        pipe = Pipeline()


        @pipe.chore(name="mpi_step", num_tasks=4, cores_per_task=1, gpus_per_task=0, mpi=True)
        def mpi_step() -> dict:
            \"\"\"Run an MPI-aware Python calculation.\"\"\"

            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            return {
                "rank": comm.Get_rank(),
                "size": comm.Get_size(),
                "processor": MPI.Get_processor_name(),
            }


        for _ in range(4):
            mpi_step()

        future = pipe.submit(log_delay=5, dashboard=False)
        print(future.result())
        """
    )


def _strategy_template() -> str:
    return dedent(
        """\
        from matensemble.chore import ChoreSpec
        from matensemble.model import Resources
        from matensemble.pipeline import Pipeline


        pipe = Pipeline()


        @pipe.chore(name="sample", num_tasks=1, cores_per_task=1)
        def sample(value: int) -> dict:
            \"\"\"Evaluate one candidate in an adaptive campaign.\"\"\"

            return {"value": value, "score": value * value}


        @pipe.strategy(bolo_list=["sample"], name="choose_next")
        def choose_next(result: dict):
            \"\"\"Spawn additional work from completed sample results.\"\"\"

            if result["value"] >= 3:
                return None
            return ChoreSpec(
                args=(result["value"] + 1,),
                kwargs={},
                qualname="sample",
                resources=Resources(num_tasks=1, cores_per_task=1),
            )


        sample(0)

        future = pipe.submit(log_delay=5, dashboard=False)
        print(future.result())
        """
    )
