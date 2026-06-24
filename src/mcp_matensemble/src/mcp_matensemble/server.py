from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

from mcp.server.fastmcp import FastMCP

from . import context
from .dashboard import (
    get_dashboard_access as dashboard_access,
    launch_dashboard as launch_dashboard_process,
    stop_dashboard as stop_dashboard_server,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mcp-matensemble")
    parser.add_argument(
        "--system",
        required=True,
        choices=context.SUPPORTED_SYSTEMS,
        help="HPC system this MCP server should default to.",
    )
    return parser


def create_server(default_system: str) -> FastMCP:
    system = context.normalize_system(default_system)
    mcp = FastMCP("mcp-matensemble")

    @mcp.tool()
    def get_api_overview() -> str:
        """Return a concise overview of the MatEnsemble workflow API."""

        return context.get_api_overview()

    @mcp.tool()
    def get_containers_overview() -> str:
        """Return MatEnsemble HPC container guidance."""

        return context.get_containers_overview()

    @mcp.tool()
    def get_examples_for_system(system_override: str | None = None) -> dict[str, str]:
        """Return generic examples plus examples for the configured system."""

        return context.get_examples_for_system(system_override or system)

    @mcp.tool()
    def get_example_batch_scripts(
        system_override: str | None = None,
    ) -> dict[str, dict[str, str]]:
        """Return example submit.slurm scripts for the configured system."""

        return context.get_example_batch_scripts(system_override or system)

    @mcp.tool()
    def get_containerfiles(system_override: str | None = None) -> dict[str, str]:
        """Return every file under containers/<system>."""

        return context.get_containerfiles(system_override or system)

    @mcp.tool()
    def get_container_build_command(system_override: str | None = None) -> dict[str, Any]:
        """Return the deterministic image tag and suggested build/pull command."""

        return context.get_container_build_command(system_override or system)

    @mcp.tool()
    def get_matensemble_core() -> dict[str, str]:
        """Return the core MatEnsemble source files."""

        return context.get_matensemble_core()

    @mcp.tool()
    def get_full_matensemble_code() -> dict[str, str]:
        """Return all source files under src/matensemble."""

        return context.get_full_matensemble_code()

    @mcp.tool()
    def get_matensemble_version() -> dict[str, str]:
        """Return the local MatEnsemble version used for deterministic tags."""

        return context.get_matensemble_version()

    @mcp.tool()
    def get_latest_container_tags() -> dict[str, Any]:
        """Return deterministic latest MatEnsemble container tags without probing GHCR."""

        return context.get_latest_container_tags()

    @mcp.tool()
    def launch_dashboard(campaign_root: str, port: int = 8000) -> dict[str, Any]:
        """Start the MatEnsemble dashboard in the background on this host."""

        return launch_dashboard_process(campaign_root, port=port)

    @mcp.tool()
    def get_dashboard_access(
        login_host: str | None = None,
        login_user: str | None = None,
        remote_port: int = 8000,
        local_port: int = 8000,
    ) -> dict[str, Any]:
        """Return the SSH tunnel command for accessing the dashboard locally."""

        return dashboard_access(
            login_host=login_host,
            login_user=login_user,
            remote_port=remote_port,
            local_port=local_port,
        )

    @mcp.tool()
    def stop_dashboard(campaign_root: str, port: int = 8000) -> dict[str, Any]:
        """Stop a dashboard started by launch_dashboard."""

        return stop_dashboard_server(campaign_root, port=port)

    @mcp.resource("matensemble://api/overview")
    def api_overview_resource() -> str:
        return context.get_api_overview()

    @mcp.resource("matensemble://containers/overview")
    def containers_overview_resource() -> str:
        return context.get_containers_overview()

    @mcp.resource("matensemble://examples")
    def examples_resource() -> str:
        return json.dumps(context.get_examples_for_system(system), indent=2)

    @mcp.resource("matensemble://examples/batch-scripts")
    def example_batch_scripts_resource() -> str:
        return json.dumps(context.get_example_batch_scripts(system), indent=2)

    @mcp.resource("matensemble://containers/files")
    def containerfiles_resource() -> str:
        return json.dumps(context.get_containerfiles(system), indent=2)

    @mcp.resource("matensemble://source/core")
    def core_source_resource() -> str:
        return json.dumps(context.get_matensemble_core(), indent=2)

    @mcp.resource("matensemble://source/full")
    def full_source_resource() -> str:
        return json.dumps(context.get_full_matensemble_code(), indent=2)

    @mcp.resource("matensemble://version")
    def version_resource() -> str:
        return json.dumps(context.get_matensemble_version(), indent=2)

    @mcp.resource("matensemble://containers/latest-tags")
    def latest_container_tags_resource() -> str:
        return json.dumps(context.get_latest_container_tags(), indent=2)

    @mcp.prompt(name="start_dashboard")
    def start_dashboard_prompt() -> str:
        """Prompt an agent to start and tunnel the MatEnsemble dashboard."""

        return (
            "start the dashboard in the matensemble_campaigns directory and provide "
            "me the command to ssh tunnel and port forward the dashboard to localhost."
        )

    return mcp


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    create_server(args.system).run()


if __name__ == "__main__":
    main()
