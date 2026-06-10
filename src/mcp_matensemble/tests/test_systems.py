from __future__ import annotations

from mcp_matensemble.systems import get_system_profile, render_environment_setup


def test_perlmutter_profile_prefers_cli():
    profile = get_system_profile("perlmutter")

    assert profile.container_runtime == "Podman-HPC"
    assert "matensemble run workflow.py" == profile.launch_command
    assert "podman-hpc pull" in profile.cli_install


def test_frontier_profile_mentions_apptainer():
    setup = render_environment_setup("frontier")

    assert "Apptainer" in setup
    assert "matensemble set-image" in setup
