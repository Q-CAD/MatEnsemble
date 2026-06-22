from __future__ import annotations

import pytest

from mcp_matensemble.systems import (
    SUPPORTED_SYSTEMS,
    UnsupportedSystemError,
    get_system_profile,
    render_environment_setup,
)


def test_perlmutter_profile_prefers_cli():
    profile = get_system_profile("perlmutter")

    assert profile.container_runtime == "Podman-HPC or Apptainer"
    assert profile.container_backends == ("podman-hpc", "apptainer")
    assert "matensemble run workflow.py" == profile.launch_command
    assert "prepare_container_pull_plan" in profile.cli_install
    assert "local MatEnsemble version" in profile.cli_install


def test_frontier_profile_mentions_apptainer():
    setup = render_environment_setup("frontier")

    assert "Apptainer" in setup
    assert "matensemble set-image" in setup


def test_supported_systems_are_strict():
    assert SUPPORTED_SYSTEMS == ("frontier", "perlmutter", "pathfinder", "linux")

    with pytest.raises(UnsupportedSystemError) as exc:
        get_system_profile("summit")

    payload = exc.value.to_error()
    assert payload["ok"] is False
    assert payload["error_code"] == "UNSUPPORTED_SYSTEM"
    assert payload["message"] == "Unsupported system: summit"
    assert payload["details"]["supported_systems"] == list(SUPPORTED_SYSTEMS)
