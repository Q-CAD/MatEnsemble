from __future__ import annotations

from mcp_matensemble.environment import (
    get_version_info,
    plan_container_setup,
    resolve_image_tag,
    run_container_setup,
)


def test_resolve_image_tag_uses_local_version():
    image = resolve_image_tag("perlmutter")

    assert image.startswith("ghcr.io/freddude2004/matensemble:perlmutter-v")


def test_plan_frontier_setup_builds_sif():
    plan = plan_container_setup(
        "frontier",
        version="0.3.11",
        output_path="matensemble.sif",
    )

    command = plan["commands"][0]
    assert command["argv"] == [
        "apptainer",
        "build",
        "matensemble.sif",
        "docker://ghcr.io/freddude2004/matensemble:frontier-v0.3.11",
    ]


def test_run_container_setup_defaults_to_dry_run():
    result = run_container_setup(
        "perlmutter",
        action="pull_image",
        version="0.3.11",
    )

    assert result["executed"] is False
    assert result["command"]["argv"] == [
        "podman-hpc",
        "pull",
        "ghcr.io/freddude2004/matensemble:perlmutter-v0.3.11",
    ]


def test_version_info_does_not_require_network():
    info = get_version_info(check_pypi=False)

    assert info["local_version"] is not None
    assert info["pypi_version"] is None
