from __future__ import annotations

from mcp_matensemble.resources import (
    container_build_instructions,
    examples_overview,
    read_container_contents,
    read_repo_example,
)


def test_read_example_source_does_not_depend_on_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    source = read_repo_example("generic_flux.mpi")

    assert "Portable Flux Workflows" in source
    assert "Frontier, Perlmutter, Pathfinder" in source
    assert "get_matensemble_system(<site>)" in source
    assert "from matensemble.pipeline import Pipeline" in source
    assert "mpi=True" in source


def test_legacy_example_names_resolve_to_portable_examples():
    source = read_repo_example("mpi")

    assert "# Example: generic_flux.mpi" in source
    assert "site-independent MatEnsemble examples" in source


def test_generic_flux_examples_are_labeled_portable():
    examples = examples_overview()
    generic_examples = [
        example for example in examples if example["system"] == "generic_flux"
    ]

    assert generic_examples
    for example in generic_examples:
        assert example["system_title"] == "Portable Flux Workflows"
        assert "site-independent MatEnsemble examples" in example["demonstrates"]
        assert set(example["compatible_systems"]) == {
            "frontier",
            "perlmutter",
            "pathfinder",
            "linux",
        }


def test_examples_include_site_batch_scripts():
    examples = {example["name"]: example for example in examples_overview()}

    assert "frontier_lammps_smoke" in examples
    assert "perlmutter_lammps_smoke" in examples
    assert "Apptainer" in examples["frontier_lammps_smoke"]["agent_guidance"]
    assert "Podman-HPC" in examples["perlmutter_lammps_smoke"]["agent_guidance"]


def test_container_context_and_install_guidance():
    contents = read_container_contents("linux")
    install = container_build_instructions("perlmutter")

    assert "Flux core" in contents
    assert "podman-hpc pull" in install
    assert "matensemble run workflow.py" in install
