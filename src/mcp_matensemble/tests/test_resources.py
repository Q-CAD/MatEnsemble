from __future__ import annotations

from pathlib import Path

from mcp_matensemble import examples
from mcp_matensemble.resources import (
    container_build_instructions,
    examples_overview,
    read_container_contents,
    read_repo_example,
)
from mcp_matensemble.examples import get_container_build_info, get_examples


def test_read_example_source_does_not_depend_on_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    source = read_repo_example("generic_flux.mpi")

    assert "example_workflows/generic/mpi/workflow.py" in source
    assert "example_workflows/generic/mpi/README.md" in source
    assert "from matensemble.pipeline import Pipeline" in source
    assert "mpi=True" in source


def test_legacy_example_names_resolve_to_portable_examples():
    source = read_repo_example("mpi")

    assert "example_workflows/generic/mpi/workflow.py" in source


def test_generic_flux_examples_are_labeled_portable():
    examples = examples_overview()
    generic_examples = [
        example for example in examples if str(example["id"]).startswith("generic_flux.")
    ]

    assert generic_examples
    for example in generic_examples:
        assert example["system"] == "linux"
        assert example["system_title"] == "Portable Linux/Flux Workflows"
        assert "Portable Linux/Flux Workflows" in example["demonstrates"]
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
    assert "example_workflows/frontier/lammps_smoke" in examples["frontier_lammps_smoke"]["agent_guidance"]
    assert "example_workflows/perlmutter/lammps_smoke" in examples["perlmutter_lammps_smoke"]["agent_guidance"]


def test_system_examples_are_loaded_directly_from_repository():
    files = get_examples("perlmutter")
    paths = {file["path"] for file in files}
    root = Path(__file__).resolve().parents[3]

    assert paths == {
        str(path.relative_to(root))
        for directory in (
            root / "example_workflows" / "generic",
            root / "example_workflows" / "perlmutter",
        )
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_every_system_context_includes_generic_workflow_examples():
    for system in ("frontier", "perlmutter", "pathfinder", "linux"):
        paths = {file["path"] for file in get_examples(system)}
        summaries = examples.list_examples_for_system(system)

        assert "example_workflows/generic/dependencies/workflow.py" in paths
        assert "example_workflows/generic/mpi/workflow.py" in paths
        assert any(item["id"] == "generic_flux.chores" for item in summaries)
        assert any(item["id"] == "generic_flux.mpi" for item in summaries)


def test_example_changes_are_visible_without_updating_mcp(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    example_dir = root / "example_workflows" / "perlmutter"
    generic_dir = root / "example_workflows" / "generic"
    container_dir = root / "containers" / "perlmutter"
    example_dir.mkdir(parents=True)
    generic_dir.mkdir(parents=True)
    container_dir.mkdir(parents=True)
    (generic_dir / "generic_workflow.py").write_text(
        "GENERIC_EXAMPLE = True\n", encoding="utf-8"
    )
    (example_dir / "new_workflow.py").write_text("NEW_EXAMPLE = True\n", encoding="utf-8")
    (container_dir / "Dockerfile.new").write_text("FROM scratch\n", encoding="utf-8")

    monkeypatch.setattr(examples, "_repo_root", lambda: root)

    assert [file["content"] for file in get_examples("perlmutter")] == [
        "GENERIC_EXAMPLE = True\n",
        "NEW_EXAMPLE = True\n",
    ]
    assert get_container_build_info("perlmutter")[0]["content"] == "FROM scratch\n"


def test_container_context_and_install_guidance():
    contents = read_container_contents("linux")
    install = container_build_instructions("perlmutter")

    assert "containers/linux/Dockerfile" in contents
    assert "containers/linux/Dockerfile.matensemble" in contents
    assert "prepare_container_pull_plan" in install
    assert "local MatEnsemble version" in install
    assert "matensemble run workflow.py" in install
