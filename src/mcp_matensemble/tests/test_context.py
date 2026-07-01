from __future__ import annotations

from pathlib import Path

import pytest

from mcp_matensemble import context


def test_examples_include_generic_and_selected_system():
    files = context.get_examples_for_system("perlmutter")

    assert "example_workflows/generic/mpi/workflow.py" in files
    assert "example_workflows/perlmutter/lammps_smoke/workflow.py" in files


def test_example_batch_scripts_include_selected_system_submit_files():
    scripts = context.get_example_batch_scripts("perlmutter")

    assert set(scripts) == {"lammps_mace", "lammps_ovito", "lammps_smoke"}
    assert scripts["lammps_smoke"]["path"] == (
        "example_workflows/perlmutter/lammps_smoke/submit.slurm"
    )
    assert "#SBATCH" in scripts["lammps_smoke"]["content"]


def test_containerfiles_read_selected_system():
    files = context.get_containerfiles("frontier")

    assert "containers/frontier/Dockerfile.base" in files
    assert "containers/frontier/Dockerfile.matensemble" in files


def test_core_source_returns_expected_files():
    files = context.get_matensemble_core()

    assert set(files) == {
        "src/matensemble/pipeline.py",
        "src/matensemble/manager.py",
        "src/matensemble/fluxlet.py",
        "src/matensemble/strategy.py",
        "src/matensemble/chore.py",
    }
    assert "class Pipeline" in files["src/matensemble/pipeline.py"]


def test_full_source_returns_package_files():
    files = context.get_full_matensemble_code()

    assert "src/matensemble/pipeline.py" in files
    assert "src/matensemble/dashboard/app.py" in files


def test_version_falls_back_to_pyproject(monkeypatch: pytest.MonkeyPatch):
    def raise_not_found(_name: str) -> str:
        raise context.metadata.PackageNotFoundError

    monkeypatch.setattr(context.metadata, "version", raise_not_found)

    result = context.get_matensemble_version()

    assert result == {
        "version": "0.5.0",
        "tag_version": "v0.5.0",
        "source": "pyproject.toml",
    }


def test_latest_container_tags_are_deterministic(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        context,
        "get_matensemble_version",
        lambda: {"version": "1.2.3", "tag_version": "v1.2.3", "source": "test"},
    )

    tags = context.get_latest_container_tags()

    assert tags["registry_probe_performed"] is False
    assert tags["tags"] == {
        "frontier": "ghcr.io/freddude2004/matensemble:frontier-v1.2.3",
        "perlmutter": "ghcr.io/freddude2004/matensemble:perlmutter-v1.2.3",
        "pathfinder": "ghcr.io/freddude2004/matensemble:pathfinder-v1.2.3",
    }


def test_container_build_command_matches_system_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        context,
        "get_matensemble_version",
        lambda: {"version": "1.2.3", "tag_version": "v1.2.3", "source": "test"},
    )

    frontier = context.get_container_build_command("frontier")
    perlmutter = context.get_container_build_command("perlmutter")

    assert frontier["command"] == [
        "apptainer",
        "build",
        "containers/frontier/matensemble.sif",
        "docker://ghcr.io/freddude2004/matensemble:frontier-v1.2.3",
    ]
    assert perlmutter["command"] == [
        "podman-hpc",
        "pull",
        "ghcr.io/freddude2004/matensemble:perlmutter-v1.2.3",
    ]


def test_file_tree_uses_fixed_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    root = tmp_path / "repo"
    (root / "example_workflows" / "generic").mkdir(parents=True)
    (root / "example_workflows" / "frontier").mkdir(parents=True)
    (root / "containers" / "frontier").mkdir(parents=True)
    (root / "src" / "matensemble").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "matensemble"\nversion = "9.9.9"\n',
        encoding="utf-8",
    )
    (root / "example_workflows" / "generic" / "workflow.py").write_text(
        "GENERIC = True\n", encoding="utf-8"
    )
    (root / "example_workflows" / "frontier" / "workflow.py").write_text(
        "FRONTIER = True\n", encoding="utf-8"
    )
    (root / "example_workflows" / "frontier" / "demo").mkdir()
    (root / "example_workflows" / "frontier" / "demo" / "submit.slurm").write_text(
        "#SBATCH -J demo\n", encoding="utf-8"
    )

    monkeypatch.setattr(context, "repo_root", lambda: root)

    assert context.get_examples_for_system("frontier") == {
        "example_workflows/frontier/demo/submit.slurm": "#SBATCH -J demo\n",
        "example_workflows/generic/workflow.py": "GENERIC = True\n",
        "example_workflows/frontier/workflow.py": "FRONTIER = True\n",
    }
    assert context.get_example_batch_scripts("frontier") == {
        "demo": {
            "path": "example_workflows/frontier/demo/submit.slurm",
            "content": "#SBATCH -J demo\n",
        }
    }
