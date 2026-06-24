from __future__ import annotations

import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any

SUPPORTED_SYSTEMS = ("frontier", "perlmutter", "pathfinder")
GHCR_NAMESPACE = "ghcr.io/freddude2004/matensemble"

CORE_SOURCE_FILES = (
    "pipeline.py",
    "manager.py",
    "fluxlet.py",
    "strategy.py",
    "chore.py",
)

API_OVERVIEW = """\
MatEnsemble is a Python workflow library for adaptive, asynchronous ensemble
simulations. Agents should treat it as a code-first workflow API: users write
Python scripts that define chores, dependencies, and adaptive strategy logic,
then run those scripts inside a Flux-capable runtime.

Core API shape:
- Create a workflow with `Pipeline()`.
- Use `@pipe.chore(...)` on top-level Python functions for Python chores.
- Calling a decorated chore queues work and returns an output reference.
- Pass output references into later chores to express dependencies.
- Use `pipe.exec(command=[...])` for executable chores such as MPI programs,
  LAMMPS commands, analysis scripts, or other container-provided binaries.
- Use `pipe.strategy(...)` for adaptive logic that can add more chores while
  the workflow is running.
- Call `pipe.submit(...)` after building the workflow graph.
- Use returned futures, `future.result()`, or `pipe.results()` to collect Python
  chore outputs.

Python chores are ordinary Python functions run by MatEnsemble workers. Keep
their inputs and outputs serializable, keep import requirements available inside
the container, and prefer explicit file paths for simulation artifacts. Python
chores are a good fit for preprocessing, small analyses, model selection,
postprocessing, bookkeeping, and adaptive decision logic.

Executable chores are external commands launched by MatEnsemble. Prefer argv
lists over shell strings. Use executable chores for compiled simulation codes,
MPI commands, GPU workloads, and existing scripts. The command must make sense
inside the selected MatEnsemble container and HPC allocation.

Operationally, MatEnsemble expects Flux. On HPC systems, users normally request
a Slurm allocation, enter the MatEnsemble container with the site CLI, start
Flux, and run the Python workflow. Inspect `status.json`,
`matensemble_workflow.log`, and per-chore stdout/stderr under the generated
workflow directory when debugging.
"""

CONTAINERS_OVERVIEW = """\
MatEnsemble on HPC is targeted at containers. The site CLI expects users to
configure a MatEnsemble container image first, then uses that image when running
workflows in an allocation.

The standard image tag pattern is:

    ghcr.io/freddude2004/matensemble:<system>-vX.Y.Z

Where `<system>` is one of `frontier`, `perlmutter`, or `pathfinder`, and
`X.Y.Z` is the local MatEnsemble version. The MCP server does not probe GHCR for
tags; it derives the expected latest tags from the checked-out MatEnsemble
version.

Frontier and Pathfinder normally use Apptainer `.sif` images built or pulled
from the GHCR image. Perlmutter normally uses `podman-hpc pull` with the GHCR
tag. If a user needs extra dependencies, start FROM the MatEnsemble image for
their system and install additional apt, uv/pip, or compiled dependencies in a
custom container layer.
"""


def normalize_system(system: str | None) -> str:
    if system is None:
        raise ValueError("system is required")
    key = system.strip().lower().replace("-", "_")
    aliases = {
        "olcf_frontier": "frontier",
        "olcf_pathfinder": "pathfinder",
        "nersc": "perlmutter",
    }
    normalized = aliases.get(key, key)
    if normalized not in SUPPORTED_SYSTEMS:
        raise ValueError(
            f"unsupported system {system!r}; expected one of {', '.join(SUPPORTED_SYSTEMS)}"
        )
    return normalized


def repo_root() -> Path:
    configured = Path.cwd()
    candidates = [configured, *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "src" / "matensemble").is_dir()
            and (candidate / "example_workflows").is_dir()
            and (candidate / "containers").is_dir()
        ):
            return candidate.resolve()
    raise RuntimeError("MatEnsemble repository root could not be located")


def get_api_overview() -> str:
    return API_OVERVIEW


def get_containers_overview() -> str:
    return CONTAINERS_OVERVIEW


def get_examples_for_system(system: str | None = None) -> dict[str, str]:
    key = normalize_system(system)
    root = repo_root()
    files: dict[str, str] = {}
    for directory in (root / "example_workflows" / "generic", root / "example_workflows" / key):
        files.update(_read_tree(directory, root))
    return files


def get_containerfiles(system: str | None = None) -> dict[str, str]:
    key = normalize_system(system)
    root = repo_root()
    return _read_tree(root / "containers" / key, root)


def get_container_build_command(system: str | None = None) -> dict[str, Any]:
    key = normalize_system(system)
    version = get_matensemble_version()["version"]
    image = container_tag(key, version)
    if key == "perlmutter":
        command = ["podman-hpc", "pull", image]
        command_text = "podman-hpc pull " + image
        backend = "podman-hpc"
    else:
        output = f"containers/{key}/matensemble.sif"
        command = ["apptainer", "build", output, f"docker://{image}"]
        command_text = f"apptainer build {output} docker://{image}"
        backend = "apptainer"
    return {
        "system": key,
        "version": version,
        "image": image,
        "backend": backend,
        "command": command,
        "command_text": command_text,
        "custom_base_image": f"FROM {image}",
        "notes": (
            "Use this MatEnsemble image as the base for custom containers, then "
            "install workflow-specific dependencies with apt, uv/pip, or compiled "
            "source as needed."
        ),
    }


def get_matensemble_core() -> dict[str, str]:
    root = repo_root()
    source_root = root / "src" / "matensemble"
    return {
        str((source_root / filename).relative_to(root)): (source_root / filename).read_text(
            encoding="utf-8", errors="replace"
        )
        for filename in CORE_SOURCE_FILES
    }


def get_full_matensemble_code() -> dict[str, str]:
    root = repo_root()
    return _read_tree(root / "src" / "matensemble", root)


def get_matensemble_version() -> dict[str, str]:
    try:
        version = metadata.version("matensemble")
        source = "package_metadata"
    except metadata.PackageNotFoundError:
        data = tomllib.loads((repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
        version = str(data["project"]["version"])
        source = "pyproject.toml"
    return {"version": version, "tag_version": f"v{version}", "source": source}


def get_latest_container_tags() -> dict[str, Any]:
    version = get_matensemble_version()["version"]
    return {
        "version": version,
        "registry_probe_performed": False,
        "pattern": f"{GHCR_NAMESPACE}:<system>-vX.Y.Z",
        "tags": {system: container_tag(system, version) for system in SUPPORTED_SYSTEMS},
    }


def container_tag(system: str, version: str) -> str:
    key = normalize_system(system)
    return f"{GHCR_NAMESPACE}:{key}-v{version}"


def _read_tree(directory: Path, root: Path) -> dict[str, str]:
    if not directory.is_dir():
        raise ValueError(f"directory not found: {directory.relative_to(root)}")
    files: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
            files[str(path.relative_to(root))] = path.read_text(
                encoding="utf-8", errors="replace"
            )
    return files
