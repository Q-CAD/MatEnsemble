from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from .systems import SUPPORTED_SYSTEMS, normalize_system

GENERIC_FLUX_COMPATIBLE_SYSTEMS = ("frontier", "perlmutter", "pathfinder", "linux")

API_GUIDANCE = """\
MatEnsemble workflows are built with matensemble.pipeline.Pipeline.

Core patterns:
- Create a Pipeline with pipe = Pipeline().
- Use @pipe.chore(...) on top-level Python functions to create delayed Python chores.
- Calling a decorated chore appends work and returns an OutputReference.
- Pass OutputReference values to later Python chores to create dependencies.
- Use pipe.exec(command=[...]) for executable chores. Prefer argv lists over shell strings.
- Use pipe.strategy(bolo_list=[...]) for adaptive strategies that can return ChoreSpec.
- Call pipe.submit(...) only after the graph has been constructed.
- Call future.result() or pipe.results() to collect Python chore outputs.

Operational expectations:
- Pipeline.submit() expects a Flux-capable runtime environment.
- For Slurm systems, launch a batch allocation that starts Flux, then run the workflow script.
- Inspect status.json, matensemble_workflow.log, and per-chore stdout/stderr under the generated workflow directory.
"""

EXAMPLE_ALIASES = {
    "chores": "generic_flux.chores",
    "common_chores": "generic_flux.chores",
    "generic_flux_chores": "generic_flux.chores",
    "generic.chores": "generic_flux.chores",
    "generic.dependencies": "generic_flux.chores",
    "generic_dependencies": "generic_flux.chores",
    "executable": "generic_flux.executable",
    "common_executable": "generic_flux.executable",
    "generic_flux_executable": "generic_flux.executable",
    "generic.executable": "generic_flux.executable",
    "generic_executable": "generic_flux.executable",
    "mpi": "generic_flux.mpi",
    "common_mpi": "generic_flux.mpi",
    "generic_flux_mpi": "generic_flux.mpi",
    "generic.mpi": "generic_flux.mpi",
    "generic_mpi": "generic_flux.mpi",
    "strategy": "generic_flux.strategy",
    "common_strategy": "generic_flux.strategy",
    "generic_flux_strategy": "generic_flux.strategy",
    "generic.strategy": "generic_flux.strategy",
    "generic_strategy": "generic_flux.strategy",
    "frontier_lammps_smoke": "frontier.lammps_smoke",
    "pathfinder_lammps_smoke": "pathfinder.lammps_smoke",
    "perlmutter_lammps_smoke": "perlmutter.lammps_smoke",
    "perlmutter_lammps_mace": "perlmutter.lammps_mace",
    "perlmutter_dependency_campaign": "perlmutter.dependency_campaign",
}

_GENERIC_DIRECTORY_ALIASES = {"chores": "dependencies"}
_IGNORED_DIRECTORY_NAMES = {"__pycache__"}


@dataclass(frozen=True)
class ExampleSummary:
    name: str
    path: str
    demonstrates: str
    id: str
    title: str
    system: str
    system_title: str
    compatible_systems: tuple[str, ...] = ()
    agent_guidance: str | None = None


def get_examples(system: str) -> list[dict[str, object]]:
    """Return generic workflows plus files for the selected system."""

    normalized = normalize_system(system)
    root = _require_repo_root()
    generic_dir = _example_system_dir(root, "linux")
    files = _read_file_tree(generic_dir, root)
    if normalized != "linux":
        files.extend(_read_file_tree(_example_system_dir(root, normalized), root))
    return files


def get_container_build_info(system: str) -> list[dict[str, object]]:
    """Return every file currently present in containers/<system>/."""

    normalized = normalize_system(system)
    root = _require_repo_root()
    container_dir = root / "containers" / normalized
    if not container_dir.is_dir():
        raise ValueError(f"container directory not found: containers/{normalized}")
    return _read_file_tree(container_dir, root)


def list_examples() -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for system in SUPPORTED_SYSTEMS:
        summaries.extend(_list_examples_for_exact_system(system))
    return summaries


def list_examples_for_system(system: str | None = None) -> list[dict[str, object]]:
    if system is None:
        return list_examples()

    normalized = normalize_system(system)
    summaries = _list_examples_for_exact_system("linux")
    if normalized != "linux":
        summaries.extend(_list_examples_for_exact_system(normalized))
    return summaries


def _list_examples_for_exact_system(system: str) -> list[dict[str, object]]:
    root = _require_repo_root()
    example_dir = _example_system_dir(root, system)
    summaries: list[dict[str, object]] = []
    for directory in sorted(path for path in example_dir.iterdir() if path.is_dir()):
        files = _iter_files(directory)
        if not files:
            continue
        public_name = _public_example_name(system, directory.name)
        prefix = "generic_flux" if system == "linux" else system
        example_id = f"{prefix}.{public_name}"
        repo_paths = ", ".join(str(path.relative_to(root)) for path in files)
        source_path = directory.relative_to(root)
        compatible = GENERIC_FLUX_COMPATIBLE_SYSTEMS if system == "linux" else ()
        system_title = _system_title(system)
        summaries.append(
            ExampleSummary(
                name=f"{prefix}_{public_name}",
                path=repo_paths,
                demonstrates=f"[{system_title}] Files loaded from {source_path}.",
                id=example_id,
                title=public_name.replace("_", " ").title(),
                system=system,
                system_title=system_title,
                compatible_systems=compatible,
                agent_guidance=f"Use the files in {source_path} as the source of truth.",
            ).__dict__
        )
    return summaries


def get_example_source(name: str) -> str:
    key = EXAMPLE_ALIASES.get(name, name)
    requested_system, example_name = _split_example_name(key)
    if requested_system is not None:
        return get_system_example(requested_system, example_name)

    matches: list[tuple[str, str]] = []
    for system in SUPPORTED_SYSTEMS:
        try:
            matches.append((system, get_system_example(system, example_name)))
        except ValueError:
            continue
    if len(matches) == 1:
        return matches[0][1]
    if not matches:
        raise ValueError(f"unknown example {name!r}")
    systems = [system for system, _ in matches]
    raise ValueError(f"ambiguous example {name!r}; specify one of {systems}")


def get_system_example(system: str, name: str) -> str:
    normalized = normalize_system(system)
    key = EXAMPLE_ALIASES.get(name, name)
    alias_system, example_name = _split_example_name(key)
    if alias_system is not None:
        normalized = alias_system

    root = _require_repo_root()
    example_root = _example_system_dir(root, normalized).resolve()
    example_name = _GENERIC_DIRECTORY_ALIASES.get(example_name, example_name)
    candidate = (example_root / example_name).resolve()
    _ensure_within(candidate, example_root, "example path")

    if candidate.is_file():
        return candidate.read_text(encoding="utf-8", errors="replace")
    if candidate.is_dir():
        return _render_file_bundle(_read_file_tree(candidate, root))
    raise ValueError(f"example not found: {candidate.relative_to(root)}")


def list_container_templates(system: str | None = None) -> list[dict[str, object]]:
    root = _require_repo_root()
    systems = (normalize_system(system),) if system else SUPPORTED_SYSTEMS
    templates: list[dict[str, object]] = []
    for name in systems:
        system_dir = root / "containers" / name
        if not system_dir.is_dir():
            continue
        for path in _iter_files(system_dir):
            templates.append(
                {
                    "system": name,
                    "filename": str(path.relative_to(system_dir)),
                    "path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                }
            )
    return templates


def get_container_template(system: str, filename: str) -> str:
    normalized = normalize_system(system)
    root = _require_repo_root()
    container_root = (root / "containers" / normalized).resolve()
    path = (container_root / filename).resolve()
    _ensure_within(path, container_root, "container template path")
    if not path.is_file():
        raise ValueError(
            f"container template not found: containers/{normalized}/{filename}"
        )
    return path.read_text(encoding="utf-8", errors="replace")


def get_container_contents(name: str) -> str:
    """Return the live contents of every container file for a system."""

    return _render_file_bundle(get_container_build_info(name))


def how_to_build_container(name: str) -> str:
    from .systems import render_environment_setup

    return render_environment_setup(name)


def _repo_root() -> Path | None:
    configured = os.environ.get("MATENSEMBLE_REPO_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / "example_workflows").is_dir() and (
            candidate / "containers"
        ).is_dir():
            return candidate.resolve()
    return None


def _require_repo_root() -> Path:
    root = _repo_root()
    if root is None:
        raise RuntimeError(
            "MatEnsemble repository root could not be located; set "
            "MATENSEMBLE_REPO_ROOT to a checkout containing example_workflows/ "
            "and containers/."
        )
    return root


def _example_system_dir(root: Path, system: str) -> Path:
    directory_name = "generic" if system == "linux" else system
    path = root / "example_workflows" / directory_name
    if not path.is_dir():
        raise ValueError(
            f"example directory not found: example_workflows/{directory_name}"
        )
    return path


def _iter_files(directory: Path) -> list[Path]:
    return [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and not any(part in _IGNORED_DIRECTORY_NAMES for part in path.parts)
    ]


def _read_file_tree(directory: Path, root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "content": path.read_text(encoding="utf-8", errors="replace"),
            "size_bytes": path.stat().st_size,
        }
        for path in _iter_files(directory)
    ]


def _render_file_bundle(files: list[dict[str, object]]) -> str:
    sections = []
    for file in files:
        sections.append(f"# --- {file['path']} ---\n{file['content']}")
    return "\n\n".join(sections)


def _split_example_name(name: str) -> tuple[str | None, str]:
    if "." not in name:
        return None, name
    prefix, example_name = name.split(".", 1)
    if prefix == "generic_flux" or prefix == "generic":
        return "linux", example_name
    if prefix in SUPPORTED_SYSTEMS:
        return prefix, example_name
    return None, name


def _public_example_name(system: str, directory_name: str) -> str:
    if system == "linux" and directory_name == "dependencies":
        return "chores"
    return directory_name


def _system_title(system: str) -> str:
    if system == "linux":
        return "Portable Linux/Flux Workflows"
    return system.replace("_", " ").title()


def _ensure_within(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside {root}") from exc
