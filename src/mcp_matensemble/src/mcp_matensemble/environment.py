from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .security import append_audit_event
from .systems import get_system_profile, normalize_system


IMAGE_REPOSITORY = "ghcr.io/freddude2004/matensemble"
ALLOWED_EXECUTABLES = {
    "apptainer",
    "conda",
    "docker",
    "matensemble",
    "podman",
    "podman-hpc",
}


@dataclass(frozen=True)
class PlannedCommand:
    name: str
    description: str
    argv: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "argv": list(self.argv),
            "command": " ".join(_shell_quote(part) for part in self.argv),
        }


def get_version_info(*, check_pypi: bool = False) -> dict[str, Any]:
    """Return local MatEnsemble version information and optionally query PyPI."""

    info: dict[str, Any] = {
        "local_version": get_local_matensemble_version(),
        "pypi_version": None,
        "pypi_error": None,
    }
    if check_pypi:
        try:
            info["pypi_version"] = get_pypi_matensemble_version()
        except Exception as exc:  # network environments vary on clusters
            info["pypi_error"] = str(exc)
    return info


def get_local_matensemble_version() -> str | None:
    """Find the MatEnsemble project version from the repo or installed package."""

    for parent in Path(__file__).resolve().parents:
        pyproject = parent / "pyproject.toml"
        if not pyproject.exists():
            continue
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError:
            continue
        project = data.get("project", {})
        if project.get("name") == "matensemble":
            return project.get("version")

    try:
        return importlib.metadata.version("matensemble")
    except importlib.metadata.PackageNotFoundError:
        return None


def get_pypi_matensemble_version(*, timeout_seconds: float = 5.0) -> str:
    with urllib.request.urlopen(
        "https://pypi.org/pypi/matensemble/json",
        timeout=timeout_seconds,
    ) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload["info"]["version"])


def resolve_image_tag(
    system: str,
    *,
    version: str | None = None,
    image_tag: str | None = None,
) -> str:
    """Return a full MatEnsemble image reference for a system."""

    key = normalize_system(system)
    if image_tag:
        if "/" in image_tag and ":" in image_tag:
            return image_tag
        return f"{IMAGE_REPOSITORY}:{image_tag}"
    if key in {"generic_flux", "conda"}:
        return "none"

    resolved_version = normalize_version(version or get_local_matensemble_version())
    return f"{IMAGE_REPOSITORY}:{key}-{resolved_version}"


def normalize_version(version: str | None) -> str:
    if not version:
        return "vX.Y.Z"
    stripped = version.strip()
    if stripped.startswith("v"):
        return stripped
    return f"v{stripped}"


def plan_container_setup(
    system: str,
    *,
    version: str | None = None,
    image_tag: str | None = None,
    output_path: str | None = None,
    runtime: str | None = None,
) -> dict[str, Any]:
    """Return commands for preparing a MatEnsemble runtime environment."""

    key = normalize_system(system)
    profile = get_system_profile(key)
    image = resolve_image_tag(key, version=version, image_tag=image_tag)
    commands = _planned_commands(
        key,
        image=image,
        output_path=output_path,
        runtime=runtime,
    )

    return {
        "system": key,
        "title": profile.title,
        "image": image,
        "runtime": profile.container_runtime,
        "commands": [command.to_dict() for command in commands],
        "notes": profile.batch_notes,
        "execute_default": False,
    }


def run_container_setup(
    system: str,
    *,
    action: str = "auto",
    version: str | None = None,
    image_tag: str | None = None,
    output_path: str | None = None,
    runtime: str | None = None,
    execute: bool = False,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """
    Run one allowlisted setup command, or return a dry-run plan.

    Long-running container builds are intentionally opt-in via ``execute=True``.
    """

    key = normalize_system(system)
    image = resolve_image_tag(key, version=version, image_tag=image_tag)
    command = _select_command(
        key,
        action=action,
        image=image,
        output_path=output_path,
        runtime=runtime,
    )
    payload: dict[str, Any] = {
        "system": key,
        "action": action,
        "image": image,
        "command": command.to_dict(),
        "executed": False,
    }
    if not execute:
        payload["message"] = "Dry run only. Pass execute=True to run this command."
        return payload

    executable = command.argv[0]
    if executable not in ALLOWED_EXECUTABLES:
        raise ValueError(f"refusing to execute unsupported command: {executable}")
    if shutil.which(executable) is None:
        payload["returncode"] = None
        payload["stdout"] = ""
        payload["stderr"] = f"required command not found on PATH: {executable}"
        return payload

    completed = subprocess.run(
        list(command.argv),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    payload.update(
        {
            "executed": True,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    )
    append_audit_event(
        {
            "tool": "run_matensemble_container_setup",
            "command": list(command.argv),
            "returncode": completed.returncode,
            "system": key,
            "action": action,
        }
    )
    return payload


def _planned_commands(
    system: str,
    *,
    image: str,
    output_path: str | None,
    runtime: str | None,
) -> tuple[PlannedCommand, ...]:
    if system in {"frontier", "pathfinder"}:
        sif = output_path or f"matensemble-{system}.sif"
        return (
            PlannedCommand(
                name="build_sif",
                description=f"Build an Apptainer SIF for {system}.",
                argv=("apptainer", "build", sif, f"docker://{image}"),
            ),
            PlannedCommand(
                name="set_cli_image",
                description="Store the image path for the MatEnsemble CLI.",
                argv=("matensemble", "set-image", sif),
            ),
            PlannedCommand(
                name="run_workflow",
                description="Run a generated MatEnsemble workflow through the site CLI.",
                argv=("matensemble", "run", "workflow.py"),
            ),
        )
    if system == "perlmutter":
        return (
            PlannedCommand(
                name="pull_image",
                description="Pull the Perlmutter image through Podman-HPC.",
                argv=("podman-hpc", "pull", image),
            ),
            PlannedCommand(
                name="set_cli_image",
                description="Store the image tag for the MatEnsemble CLI.",
                argv=("matensemble", "set-image", image),
            ),
            PlannedCommand(
                name="run_workflow",
                description="Run a generated MatEnsemble workflow through the site CLI.",
                argv=("matensemble", "run", "workflow.py"),
            ),
        )
    if system == "linux":
        chosen_runtime = runtime or "docker"
        if chosen_runtime not in {"docker", "podman"}:
            raise ValueError("linux runtime must be 'docker' or 'podman'")
        return (
            PlannedCommand(
                name="pull_image",
                description=f"Pull the generic Linux image with {chosen_runtime}.",
                argv=(chosen_runtime, "pull", image),
            ),
            PlannedCommand(
                name="run_container",
                description="Open the container in the current workspace.",
                argv=(
                    chosen_runtime,
                    "run",
                    "--rm",
                    "-it",
                    "-v",
                    "$PWD:$PWD",
                    "-w",
                    "$PWD",
                    image,
                    "bash",
                ),
            ),
        )
    if system == "conda":
        return (
            PlannedCommand(
                name="create_conda_env",
                description="Create the MatEnsemble conda environment from environment.yaml.",
                argv=("conda", "env", "create", "-f", "environment.yaml"),
            ),
        )
    return (
        PlannedCommand(
            name="check_flux",
            description="Verify that Flux can see runtime resources.",
            argv=("flux", "resource", "list"),
        ),
    )


def _select_command(
    system: str,
    *,
    action: str,
    image: str,
    output_path: str | None,
    runtime: str | None,
) -> PlannedCommand:
    commands = _planned_commands(
        system,
        image=image,
        output_path=output_path,
        runtime=runtime,
    )
    if action == "auto":
        return commands[0]
    by_name = {command.name: command for command in commands}
    if action not in by_name:
        raise ValueError(f"unknown action {action!r}; expected one of {sorted(by_name)}")
    return by_name[action]


def _shell_quote(value: str) -> str:
    if value and all(ch.isalnum() or ch in "@%_+=:,./-$" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"
