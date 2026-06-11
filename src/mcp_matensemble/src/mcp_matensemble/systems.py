from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_SYSTEMS = ("frontier", "perlmutter", "pathfinder", "linux")


class UnsupportedSystemError(ValueError):
    def __init__(self, system: str | None):
        self.system = system
        super().__init__(f"Unsupported system: {system}")

    def to_error(self) -> dict[str, object]:
        return {
            "ok": False,
            "error_code": "UNSUPPORTED_SYSTEM",
            "message": f"Unsupported system: {self.system}",
            "details": {"supported_systems": list(SUPPORTED_SYSTEMS)},
        }


@dataclass(frozen=True)
class SystemProfile:
    name: str
    title: str
    recommended_image: str
    container_runtime: str
    container_backends: tuple[str, ...]
    install_summary: str
    container_summary: tuple[str, ...]
    cli_install: str
    interactive_setup: str
    launch_command: str
    batch_notes: str
    external_docs: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "title": self.title,
            "recommended_image": self.recommended_image,
            "container_runtime": self.container_runtime,
            "container_backends": list(self.container_backends),
            "install_summary": self.install_summary,
            "container_summary": list(self.container_summary),
            "cli_install": self.cli_install,
            "interactive_setup": self.interactive_setup,
            "launch_command": self.launch_command,
            "batch_notes": self.batch_notes,
            "external_docs": list(self.external_docs),
            "limitations": list(self.limitations),
        }


PROFILES: dict[str, SystemProfile] = {
    "linux": SystemProfile(
        name="linux",
        title="Generic Linux Container",
        recommended_image="ghcr.io/freddude2004/matensemble:linux-vX.Y.Z",
        container_runtime="Docker",
        container_backends=("docker",),
        install_summary=(
            "Use the generic Linux MatEnsemble image for local/containerized development "
            "where site-specific GPU launch wrappers are not needed."
        ),
        container_summary=(
            "Linux base image is Ubuntu 24.04 based.",
            "Includes Python 3.12 via uv, MPICH/mpi4py, Flux, Fluxion, LAMMPS, Jupyter, and MatEnsemble.",
            "This is the best default container context for non-Frontier, non-Perlmutter development.",
        ),
        cli_install="No site CLI is required for generic Linux container workflows.",
        interactive_setup=(
            'docker run --rm -it -v "$PWD:$PWD" -w "$PWD" '
            "ghcr.io/freddude2004/matensemble:linux-vX.Y.Z bash"
        ),
        launch_command="flux start --test-size=4 python workflow.py",
        batch_notes=(
            "For local container tests, start a small Flux instance with `flux start --test-size=<n>`."
        ),
        external_docs=("https://flux-framework.readthedocs.io/",),
        limitations=(
            "Local multi-node simulation uses flux start --test-size and may not mirror site MPI bindings.",
        ),
    ),
    "frontier": SystemProfile(
        name="frontier",
        title="Frontier (OLCF)",
        recommended_image="ghcr.io/freddude2004/matensemble:frontier-vX.Y.Z",
        container_runtime="Apptainer or Podman-HPC",
        container_backends=("apptainer", "podman-hpc"),
        install_summary=(
            "Build or pull a Frontier MatEnsemble Apptainer image, allocate nodes "
            "with Slurm, then use the MatEnsemble CLI to run scripts through Flux."
        ),
        container_summary=(
            "Frontier image is based on the OLCF CPE Ubuntu stack.",
            "Includes Cray MPICH-aware mpi4py, Flux, Fluxion, LAMMPS, OVITO, and MatEnsemble.",
            "LAMMPS is built for Frontier AMD GPUs with Kokkos/HIP support.",
            "Use Apptainer `.sif` or sandbox images on Frontier.",
        ),
        cli_install=(
            "uv run --package mcp-matensemble matensemble-agent-install --system frontier\n"
            "Use prepare_container_pull_plan(system='frontier') to resolve and pull the local-version GHCR image.\n"
            "matensemble set-image /path/to/matensemble.sif"
        ),
        interactive_setup=(
            "salloc -A <project_id> -t <HH:MM:SS> -N <nodes>\n"
            "apptainer build matensemble.sif docker://ghcr.io/freddude2004/matensemble:frontier-vX.Y.Z\n"
            "matensemble set-image /path/to/matensemble.sif"
        ),
        launch_command="matensemble run workflow.py",
        batch_notes=(
            "Run inside an existing Slurm allocation. The Frontier CLI expands to "
            "`srun --external-launcher --mpi=pmi2 apptainer exec <image> flux start python workflow.py`. "
            "Request at least 2 nodes because Flux uses one node as a broker/orchestrator."
        ),
        external_docs=(
            "https://docs.olcf.ornl.gov/systems/frontier_user_guide.html",
            "https://docs.olcf.ornl.gov/software/containers_on_frontier.html",
        ),
    ),
    "pathfinder": SystemProfile(
        name="pathfinder",
        title="Pathfinder (OLCF)",
        recommended_image="ghcr.io/freddude2004/matensemble:pathfinder-vX.Y.Z",
        container_runtime="Apptainer or Podman-HPC",
        container_backends=("apptainer", "podman-hpc"),
        install_summary=(
            "Build or pull a Pathfinder MatEnsemble Apptainer image, allocate nodes "
            "with Slurm, then run the workflow inside Flux."
        ),
        container_summary=(
            "Pathfinder image is Ubuntu-based.",
            "Includes Python 3.12 via uv, OpenMPI/mpi4py, Flux, Fluxion, LAMMPS, and MatEnsemble.",
            "Use Apptainer `.sif` or sandbox images on Pathfinder.",
        ),
        cli_install=(
            "uv run --package mcp-matensemble matensemble-agent-install --system pathfinder\n"
            "Use prepare_container_pull_plan(system='pathfinder') to resolve and pull the local-version GHCR image.\n"
            "matensemble set-image /path/to/matensemble.sif"
        ),
        interactive_setup=(
            "salloc -A <project_id> -t <HH:MM:SS> -N <nodes>\n"
            "apptainer build matensemble.sif docker://ghcr.io/freddude2004/matensemble:pathfinder-vX.Y.Z"
        ),
        launch_command="matensemble run workflow.py",
        batch_notes=(
            "Use the same high-level pattern as Frontier: Slurm allocation, Apptainer, Flux, then Python. "
            "Request at least 2 nodes because Flux uses one node as a broker/orchestrator."
        ),
        external_docs=("https://docs.olcf.ornl.gov/",),
    ),
    "perlmutter": SystemProfile(
        name="perlmutter",
        title="Perlmutter (NERSC)",
        recommended_image="ghcr.io/freddude2004/matensemble:perlmutter-vX.Y.Z",
        container_runtime="Podman-HPC or Apptainer",
        container_backends=("podman-hpc", "apptainer"),
        install_summary=(
            "Use Podman-HPC with the MatEnsemble Perlmutter CLI. The CLI hides "
            "the Flux resource config, Slurm/NVIDIA/PMI bind mounts, and container launch details."
        ),
        container_summary=(
            "Perlmutter image is based on NERSC GPU/Flux container images.",
            "Includes Flux, CUDA-oriented LAMMPS, mpi4py, MACE/Torch-related packages, and MatEnsemble.",
            "The MatEnsemble CLI is strongly preferred because Perlmutter Podman-HPC launch details are verbose.",
        ),
        cli_install=(
            "uv run --package mcp-matensemble matensemble-agent-install --system perlmutter\n"
            "Use prepare_container_pull_plan(system='perlmutter') to resolve and pull "
            "ghcr.io/freddude2004/matensemble:perlmutter-vX.Y.Z from the local MatEnsemble version.\n"
            "matensemble set-image ghcr.io/freddude2004/matensemble:perlmutter-vX.Y.Z"
        ),
        interactive_setup=(
            "salloc -A <account_id> -C gpu --qos=debug -t <HH:MM:SS> -N <nodes> "
            "--ntasks-per-node=1 --gpus-per-node=4 --gpu-bind=closest"
        ),
        launch_command="matensemble run workflow.py",
        batch_notes=(
            "Run inside an existing Slurm allocation. The Perlmutter CLI writes a temporary "
            "Flux resource config to SCRATCH and launches Podman-HPC with the required PMI, Slurm, "
            "NVIDIA, CXI, and library bind settings. Request at least 2 nodes because Flux uses "
            "one node as a broker/orchestrator."
        ),
        external_docs=(
            "https://docs.nersc.gov/systems/perlmutter/",
            "https://docs.nersc.gov/development/containers/podman-hpc/",
        ),
        limitations=(
            "Perlmutter launch details are best handled through the site CLI until direct Flux resource config support is expanded.",
        ),
    ),
}


def normalize_system(name: str | None) -> str:
    if not name:
        raise UnsupportedSystemError(name)
    key = name.strip().lower().replace("-", "_")
    aliases = {
        "generic": "linux",
        "generic_flux": "linux",
        "flux": "linux",
        "local": "linux",
        "local_conda": "linux",
        "devcontainer": "linux",
        "docker": "linux",
        "nersc": "perlmutter",
        "olcf_frontier": "frontier",
        "olcf_pathfinder": "pathfinder",
    }
    normalized = aliases.get(key, key)
    if normalized not in SUPPORTED_SYSTEMS:
        raise UnsupportedSystemError(name)
    return normalized


def list_systems() -> list[dict[str, object]]:
    return [PROFILES[name].to_dict() for name in SUPPORTED_SYSTEMS]


def get_system_profile(name: str | None) -> SystemProfile:
    key = normalize_system(name)
    return PROFILES[key]


def render_environment_setup(name: str | None) -> str:
    profile = get_system_profile(name)
    return f"""\
# {profile.title}

Recommended image: `{profile.recommended_image}`
Runtime: `{profile.container_runtime}`

## Summary

{profile.install_summary}

## Container Contents

{_bullet_lines(profile.container_summary)}

## Install/Configure

```bash
{profile.cli_install}
```

## Allocation/Runtime Setup

```bash
{profile.interactive_setup}
```

## Run A Generated Workflow

```bash
{profile.launch_command}
```

## Notes

{profile.batch_notes}
"""


def _bullet_lines(items: tuple[str, ...]) -> str:
    return "\n".join(f"- {item}" for item in items)
