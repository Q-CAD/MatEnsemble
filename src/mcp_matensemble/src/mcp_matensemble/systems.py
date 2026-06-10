from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemProfile:
    name: str
    title: str
    recommended_image: str
    container_runtime: str
    install_summary: str
    container_summary: tuple[str, ...]
    cli_install: str
    interactive_setup: str
    launch_command: str
    batch_notes: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "title": self.title,
            "recommended_image": self.recommended_image,
            "container_runtime": self.container_runtime,
            "install_summary": self.install_summary,
            "container_summary": list(self.container_summary),
            "cli_install": self.cli_install,
            "interactive_setup": self.interactive_setup,
            "launch_command": self.launch_command,
            "batch_notes": self.batch_notes,
        }


PROFILES: dict[str, SystemProfile] = {
    "generic_flux": SystemProfile(
        name="generic_flux",
        title="Portable Flux Workflows",
        recommended_image="none",
        container_runtime="none",
        install_summary=(
            "Use these site-independent workflow patterns in an environment where "
            "MatEnsemble, flux-core, flux-python, and the workflow's science "
            "dependencies are already importable."
        ),
        container_summary=(
            "No site-specific container is assumed for the workflow pattern.",
            "The same Python workflow structure can be launched on Frontier, Perlmutter, Pathfinder, Linux containers, or another Flux-capable runtime.",
        ),
        cli_install="No MatEnsemble site CLI is required for generic Flux.",
        interactive_setup="Start or enter a Flux allocation, then run `flux resource list`.",
        launch_command="python workflow.py",
        batch_notes="If using Slurm, start Flux inside the allocation before running Python.",
    ),
    "conda": SystemProfile(
        name="conda",
        title="Local Conda Environment",
        recommended_image="none",
        container_runtime="conda",
        install_summary=(
            "Use the repository environment.yaml for CPU/local development. This "
            "path is not intended to provide GPU-enabled LAMMPS container support."
        ),
        container_summary=(
            "No container is used.",
            "The environment.yaml installs MatEnsemble's Python dependencies without GPU support.",
        ),
        cli_install="No MatEnsemble site CLI is required for the conda path.",
        interactive_setup="conda env create -f environment.yaml\nconda activate matensemble",
        launch_command="python workflow.py",
        batch_notes="Use this for local script development, not production HPC GPU campaigns.",
    ),
    "linux": SystemProfile(
        name="linux",
        title="Generic Linux Container",
        recommended_image="ghcr.io/freddude2004/matensemble:linux-vX.Y.Z",
        container_runtime="Docker, Podman, or Apptainer",
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
    ),
    "frontier": SystemProfile(
        name="frontier",
        title="Frontier (OLCF)",
        recommended_image="ghcr.io/freddude2004/matensemble:frontier-vX.Y.Z",
        container_runtime="Apptainer",
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
            "curl -fsSL https://raw.githubusercontent.com/FredDude2004/MatEnsemble/main/src/cli/install.sh | bash\n"
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
            "`srun --external-launcher --mpi=pmi2 apptainer exec <image> flux start python workflow.py`."
        ),
    ),
    "pathfinder": SystemProfile(
        name="pathfinder",
        title="Pathfinder (OLCF)",
        recommended_image="ghcr.io/freddude2004/matensemble:pathfinder-vX.Y.Z",
        container_runtime="Apptainer",
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
            "curl -fsSL https://raw.githubusercontent.com/FredDude2004/MatEnsemble/main/src/cli/install.sh | bash\n"
            "matensemble set-image /path/to/matensemble.sif"
        ),
        interactive_setup=(
            "salloc -A <project_id> -t <HH:MM:SS> -N <nodes>\n"
            "apptainer build matensemble.sif docker://ghcr.io/freddude2004/matensemble:pathfinder-vX.Y.Z"
        ),
        launch_command="matensemble run workflow.py",
        batch_notes=(
            "Use the same high-level pattern as Frontier: Slurm allocation, Apptainer, Flux, then Python."
        ),
    ),
    "perlmutter": SystemProfile(
        name="perlmutter",
        title="Perlmutter (NERSC)",
        recommended_image="ghcr.io/freddude2004/matensemble:perlmutter-vX.Y.Z",
        container_runtime="Podman-HPC",
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
            "curl -fsSL https://raw.githubusercontent.com/FredDude2004/MatEnsemble/main/src/cli/install.sh | bash\n"
            "podman-hpc pull ghcr.io/freddude2004/matensemble:perlmutter-vX.Y.Z\n"
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
            "NVIDIA, CXI, and library bind settings."
        ),
    ),
}


def normalize_system(name: str | None) -> str:
    if not name:
        return "generic_flux"
    key = name.strip().lower().replace("-", "_")
    aliases = {
        "generic": "generic_flux",
        "flux": "generic_flux",
        "local": "conda",
        "local_conda": "conda",
        "devcontainer": "linux",
        "docker": "linux",
        "nersc": "perlmutter",
        "olcf_frontier": "frontier",
        "olcf_pathfinder": "pathfinder",
    }
    return aliases.get(key, key)


def list_systems() -> list[dict[str, object]]:
    return [profile.to_dict() for profile in PROFILES.values()]


def get_system_profile(name: str | None) -> SystemProfile:
    key = normalize_system(name)
    if key not in PROFILES:
        raise ValueError(f"unknown system {name!r}; expected one of {sorted(PROFILES)}")
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
