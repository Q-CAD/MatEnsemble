[![PyPI version](https://badge.fury.io/py/matensemble.svg)](https://pypi.org/project/matensemble/)
[![Documentation](https://readthedocs.org/projects/matensemble/badge/?version=latest)](https://matensemble.readthedocs.io/en/latest/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

<p align="center">
  <img src="images/Logo-Matensemble.png" alt="MatEnsemble" width="720" />
</p>

# MatEnsemble

MatEnsemble is a Python library for **high-throughput workflows** on HPC systems. You define a directed acyclic graph (DAG) of tasks—**Python callables** or **executable commands**—and MatEnsemble submits work through **[Flux](https://flux-framework.readthedocs.io/)**, tracks completions, **adapts** scheduling to free CPUs and GPUs, and writes structured logs and per-job output directories.

An optional in-tree **dynopro** stack supports streaming dynamics and on-the-fly analysis for advanced materials simulation workflows.

## Features

- **DAG-based workflows** with dependencies via deferred return values (`OutputReference`)
- **Adaptive scheduling** that back-fills the allocation as tasks finish (with a non-adaptive mode when you need it)
- **Two job flavors**: Python jobs (remotely unpickled and executed by `matensemble.runtime_worker`) and argv-style **executable** jobs
- **Resource requests**: tasks, cores per task, GPUs per task, optional MPI (`pmi2`) via Flux
- **Observability**: `status.json`, `matensemble_workflow.log`, per-job `stdout` / `stderr`, pickle and JSON result artifacts; optional **web dashboard** (FastAPI on port 8000)

<p align="center">
  <img src="images/Cap_1_adaptive_task_management.png" alt="Adaptive task management" width="620" />
</p>

<p align="center">
  <img src="images/Cap_2_dynopro.png" alt="On-the-fly dynamics and analysis" width="620" />
</p>

## Documentation

Hosted documentation (overview, architecture, tutorials, API reference):

**[matensemble.readthedocs.io](https://matensemble.readthedocs.io/en/latest/)**

To build the docs locally:

```bash
uv sync --group dev
uv run sphinx-build -b html docs/source docs/build
```

Then open `docs/build/index.html` in a browser.

## Installation

### PyPI (Python layer)

Requires **Python 3.12+**. Flux **C libraries** and a running Flux session come from your HPC environment; the PyPI package adds the Python side.

```bash
pip install "matensemble[flux]"
```

The `flux` extra installs the Flux Python bindings (`flux-python`). For editing workflows only (no Flux on the machine):

```bash
pip install matensemble
```

### Containers (recommended on many clusters)

OCI images are published to GitHub Container Registry, for example:

`ghcr.io/freddude2004/matensemble:baseline-vX.Y.Z`

See the [container packages](https://github.com/FredDude2004/MatEnsemble/pkgs/container/matensemble) and the [Quick start](https://matensemble.readthedocs.io/en/latest/quickstart.html) in the docs for Apptainer/Singularity and site-specific notes.

### Development install

From a clone of this repository:

```bash
uv sync
uv sync --group dev   # optional: docs and pytest tooling
uv run pytest
```

Or with pip:

```bash
pip install -e ".[flux]"
```

Site-specific Conda-style environment files live under `scripts/` (for example `scripts/baseline/environment.yaml`, `scripts/frontier/`, `scripts/perlmuter/`). Align Python with **3.12+** and Flux with your center’s modules.

## Quick example

```python
from matensemble.pipeline import Pipeline

pipe = Pipeline()
pipe.exec(command=["/bin/echo", "hello from MatEnsemble"])
pipe.submit()
```

For Python jobs, dependency graphs, and the required split between an importable **job module** and a **runner script**, see the [Tutorials](https://matensemble.readthedocs.io/en/latest/tutorials.html).

## Examples in the repository

Illustrative workflows live under [`example_workflows/`](https://github.com/FredDude2004/MatEnsemble/tree/main/example_workflows).

## Requirements and runtime

- A **Flux allocation** (or equivalent) on the machine where you call `Pipeline.submit()`
- For MPI Python or executable jobs: a coherent MPI/Flux setup (e.g. PMI2) as expected by your site
- Optional: SSH port forwarding if you enable the dashboard on a compute node (see the architecture guide in the docs)

## Related links

- [Flux documentation](https://flux-framework.readthedocs.io/)
- [Flux Python guide](https://flux-framework.readthedocs.io/projects/flux-core/en/latest/guide/start.html)
- [Slurm documentation](https://slurm.schedmd.com/documentation.html) (common front-end to batch allocations)
- [LAMMPS manual](https://docs.lammps.org/Manual.html) (often used alongside ensemble MD workflows)

## Authors

Soumendu Bagchi, Kaleb Duchesneau (see `pyproject.toml` for contact details).

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).
