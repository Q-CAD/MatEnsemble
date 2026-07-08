[![PyPI version](https://badge.fury.io/py/matensemble.svg)](https://pypi.org/project/matensemble/)
[![Documentation](https://readthedocs.org/projects/matensemble/badge/?version=latest)](https://matensemble.readthedocs.io/en/latest/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

<p align="center">
  <img src="media/Logo-Matensemble.png" alt="MatEnsemble" width="720" />
</p>

# MatEnsemble

MatEnsemble is a framework to build, orchestrate, and asynchronously manage extremely scalable adaptive-learning workflows, especially targeted for compute-intensive AI-driven high-throughput and ensemble-driven materials modeling simulations (e.g., atomistic modeling, Phase-Field, etc.) as efficiently as possible.

Here is an example which defines a simple MPI chore, adds ten independent instances of it to a pipeline, and submits the workflow.

```python
pipe = Pipeline()

# register a function the MatEnsemble
@pipe.chore(num_tasks=10, cores_per_task=1, gpus_per_task=0, mpi=True)
def mpi_hello_world():
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()

    print(f"Hello World! I am process {rank} of {size} on {name}.")

# adding 10 mpi_hello_world chores to the pipeline
for _ in range(10):
    mpi_hello_world()

pipe.submit(log_delay=1)
```

Rather than launching the MPI jobs directly, you describe the work declaratively with the @pipe.chore decorator and enqueue each invocation. MatEnsemble then schedules the chores through Flux, allocating the requested resources (num_tasks, CPU cores, GPUs, and MPI support) for each job.

## Installation

If you are on the OLCF Frontier or Pathfinder HPC systems or the NERSC Perlmutter system then you can
use our installation script to install MatEnsemble.

```bash
curl -fsSL https://raw.githubusercontent.com/Q-CAD/MatEnsemble/refs/heads/main/install.sh | bash
```

For general installation see our [documentation](https://matensemble.readthedocs.io/en/latest/)

---

While MatEnsemble can operate on a personal macOS or Linux workstation, orchestrating arbitrary Python callables and shell commands through explicit, resource- and dependency-aware execution graphs from a single Python workflow driver, it is primarily designed for the autonomous execution of large batches of user-defined, adaptively and hierarchically scheduled tasks on HPC systems, especially petascale and exascale platforms such as Perlmutter, Frontier, and Aurora.

## Minimal Code Example

MatEnsemble workflows are ordinary Python scripts (and/or shell commands) which can be use to: 1. define resource-aware chores, 2. pass chore outputs into later chores to create a DAG, and 3. add a strategy when the workflow should decide what to launch next while the campaign is already running.

```python
from matensemble.pipeline import Pipeline
from matensemble.model import Resources
from matensemble.chore import ChoreSpec

pipe = Pipeline()

md_resources = dict(num_tasks=128, cores_per_task=1, gpus_per_task=4, mpi=True)
analysis_resources = dict(num_tasks=1, cores_per_task=8)


@pipe.chore(name="simulate", **md_resources)
def simulate(candidate):
    # Run LAMMPS, DFT, phase-field, or another science application here.
    return {"trajectory": "traj.dump", "candidate": candidate}


@pipe.chore(name="score", **analysis_resources)
def score(simulation):
    # Analyze the completed simulation and propose the next high-value sample.
    return {
        "uncertainty": 0.18,
        "next_candidate": {"temperature": 1750, "composition": "SiO2"},
    }


@pipe.strategy(bolo_list=["score"], **analysis_resources)
def adapt(report):
    if report["uncertainty"] < 0.05:
        return None

    return ChoreSpec(
        args=(report["next_candidate"],),
        kwargs={},
        resources=Resources(**md_resources),
        qualname="simulate",
    )


seed = {"temperature": 1600, "composition": "SiO2"}
trajectory = simulate(seed)
score(trajectory)  # OutputReference creates the simulate -> score DAG edge.

future = pipe.submit(log_delay=10)
results = future.result()
```

## Publications

1. Bagchi, Soumendu, et al. "Towards “on-demand” van der Waals epitaxy with adaptive ensemble sampling atomistic workflows." Digital Discovery (2026) https://doi.org/10.1039/d6dd00049e.
2. Morelock, Ryan, et al. "pyRMG: A framework for high-throughput, large-cell DFT calculations on supercomputers." The Journal of Chemical Physics 164.5 (2026).
