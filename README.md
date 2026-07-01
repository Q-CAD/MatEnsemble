[![PyPI version](https://badge.fury.io/py/matensemble.svg)](https://pypi.org/project/matensemble/)
[![Documentation](https://readthedocs.org/projects/matensemble/badge/?version=latest)](https://matensemble.readthedocs.io/en/latest/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

<p align="center">
  <img src="media/Logo-Matensemble.png" alt="MatEnsemble" width="720" />
</p>

# MatEnsemble

MatEnsemble is a Python library for **high-throughput workflows** on HPC systems. You define a directed acyclic graph (DAG) of chores—**Python callables** or **executable commands**—and MatEnsemble submits work through **[Flux](https://flux-framework.readthedocs.io/)**, tracks completions, **adapts** scheduling to free CPUs and GPUs, and writes structured logs and per-chore output directories.

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

If you are trying to use MatEnsemble on the Frontier, Pathfinder or Perlmutter super
computers then you can quickly install MatEnsemble with our install script:

```bash
curl -fsSL https://raw.githubusercontent.com/FredDude2004/MatEnsemble/refs/heads/main/install.sh | bash
```

For more general installation see our [documentation](https://matensemble.readthedocs.io/en/latest/)

## Related

- [Flux documentation](https://flux-framework.readthedocs.io/)
- [Flux Python guide](https://flux-framework.readthedocs.io/projects/flux-core/en/latest/guide/start.html)
- [Slurm documentation](https://slurm.schedmd.com/documentation.html) (common front-end to batch allocations)
- [LAMMPS manual](https://docs.lammps.org/Manual.html) (often used alongside ensemble MD workflows)

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).
