[![PyPI version](https://badge.fury.io/py/matensemble.svg)](https://pypi.org/project/matensemble/)
[![Documentation](https://readthedocs.org/projects/matensemble/badge/?version=latest)](https://matensemble.readthedocs.io/en/latest/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

<p align="center">
  <img src="media/Logo-Matensemble.png" alt="MatEnsemble" width="720" />
</p>

# MatEnsemble

MatEnsemble orchestrates large batches of Flux-scheduled tasks on HPC systems: Python callables, shell commands, explicit resource requests, and dependency-aware execution graphs from a single Python driver process.

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
