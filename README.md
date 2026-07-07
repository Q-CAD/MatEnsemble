[![PyPI version](https://badge.fury.io/py/matensemble.svg)](https://pypi.org/project/matensemble/)
[![Documentation](https://readthedocs.org/projects/matensemble/badge/?version=latest)](https://matensemble.readthedocs.io/en/latest/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

<p align="center">
  <img src="media/Logo-Matensemble.png" alt="MatEnsemble" width="720" />
</p>

# MatEnsemble

MatEnsemble is a framework to build, orchestrate, and asynchronously manage extremely scalable adaptive-learning workflows, especially targeted for compute-intensive AI-driven high-throughput and ensemble-driven materials modeling simulations (e.g., atomistic modeling, Phase-Field, etc.) as efficiently as possible. 

While it can in general run on your personal Mac/Linux workstation and orchestrate any python callable, shell commands with explicit resource and dependency-aware execution graphs from a single python workflow driver process,  MatEnsemble shines with ***user-defined aut0nomous stretegic*** execution of large batches of adaptively and hierarchically-scheduled tasks on HPC systems, often on Peta and Exascale computing facilities, e.g., Perlmutter, Frontier, Aurora etc.

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
