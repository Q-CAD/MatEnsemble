[![PyPI version](https://badge.fury.io/py/matensemble.svg)](https://badge.fury.io/py/matensemble)
[![Documentation Status](https://readthedocs.org/projects/matensemble/badge/?version=latest)](https://matensemble.readthedocs.io/en/latest/?badge=latest)
<!-- [![Build Status](https://github.com/username/matensemble/workflows/Build/badge.svg)](https://github.com/username/matensemble/actions) -->
[![Coverage Status](https://coveralls.io/repos/github/username/matensemble/badge.svg?branch=main)](https://coveralls.io/github/username/matensemble?branch=main)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

 <img src="images/Logo-Matensemble.png" alt="MatEnsemble Logo" width="800"/>
An adaptive and highly asynchronous ensemble simulation workflow manager
 with in-memory dynamics (GPU) and on-the-fly analysis (CPU) capabilties.

## Core Capabiliies:
- Adaptive task management (with the ability to integrate with autonmous data acquistion algorirthms)
 <img src="images/Cap_1_adaptive_task_management.png" alt="Adaptive-task-management" width="600"/>

- On-the-fly streaming of materials dynamics with custom analysis algorithms
 <img src="images/Cap_2_dynopro.png" alt="on-the-fly dynamics" width="600"/>

## User Guide:
Open the documentation
```sh
<browser-of-choice> docs/index.html
```


## Installation Guide

### Prerequisites
- Anaconda or Miniconda
- Git
- (optional) C++ compiler (gcc/clang) (for online dynamics using lammps, rmg etc.)
- (optional) CMake

### Quick Install (linux systems)
```bash
git clone https://github.com/Q-CAD/MatEnsemble.git
cd matensemble
chmod +x install.sh
./install.sh <path-to-env>
```
<!-- ### Customized Install
Environment Setup (`environment.yaml`)

```bash
# filepath: environment.yaml
name: matensemble
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - flux-core>=0.48.0
  - flux-sched>=0.25.0
  - cmake>=3.20
  - gcc>=11.0
  - make
  ``` -->

## Create and activate conda environment:
```bash
conda env create -f environment.yaml --prefix <path-to-env>
conda activate <path-to-env>
```

### Install Python dependencies
```bash
pip install -r requirements.txt
```
### Install package in development mode
```bash
pip install -e .
```

**Links to documentation for Ease of Access: **
* [MatEnsemble Dependency](https://github.com/BagchiS6/AutoPF)
* [Slurm Documentation](https://slurm.schedmd.com/documentation.html)
* [Flux Documentation](https://flux-framework.readthedocs.io/en/latest/guides/learning_guide.html)
* [Python Flux Guide](https://flux-framework.readthedocs.io/projects/flux-core/en/latest/guide/start.html)
* [Baseline Super-Computer User Guide](https://docs.cades.olcf.ornl.gov/baseline_user_guide/baseline_user_guide.html)
* [Frontier Super-Computer User Guide](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#slurm)
* [Open MPI Documentation (Message Passing Interface)](https://www.open-mpi.org/)
* [LAMMPS Documentation](https://docs.lammps.org/Manual.html)



