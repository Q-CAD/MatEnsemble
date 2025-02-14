


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
 <img src="images/Cap_1_adaptive_task_management.png" alt="Adaptive-task-management" width="400"/>
- On-the-fly streaming of materials dynamics with custom analysis algorithms
 <img src="images/Cap_2_dynopro.png" alt="Adaptive-task-management" width="400"/>

## Installation Guide

### Prerequisites
- Anaconda or Miniconda
- Git
- (optional) C++ compiler (gcc/clang) (for online dynamics using lammps, rmg etc.)
- (optional) CMake

### Quick Install (linux systems)
```bash
git clone https://github.com/your-username/matensemble.git
cd matensemble
chmod +x install.sh
./install.sh
```
### Customized Install
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
  ```

## Create and activate conda environment:
```bash
conda env create -f environment.yaml
conda activate matensemble
```

### Install Python dependencies
```bash
pip install -r requirements.txt
```
### Install package in development mode
```bash
pip install -e .
```

