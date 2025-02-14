# MatEnsemble

An adaptive and highly asynchronous ensemble simulation workflow manager

## Installation Guide

### Prerequisites
- Anaconda or Miniconda
- Git
- (optional) C++ compiler (gcc/clang) (for online MD using lammps, mg etc.)
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

# Create and activate conda environment:
```bash
conda env create -f environment.yaml
conda activate matensemble
```

# Install Python dependencies
```bash
pip install -r requirements.txt
```
# Install package in development mode
```bash
pip install -e .
```

