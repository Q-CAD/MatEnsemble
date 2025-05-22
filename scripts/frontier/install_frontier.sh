#!/bin/bash

# load the necessary modules
module load PrgEnv-amd/8.6.0
module load cray-mpich/8.1.31
module load amd/6.2.4 
module load rocm/6.2.4
module load libfabric/1.22.0
module load miniforge3

# Install Spack and build environment
# git clone https://github.com/spack/spack.git
# . spack/share/spack/setup-env.sh
export TMPDIR= # can set a custom to /tmp to avoid disk space issues
# spack env create matensemble_spack_env spack.yaml
# spack env activate matensemble_spack_env
# # cat spack_packages.json | jq -r '.[] | .spec' | xargs -I {} spack add {}
# spack install # This will take a while


# Load an existing Spack environment for TEAM MAT201

# spack env activate matensemble_spack_env

# HAVE TO FIGURE HOW TO BE IN THE RIGHT CONDA ENVIRONMENT EVEN AFTER CREATING AND LOADING A SPACK ENVIRONMENT
# THIS SEEMS NONTRIVIAL. ANY SUGGESTIONS?
# UNTIL THEN WE WILL ASSUME THAT USER IN FRONTIER HAS ACCESS TO THE SPACK ENV AT /lustre/orion/mat201/world-shared/QCAD/Mar_4_2025/spack/
# AND LOAD THE SPACK ENVIRONMENT USING THE FOLLOWING COMMANDS
# #-------------- setting up spack and conda for matEnsemble------------------------
# . /lustre/orion/mat201/world-shared/QCAD/Mar_4_2025/spack/share/spack/setup-env.sh
# which spack
# spack env activate matensemble_spack_env
# spack load flux-sched
# which flux
#------------------------------------------------------

# Check if environment path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/conda/environment"
    echo "Example: $0 ~/envs/matensemble_env"
    exit 1
fi

ENV_PATH="$1"

# Check if directory exists
if [ -d "$ENV_PATH" ]; then
    read -p "Directory $ENV_PATH already exists. Do you want to remove it and continue? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY]) 
            echo "Removing existing environment..."
            rm -rf "$ENV_PATH"
            ;;
        *)
            echo "Installation cancelled"
            exit 1
            ;;
    esac
fi

# Create and activate conda environment with custom path
echo "Creating conda environment at $ENV_PATH..."
conda env create -f environment.yaml --prefix "$ENV_PATH"

if [ $? -ne 0 ]; then
    echo "Failed to create conda environment"
    exit 1
fi

source activate "$ENV_PATH"

which python
# Check if conda environment is activated
if [ -z "$CONDA_PREFIX" ]; then
    echo "Failed to activate conda environment"
    exit 1
fi
echo "Activated conda environment at $CONDA_PREFIX"

# install ovito
conda install --strict-channel-priority -c https://conda.ovito.org -c conda-forge ovito=3.12.2

# Clone and build LAMMPS and mpi4py from source
chmod +x build_lammps.sh
./build_lammps.sh "$ENV_PATH"

# Install Python dependencies
pip install -r requirements.txt

cd ../../
# Install package in development mode
pip install -e .