#!/bin/bash

# load the necessary modules
module load PrgEnv-amd/8.6.0
module load cray-mpich/8.1.31
module load amd/6.2.4 
module load rocm/6.2.4
module load libfabric/1.22.0

# Install Spack and build environment
git clone https://github.com/spack/spack.git
. spack/share/spack/setup-env.sh
export TMPDIR= # can set a custom to /tmp to avoid disk space issues
spack env create matensemble_spack_env spack.yaml
spack env activate matensemble_spack_env
# cat spack_packages.json | jq -r '.[] | .spec' | xargs -I {} spack add {}
spack install # This will take a while


# Create and activate conda environment
conda env create -f environment.yaml
conda activate matensemble_env

# Clone and build LAMMPS and mpi4py from source
chmod +x build_lammps.sh
./build_lammps.sh

# Install Python dependencies
pip install -r requirements.txt

cd ../../
# Install package in development mode
pip install -e .