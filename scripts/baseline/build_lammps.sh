#!/bin/bash

# This script builds LAMMPS from source with specific packages and configurations.
# It assumes that the user has a working installation of conda and the necessary modules loaded.
# Usage: ./build_lammps.sh
# Load necessary modules
# module load python
module load gcc
module load fftw
module load cmake

echo "Starting to Build LAMMPS at this point"
which python
# source activate "$ENV_PATH"
# Avoid conflicts for mpi4py installation
#MPICC="mpicc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py

# Install Cython using conda
# This is necessary for building LAMMPS with Python support
#conda install cython

# Get current date in MonthName_YYYY format
CURRENT_DATE=$(date +%B_%Y)
LAMMPS_DIR="lammps_${CURRENT_DATE}"

# Clone LAMMPS if not exists with dated directory
if [ ! -d "./${LAMMPS_DIR}" ]; then
    git clone https://github.com/lammps/lammps.git "${LAMMPS_DIR}"
fi

cd "${LAMMPS_DIR}"

rm -rf ./build # Remove any existing build directory
# Create a new build directory
mkdir build 
cd build

cmake  -D CMAKE_BUILD_TYPE=Release \
            -D LAMMPS_EXCEPTIONS=ON \
            -D BUILD_SHARED_LIBS=ON \
            -D PKG_MANYBODY=ON -D PKG_MC=ON -D PKG_MOLECULE=ON -D PKG_KSPACE=ON -D PKG_REPLICA=ON -D PKG_ASPHERE=ON -D PKG_ML-SNAP=ON -D PKG_REAXFF=ON \
            -D PKG_RIGID=ON -D PKG_MPIIO=ON -D PKG_QEQ=ON -D PKG_PYTHON=On \
            -D PKG_INTERLAYER=ON -D CMAKE_POSITION_INDEPENDENT_CODE=ON -D CMAKE_EXE_FLAGS="-dynamic" ../cmake
make -j64
make install-python
