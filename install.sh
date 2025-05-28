#!/bin/bash

# Create and activate conda environment
conda env create -f environment.yaml --prefix $1
source activate $1

# Clone and build LAMMPS
git clone https://github.com/lammps/lammps.git
cd lammps
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

# Install Ovito (with -y flag to automatically accept prompts)
conda install -y --strict-channel-priority -c https://conda.ovito.org -c conda-forge ovito=3.12.1

# return to the root directory
cd ../../
# Install Python dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .