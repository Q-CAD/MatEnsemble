#!/bin/bash

# Create and activate conda environment
conda env create -f environment.yaml
conda activate matensemble

# Clone and build LAMMPS
git clone https://github.com/lammps/lammps.git
cd lammps
mkdir build
cd build
cmake ../cmake -D BUILD_MPI=yes -D PKG_MANYBODY=yes -D PKG_MOLECULE=yes
make -j4
make install

# Install Python dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .