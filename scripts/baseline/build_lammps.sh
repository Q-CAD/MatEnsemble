#!/bin/bash
module load gcc
module load fftw
module load cmake

conda install cython

# Get current date in MonthName_YYYY format
CURRENT_DATE=$(date +%B_%Y)
LAMMPS_DIR="lammps_${CURRENT_DATE}"

# Clone LAMMPS if not exists with dated directory
if [ ! -d "./${LAMMPS_DIR}" ]; then
    git clone https://github.com/lammps/lammps.git "${LAMMPS_DIR}"
fi

cd "${LAMMPS_DIR}"

rm -rf ./build ./install

mkdir build install

cmake  -D CMAKE_BUILD_TYPE=Release \
            -D LAMMPS_EXCEPTIONS=ON \
            -D BUILD_SHARED_LIBS=ON \
            -D PKG_MANYBODY=ON -D PKG_MC=ON -D PKG_MOLECULE=ON -D PKG_KSPACE=ON -D PKG_REPLICA=ON -D PKG_ASPHERE=ON -D PKG_ML-SNAP=ON -D PKG_REAXFF=ON \
            -D PKG_RIGID=ON -D PKG_MPIIO=ON -D PKG_QEQ=ON -D PKG_PYTHON=On \
            -D PKG_INTERLAYER=ON -D CMAKE_POSITION_INDEPENDENT_CODE=ON -D CMAKE_EXE_FLAGS="-dynamic" ../cmake
make -j64
make install-python
