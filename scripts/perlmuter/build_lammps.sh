#!/bin/bash
module load PrgEnv-gnu/8.5.0
module load cudatoolkit/12.4
module load craype-accel-nvidia80
module load python/3.13

export MPICH_GPU_SUPPORT_ENABLED=1

# Check if environment path is provided as argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <conda_env_path>"
    exit 1
fi

CONDA_ENV_PATH=$1

source activate $CONDA_ENV_PATH
MPICC="cc -shared" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py==4.0.3

git clone -b patch_2Apr2025 https://github.com/lammps/lammps.git
cd lammps
mkdir build
cd build

# have to make 'cmake' available :)
module load cmake

cmake -DCMAKE_INSTALL_PREFIX=$PWD/../install_pm -D CMAKE_BUILD_TYPE=Release \
            -D CMAKE_Fortran_COMPILER=ftn -D CMAKE_C_COMPILER=cc -D CMAKE_CXX_COMPILER=CC \
            -D MPI_C_COMPILER=cc -D MPI_CXX_COMPILER=CC -D LAMMPS_EXCEPTIONS=ON \
            -D BUILD_SHARED_LIBS=ON -D PKG_KOKKOS=yes -D Kokkos_ARCH_AMPERE80=ON -D Kokkos_ENABLE_CUDA=yes \
            -D PKG_MOLECULE=on \
            -D PKG_BODY=on \
            -D PKG_RIGID=on \
            -D PKG_MC=on \
            -D PKG_MANYBODY=on \
            -D PKG_REAXFF=on \
            -D PKG_REPLICA=on \
            -D PKG_QEQ=on \
            -D PKG_INTERLAYER=on \
            -D MLIAP_ENABLE_PYTHON=yes \
            -D PKG_PYTHON=yes \
            -D PKG_ML-SNAP=yes \
            -D PKG_ML-IAP=yes \
            -D PKG_ML-PACE=yes \
            -D PKG_SPIN=yes \
            -D CMAKE_POSITION_INDEPENDENT_CODE=ON -D CMAKE_EXE_FLAGS="-dynamic" ../cmake
make -j64
make install-python
