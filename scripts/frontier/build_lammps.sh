#!/bin/bash

# WORK IN PROGRESS -- this script is not ready for use in production. Please use the make-based build script.

# LAMMPS build script
# - Uses `cmake` to build a Kokkos-enabled binary of LAMMPS with the HIP backend
# - Contains support for GPU-aware MPI

# Caveats:
# - KSPACE: CMake build does not support hipFFT as of December 2023.
#   [PR #4007](https://github.com/lammps/lammps/pull/4007) will resolve this.

# Last modified: July 6, 2024

# Frontier has 3 PrgEnv (programming environments) available:
#   PrgEnv-cray -- HPE/Cray, clang-based
#   PrgEnv-amd -- AMD, clang-based, automatically includes setup of device libraries (no need to load amd-mixed/rocm)
#   PrgEnv-gnu -- GNU compiler toolchain
# All 3 PrgEnv's have `cc/CC/ftn` as the compiler wrapper around `C/C++/Fortran`.
# These wrappers automatically link cray-mpich and some other things, depending on loaded modules

# However, our LAMMPS build uses ``hipcc`` as the compiler, since that's what Kokkos currently requires, so we use PrgEnv-amd
# ``hipcc`` is making calls to ``amdclang++``, so it makes the most sense to load PrgEnv-amd
# module load PrgEnv-amd

# If we're using PrgEnv-cray or gnu, we need to explicitly load device libraries.
# `amd-mixed` is the vendor-provided equivalent of the `rocm` modules (which are maintained by OLCF).
# `amd-mixed` is compatible with Cray-PE (ie, cray-mpich, cray-libsci, etc), so this is preferred.
# `rocm` module is built by OLCF and is not guaranteed to be compatible with everything

# load the necessary modules
module load PrgEnv-amd/8.6.0
module load cray-mpich/8.1.31
module load amd/6.2.4 
module load rocm/6.2.4
module load libfabric/1.22.0

# The `cmake` module is needed for building with `cmake`
module load cmake

# FFTW3 for host-based FFT
module load cray-fftw

export MPICH_GPU_SUPPORT_ENABLED=1

# Check if environment path is provided as argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <conda_env_path>"
    exit 1
fi

CONDA_ENV_PATH=$1

# Activate the conda environment using the provided path
module load miniforge3
source  activate "$CONDA_ENV_PATH"
which python

# check the moadules loaded
echo "Loaded modules:"
module list

# Get current date in MonthName_YYYY format
CURRENT_DATE=$(date +%B_%Y)
LAMMPS_DIR="lammps_${CURRENT_DATE}"

# Clone LAMMPS if not exists with dated directory
if [ ! -d "./${LAMMPS_DIR}" ]; then
    git clone https://github.com/lammps/lammps.git "${LAMMPS_DIR}"
fi

cd "${LAMMPS_DIR}"

rm -rf ./build install # Remove any existing build directory
# Create a new build directory

mkdir build install

LMP_MACH=frontier_gfx90a
INSTDIR=${PWD}/install
# Explanation of compile flags:
#   -fdenormal-fp-math=ieee         -- specify to handle denormals in the same way that CUDA devices would
#   -fgpu-flush-denormals-to-zero   -- specify to handle denormals in the same way that CUDA devices would
#   -munsafe-fp-atomics             -- Tell the compiler to try to use hardware-based atomics on the GPU. Doesn't pose danger to correctness
#   -I${MPICH_DIR}/include          -- only necessary if not using Cray compiler wrappers
FLAGS='-fdenormal-fp-math=ieee -fgpu-flush-denormals-to-zero -munsafe-fp-atomics -I${MPICH_DIR}/include'

# Explanation of link flags:
#   -L${MPICH_DIR}/lib -lmpi        -- link cray-mpich
#   ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}      -- These environment variables are provided by cray-mpich.
#                                                                          They specify the library needed to use GPU-aware MPI.
#                                                                          Setting MPICH_GPU_SUPPORT_ENABLED=1 at run-time requires this library to be linked.
LINKFLAGS="-L${MPICH_DIR}/lib -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}"

# Optional: add a RPATH value to point to the sbcast --send-libs destination in /tmp
#           At scale, we recommend using `sbcast` to scatter the binary & it's libraries to node-local
#           storage on each node (ie, /tmp or /mnt/bb/$USER for NVME)
#           You can RPATH this directory ahead of time for ease of use
export HIPCC_LINK_FLAGS_APPEND="-Wl,-rpath,/tmp/lmp_${LMPMACH}_libs"
#export HIPCC_LINK_FLAGS_APPEND=""


# Install Rocm-aware MPI
MPICC="hipcc -L${MPICH_DIR}/lib -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a} -I${MPICH_DIR}/include" pip install --no-cache-dir --no-binary=mpi4py mpi4py

# cython is requried for packages ML-IAP
#
# conda install -c conda-forge cython

# start the build process
cd build

# FFT: this build uses KISS-FFT, the LAMMPS built-in. In order to use hipFFT, you must modify
#   lammps/cmake/Modules/Packages/KSPACE.cmake to allow HIPFFT as a valid FFT in `set(FFT_VALUES...)`
cmake \
       -DPKG_KOKKOS=on \
       -DPKG_MOLECULE=on \
       -DPKG_KSPACE=on \
       -DPKG_BODY=on \
       -DPKG_RIGID=on \
       -DPKG_MC=on \
       -DPKG_MANYBODY=on \
       -DPKG_REAXFF=on \
       -DPKG_REPLICA=on \
       -DPKG_QEQ=on \
       -DPKG_INTERLAYER=on \
       -DBUILD_SHARED_LIBS=yes \
       -DBUILD_MPI=yes \
       -DMLIAP_ENABLE_PYTHON=yes \
       -DPKG_PYTHON=yes \
       -DPKG_ML-SNAP=yes \
       -DPKG_ML-IAP=yes \
       -DPKG_ML-PACE=yes \
       -DPKG_SPIN=yes \
       -DPYTHON_EXECUTABLE:FILEPATH=`which python` \
       -DCMAKE_INSTALL_PREFIX=$INSTDIR \
       -DMPI_CXX_COMPILER=${ROCM_PATH}/bin/hipcc \
       -DCMAKE_CXX_COMPILER=${ROCM_PATH}/bin/hipcc \
       -DCMAKE_BUILD_TYPE=RelWithDebInfo \
       -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,${CRAY_LD_LIBRARY_PATH} ${CRAY_MPICH_DIR}/lib/libmpi_amd.so ${CRAY_MPICH_ROOTDIR}/gtl/lib/libmpi_gtl_hsa.so" \
       -DKokkos_ENABLE_HIP=on \
       -DFFT=FFTW3 \
       -DKokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS=ON \
       -DKokkos_ENABLE_SERIAL=on \
       -DBUILD_OMP=on \
       -DCMAKE_CXX_STANDARD=14 \
       -DKokkos_ARCH_VEGA90A=ON \
       -DCMAKE_CXX_FLAGS="${FLAGS}" \
       -DCMAKE_EXE_LINKER_FLAGS="${LINKFLAGS}" \
       -DLAMMPS_MACHINE=${LMPMACH} \
       ../cmake

make VERBOSE=1 -j 64
cp liblammps.* ../python/lammps
make install-python
