#!/bin/bash
#
# Change to your account
# Also change in the srun command below
#SBATCH -A CPH162
#
# Job naming stuff
#SBATCH -J rmgmp-541837
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Requested time
#SBATCH -t 02:00:00
#
# Requested queue
#SBATCH -p batch
#
# Number of frontier nodes to use.
# Set the same value in the SBATCH line and NNODES
#SBATCH -N 20
#
# OMP num threads. Frontier reserves 8 of 64 cores on a node
# for the system. There are 8 logical GPUs per node so we use
# 8 MPI tasks/node with 7 OMP threads per node
export OMP_NUM_THREADS=7
#
# RMG threads. Max of 7 same as for OMP_NUM_THREADS but in some
# cases running with fewer may yield better performance because
# of cache effects.
export RMG_NUM_THREADS=5
#
# Don't change these
export MPICH_OFI_NIC_POLICY=NUMA
export MPICH_GPU_SUPPORT_ENABLED=0
#
# Load modules

module load PrgEnv-gnu
module load cmake
module load bzip2
module load boost
module load craype-x86-milan
module load cray-fftw
module load cray-hdf5-parallel
module load craype-accel-amd-gfx90a
module load rocm
module load libfabric

#---------------------- SETUP FOR MATENSEMBLE IN FRONTIER -------------------------------------------------------------------
. /lustre/orion/mat201/world-shared/QCAD/Mar_4_2025/spack/share/spack/setup-env.sh
which spack
spack env activate matensemble_spack_env
spack load flux-sched
which flux
export PYTHONPATH=$PYTHONPATH:/lustre/orion/mat201/world-shared/QCAD/python_envs/matensemble_02252025/lib/python3.11/site-ackages

module load miniforge3
source activate /lustre/orion/mat201/world-shared/QCAD/python_envs/matensemble_02252025

# just in case there are python conflicts due to spack and conda
CONDA_PYTHON_EXE=/lustre/orion/mat201/world-shared/QCAD/python_envs/matensemble_02252025/bin/python
echo $CONDA_PYTHON_EXE

srun -N $SLURM_NNODES -n $SLURM_NNODES --external-launcher --mpi=pmi2 --gpu-bind=closest flux start $CONDA_PYTHON_EXE example_matensemble.py
