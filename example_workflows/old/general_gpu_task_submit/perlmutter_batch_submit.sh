#!/bin/bash
#SBATCH -A m5014_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32

. /global/cfs/cdirs/m526/sbagchi/spack/share/spack/setup-env.sh
spacktivate -p spack_matensemble_env
unset LUA_PATH LUA_CPATH
module load python/3.13
source activate /global/cfs/cdirs/m5014/conda_matensemble_env

srun flux start python example_matensemble_gpu.py
