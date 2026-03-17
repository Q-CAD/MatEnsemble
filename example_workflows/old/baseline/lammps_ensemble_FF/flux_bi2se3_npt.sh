#!/bin/bash

#SBATCH -A mat269
#SBATCH -p batch
#SBATCH -J lmp_TEST
#SBATCH -N 19
#SBATCH -t 01:00:00

module load gcc
module load fftw
module load python
module load openmpi  # Ensure correct MPI module is loaded

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda deactivate
source activate /autofs/nccsopen-svm1_proj/mat269/baseline_matensembe_env 

export OMP_NUM_THREADS=1

# Fix potential MPI communication issues
export OMPI_MCA_btl="self,tcp"  # Disable shared memory transport (sm, smcuda)

srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} flux start python lammps_flux_for_kernel.py -ffd Bi2Se3_runs/ -id flux_inputs/npt_Bi2Se3_inputs/

