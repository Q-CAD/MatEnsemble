#!/bin/bash

#SBATCH -A mat269
#SBATCH -p batch
#SBATCH -J matensemble_BO
#SBATCH -N 76
#SBATCH -t 24:00:00

module purge
module load gcc/12.2.0 python  fftw/3.3.10 openmpi/4.0.4
source activate /gpfs/wolf2/cades/mat269/world-shared/test_MatEnsemble_build/matensemble_env  
which python
export OMP_NUM_THREADS=1
export OMPI_MCA_btl=self,vader,tcp

srun --mpi=pmi2 -N ${SLURM_NNODES} -n ${SLURM_NNODES} flux start python BO_online.py input_paramters.json
