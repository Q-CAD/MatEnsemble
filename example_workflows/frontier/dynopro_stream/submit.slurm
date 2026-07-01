#!/usr/bin/bash

#SBATCH -A lrn090
#SBATCH -J me_dynopro_stream
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 0:05:00
#SBATCH -N 2
#SBATCH -C nvme

if ! command -v matensemble &>/dev/null; then
	echo "Install the MatEnsemble CLI for Frontier"
	echo "curl -fsSL https://raw.githubusercontent.com/FredDude2004/MatEnsemble/main/src/cli/install.sh | bash"
	exit 1
fi

module reset
module load olcf-container-tools
module load apptainer-enable-mpi apptainer-enable-gpu

export MATENSEMBLE_DYNOPRO_NNODES="${SLURM_JOB_NUM_NODES:-1}"
export MATENSEMBLE_GPUS_PER_NODE="${MATENSEMBLE_GPUS_PER_NODE:-8}"
export MATENSEMBLE_CORES_PER_NODE="${MATENSEMBLE_CORES_PER_NODE:-64}"

matensemble set-image ./matensemble.sif
matensemble run workflow.py
