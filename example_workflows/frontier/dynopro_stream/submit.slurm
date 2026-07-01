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

total_nodes="${SLURM_JOB_NUM_NODES:-1}"
worker_nodes="$((total_nodes - 1))"
if ((worker_nodes < 1)); then
	worker_nodes=1
fi

export MATENSEMBLE_DYNOPRO_NNODES="${MATENSEMBLE_DYNOPRO_NNODES:-$worker_nodes}"
export MATENSEMBLE_GPUS_PER_NODE="${MATENSEMBLE_GPUS_PER_NODE:-8}"
export MATENSEMBLE_CORES_PER_NODE="${MATENSEMBLE_CORES_PER_NODE:-64}"

echo "SLURM nodes: ${total_nodes}"
echo "Dynopro worker nodes: ${MATENSEMBLE_DYNOPRO_NNODES}"

matensemble set-image ./matensemble.sif
matensemble run workflow.py
