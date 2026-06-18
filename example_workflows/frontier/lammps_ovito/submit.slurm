#!/usr/bin/bash

 #SBATCH -A lrn090
 #SBATCH -J post_matsemble
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

matensemble set-image ./matensemble.sif
matensemble run workflow.py
