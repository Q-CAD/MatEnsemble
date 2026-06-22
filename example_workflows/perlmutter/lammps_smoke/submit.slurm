#!/usr/bin/bash

#SBATCH -A nstaff
#SBATCH -C gpu              # Specifically request the 40gb GPUs; this is purely for throughput reasons
#SBATCH --qos debug         # Update as appropriate
#SBATCH -t 0:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1 # 1 Flux broker per node
#SBATCH --gpus-per-node=4   # 4 GPUs per node
#SBATCH --gpu-bind=closest

if ! command -v matensemble &>/dev/null; then
	echo "Install the MatEnsemble CLI for Perlmutter"
	echo "echo 'curl -fsSL https://raw.githubusercontent.com/FredDude2004/MatEnsemble/main/src/cli/install.sh | bash'"
	exit 1
fi

matensemble set-image ghcr.io/freddude2004/matensemble:perlmutter-v0.3.11
matensemble run workflow.py
