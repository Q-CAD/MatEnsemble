#!/usr/bin/bash

#SBATCH -A m5064_g
#SBATCH -C gpu
#SBATCH --qos interactive
#SBATCH -t 4:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

if ! command -v matensemble &>/dev/null; then
	echo "Install the MatEnsemble CLI for Perlmutter"
	echo "curl -fsSL https://raw.githubusercontent.com/FredDude2004/MatEnsemble/main/src/cli/install.sh | bash"
	exit 1
fi

matensemble set-image ghcr.io/freddude2004/matensemble:perlmutter-v0.3.11
matensemble run recryst_mace_matensemble_dag.py
