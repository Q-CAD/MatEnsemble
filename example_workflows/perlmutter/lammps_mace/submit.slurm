#!/usr/bin/bash

#SBATCH -A m5014_g
#SBATCH -J lammps_ovito_coordination_perlmu
#SBATCH -o launch_logs/%x-%j.out
#SBATCH -e launch_logs/%x-%j.err
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --qos debug
#SBATCH -C gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

if ! command -v matensemble &>/dev/null; then
	echo "Install the MatEnsemble CLI for Perlmutter"
	echo "echo 'curl -fsSL https://raw.githubusercontent.com/FredDude2004/MatEnsemble/main/src/cli/install.sh | bash'"
	exit 1
fi

matensemble set-image ghcr.io/freddude2004/matensemble:perlmutter-v0.3.11
matensemble run workflow.py

