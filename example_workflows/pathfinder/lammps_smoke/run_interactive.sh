#!/usr/bin/env bash

#SBATCH -A <account>
#SBATCH -J pathfinder_matensemble_smoke
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -t 00:10:00
#SBATCH -N 2

if ! command -v matensemble >/dev/null 2>&1; then
	echo "Install the MatEnsemble CLI for Pathfinder"
	echo "curl -fsSL https://raw.githubusercontent.com/FredDude2004/MatEnsemble/main/src/cli/install.sh | bash"
	exit 1
fi

matensemble set-image ./matensemble.sif
matensemble run workflow.py
