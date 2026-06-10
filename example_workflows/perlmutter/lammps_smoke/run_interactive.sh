#!/usr/bin/bash

if ! command -v matensemble &>/dev/null; then
	echo "Install the MatEnsemble CLI for Perlmutter"
	echo "curl -fsSL https://raw.githubusercontent.com/FredDude2004/MatEnsemble/refs/heads/main/src/cli/install.sh | bash"
	exit 1
fi

matensemble set-image ghcr.io/freddude2004/matensemble:perlmutter-v0.3.11
matensemble run lammps_hello_world.py
