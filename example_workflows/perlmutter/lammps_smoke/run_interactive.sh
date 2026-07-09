#!/usr/bin/bash

if ! command -v matensemble &>/dev/null; then
	echo "Install the MatEnsemble CLI for Perlmutter"
	echo "echo 'curl -fsSL https://raw.githubusercontent.com/Q-CAD/MatEnsemble/main/src/cli/install.sh | bash'"
	exit 1
fi

matensemble set-image ghcr.io/q-cad/matensemble:perlmutter-v0.3.11
matensemble run workflow.py
