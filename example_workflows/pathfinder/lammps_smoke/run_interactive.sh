#!/usr/bin/env bash

if ! command -v matensemble >/dev/null 2>&1; then
	echo "Install the MatEnsemble CLI for Pathfinder"
	echo "curl -fsSL https://raw.githubusercontent.com/Q-CAD/MatEnsemble/main/src/cli/install.sh | bash"
	exit 1
fi

matensemble set-image ./matensemble.sif
matensemble run workflow.py
