#!/usr/bin/bash

module reset
module load olcf-container-tools
module load apptainer-enable-mpi
module load apptainer-enable-gpu

matensemble set-image <path/to/sif/or/sandbox>
matensemble run online_dynamics.py 0 5

