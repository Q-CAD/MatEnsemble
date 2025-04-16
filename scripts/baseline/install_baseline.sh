#!/bin/bash

module load python
# Check if environment path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/conda/environment"
    echo "Example: $0 ~/envs/matensemble_env"
    exit 1
fi

ENV_PATH="$1"

# Check if directory exists
if [ -d "$ENV_PATH" ]; then
    read -p "Directory $ENV_PATH already exists. Do you want to remove it and continue? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY]) 
            echo "Removing existing environment..."
            rm -rf "$ENV_PATH"
            ;;
        *)
            echo "Installation cancelled"
            exit 1
            ;;
    esac
fi

# Create and activate conda environment with custom path
echo "Creating conda environment at $ENV_PATH..."
conda env create -f environment.yaml --prefix "$ENV_PATH"

if [ $? -ne 0 ]; then
    echo "Failed to create conda environment"
    exit 1
fi

source activate "$ENV_PATH"

# Clone and build LAMMPS (and possibly also mpi4py?) from source
chmod +x build_lammps.sh
./build_lammps.sh

# Install Python dependencies
pip install -r requirements.txt

# go back to the root directory of the repository
cd ../../
# Install package in development mode
pip install -e .
