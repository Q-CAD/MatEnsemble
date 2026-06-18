#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

export RECRYST_CONFIG=recryst_config_dependency_smoke.json
export RECRYST_TASKS_PER_CHORE=1
export RECRYST_CORES_PER_TASK=32
export RECRYST_GPUS_PER_TASK=1
export RECRYST_LAMMPS_GPUS_PER_NODE=1
export RECRYST_GPU_BIND=closest
export RECRYST_OMP_NUM_THREADS=1
export RECRYST_KOKKOS_THREADS=1

./run_recryst_flux_dag.sh \
  --workflow full-dag \
  --spring-states amorphous crystal \
  --cases Bi2Se3-Al2O3 \
  --amorphous-seeds 777 \
  --crystal-seeds 777
