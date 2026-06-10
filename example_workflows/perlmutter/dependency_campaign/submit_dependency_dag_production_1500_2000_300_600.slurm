#!/usr/bin/env bash
#SBATCH -A m5064_g
#SBATCH -C gpu
#SBATCH --qos regular
#SBATCH -t 24:00:00
#SBATCH -N 6
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J recryst_dag
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd /pscratch/sd/s/sbagchi/ORNL_Work/Hetero_interfaces_MACE/FreeEnergy_/matensemble_dependency_campaign
mkdir -p logs

export RECRYST_CONFIG=recryst_config_dependency_production_1500_2000_300_600.json
export RECRYST_TASKS_PER_CHORE=1
export RECRYST_CORES_PER_TASK=32
export RECRYST_GPUS_PER_TASK=1
export RECRYST_LAMMPS_GPUS_PER_NODE=1
export RECRYST_GPU_BIND=closest
export RECRYST_OMP_NUM_THREADS=1
export RECRYST_KOKKOS_THREADS=1

CASES=(Bi2Se3-Al2O3 Bi2Se3-InP Bi2Se3-In2Se3 Bi2Se3-NbSe2 Bi2Se3-4L)
AMORPHOUS_SEEDS=(101 202 303)
CRYSTAL_SEEDS=(101)

./run_recryst_flux_dag.sh \
  --workflow full-dag \
  --spring-states amorphous crystal \
  --cases "${CASES[@]}" \
  --amorphous-seeds "${AMORPHOUS_SEEDS[@]}" \
  --crystal-seeds "${CRYSTAL_SEEDS[@]}" \
  --skip-existing

./run_dependency_dag_analysis_1500_2000_300_600.sh
