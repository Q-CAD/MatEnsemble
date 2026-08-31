#!/bin/bash
#SBATCH -A <PROJECT_ID>
#SBATCH -J parsl-rastrigin-local
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -t 00:10:00
#SBATCH -S 8
#SBATCH -o parsl-rastrigin-%j.out
#SBATCH -e parsl-rastrigin-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Easiest path when the site module is available:
module load parsl/2024.12.2

# Alternative conda path:
# module load PrgEnv-gnu/8.6.0
# module load miniforge3/23.11.0-0
# conda activate /ccs/proj/<PROJECT_ID>/$USER/envs/frontier/parsl-rastrigin

python3 adaptive_rastrigin_parsl.py \
  --parsl-config local-htex \
  --outdir "rastrigin_outputs_${SLURM_JOB_ID}" \
  --run-dir "parsl_runinfo_${SLURM_JOB_ID}" \
  --arm-grid 8 \
  --n-candidates 128 \
  --initial-batch 56 \
  --max-in-flight 56 \
  --target-wall-s 240 \
  --max-workers-per-node 56 \
  --sleep-scale 1.0 \
  --monitoring
