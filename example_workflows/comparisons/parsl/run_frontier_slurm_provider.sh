#!/bin/bash
# Run this from a Frontier login node. Parsl's SlurmProvider will submit a worker block.
# Usage: ./run_frontier_slurm_provider.sh <PROJECT_ID>
set -euo pipefail
PROJECT_ID="${1:?Usage: $0 <PROJECT_ID>}"

module load parsl/2024.12.2

python3 adaptive_rastrigin_parsl.py \
  --parsl-config frontier-slurm \
  --account "$PROJECT_ID" \
  --partition batch \
  --nodes-per-block 1 \
  --init-blocks 1 \
  --max-blocks 1 \
  --block-walltime 00:10:00 \
  --outdir "rastrigin_outputs_slurm_provider_$(date +%Y%m%d_%H%M%S)" \
  --run-dir "parsl_runinfo_slurm_provider" \
  --arm-grid 8 \
  --n-candidates 128 \
  --initial-batch 56 \
  --max-in-flight 56 \
  --target-wall-s 240 \
  --max-workers-per-node 56 \
  --sleep-scale 1.0 \
  --monitoring
