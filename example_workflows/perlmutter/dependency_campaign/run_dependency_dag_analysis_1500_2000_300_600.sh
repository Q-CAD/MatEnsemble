#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${CAMPAIGN_DIR}/.." && pwd)"
PYTHON_BIN="/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python"

export MPLCONFIGDIR="${ROOT_DIR}/.matplotlib-cache"
export XDG_CACHE_HOME="${ROOT_DIR}/.cache"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" analysis_tools/make_recryst_multiseed_report.py \
  --config matensemble_dependency_campaign/recryst_config_dependency_production_1500_2000_300_600.json \
  --base matensemble_dependency_campaign/recryst_runs_dependency_dag_prod_1500_2000_300_600 \
  --report-dir matensemble_dependency_campaign/reports/dependency_dag_production_1500_2000_300_600 \
  --amorphous-seeds 101 202 303 \
  --crystal-seeds 101
