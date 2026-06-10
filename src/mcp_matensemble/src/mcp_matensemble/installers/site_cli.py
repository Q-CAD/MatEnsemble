from __future__ import annotations


FRONTIER_CLI = r'''#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/matensemble"
CONFIG_FILE="$CONFIG_DIR/cli.env"

err() { echo "matensemble: error: $*" >&2; exit 1; }

usage() {
  cat <<'USAGE'
Usage:
  matensemble set-image <sif-file or sandbox>
  matensemble run <script.py> [args...]
  matensemble shell
USAGE
}

load_config() {
  if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
  fi
  : "${MATENSEMBLE_IMAGE:=}"
}

save_image() {
  local image="$1"
  mkdir -p "$CONFIG_DIR"
  if [[ "$image" != *"://" ]]; then
    [[ -e "$image" ]] || err "image path does not exist: $image"
    image="$(cd "$(dirname "$image")" && pwd)/$(basename "$image")"
  fi
  printf 'export MATENSEMBLE_SYSTEM=frontier\nexport MATENSEMBLE_IMAGE=%q\n' "$image" > "$CONFIG_FILE"
  echo "Set Frontier MatEnsemble image: $image"
}

check_allocation() {
  [[ -n "${SLURM_JOB_ID:-}" ]] || err "not inside a SLURM allocation; use salloc/sbatch first"
  [[ -n "${SLURM_NNODES:-}" || -n "${SLURM_JOB_NUM_NODES:-}" ]] || err "SLURM node count unavailable"
}

run_workflow() {
  local script="$1"; shift || true
  [[ -f "$script" ]] || err "script does not exist: $script"
  load_config
  [[ -n "$MATENSEMBLE_IMAGE" ]] || err "no image configured; run: matensemble set-image <image>"
  check_allocation
  local nnodes="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES}}"
  exec srun -N "$nnodes" -n "$nnodes" --external-launcher --gpu-bind=closest --mpi=pmi2 \
    apptainer exec "$MATENSEMBLE_IMAGE" flux start python "$script" "$@"
}

shell_workflow() {
  load_config
  [[ -n "$MATENSEMBLE_IMAGE" ]] || err "no image configured; run: matensemble set-image <image>"
  check_allocation
  local nnodes="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES}}"
  exec srun -N "$nnodes" -n "$nnodes" --external-launcher --gpu-bind=closest --mpi=pmi2 --pty \
    apptainer exec "$MATENSEMBLE_IMAGE" flux start
}

case "${1:-}" in
  set-image) [[ $# -eq 2 ]] || { usage; exit 2; }; save_image "$2" ;;
  run) [[ $# -ge 2 ]] || { usage; exit 2; }; shift; run_workflow "$@" ;;
  shell|interactive) shell_workflow ;;
  -h|--help|help|"") usage ;;
  *) usage >&2; exit 2 ;;
esac
'''


PERLMUTTER_CLI = r'''#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/matensemble"
CONFIG_FILE="$CONFIG_DIR/cli.env"

err() { echo "matensemble: error: $*" >&2; exit 1; }

usage() {
  cat <<'USAGE'
Usage:
  matensemble set-image <image-tag>
  matensemble run <script.py> [args...]
USAGE
}

load_config() {
  if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
  fi
  : "${MATENSEMBLE_IMAGE:=}"
}

save_image() {
  local image="$1"
  mkdir -p "$CONFIG_DIR"
  printf 'export MATENSEMBLE_SYSTEM=perlmutter\nexport MATENSEMBLE_IMAGE=%q\n' "$image" > "$CONFIG_FILE"
  echo "Set Perlmutter MatEnsemble image: $image"
}

check_allocation() {
  [[ -n "${SLURM_JOB_ID:-}" ]] || err "not inside a SLURM allocation; use salloc/sbatch first"
  [[ -n "${SLURM_NNODES:-}" || -n "${SLURM_JOB_NUM_NODES:-}" ]] || err "SLURM node count unavailable"
  [[ -n "${SCRATCH:-}" ]] || err "SCRATCH is not set"
}

run_workflow() {
  local script="$1"; shift || true
  [[ -f "$script" ]] || err "script does not exist: $script"
  load_config
  [[ -n "$MATENSEMBLE_IMAGE" ]] || err "no image configured; run: matensemble set-image <image-tag>"
  check_allocation
  command -v podman-hpc >/dev/null 2>&1 || err "required command not found: podman-hpc"
  local nnodes="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES}}"
  export MATENSEMBLE_GPUS_PER_NODE="${MATENSEMBLE_GPUS_PER_NODE:-4}"
  local container_script="${SCRATCH}/matensemble_flux_${SLURM_JOB_ID}_$$.sh"
  local gpu_range="0-$((MATENSEMBLE_GPUS_PER_NODE - 1))"
  local flux_cmd=(python "$script" "$@")
  local flux_cmd_quoted=""
  printf -v flux_cmd_quoted '%q ' "${flux_cmd[@]}"
  cat >"$container_script" <<SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
python3 << 'PYEOF' > /tmp/R.json
import json, os, socket
nodes = os.environ.get("SLURM_JOB_NODELIST", socket.gethostname())
ranks = int(os.environ.get("SLURM_NNODES", os.environ.get("SLURM_JOB_NUM_NODES", "1")))
ncores = os.cpu_count()
r = {"version": 1, "execution": {"R_lite": [{"rank": str(i), "children": {"core": f"0-{ncores-1}", "gpu": "${gpu_range}"}} for i in range(ranks)], "starttime": 0.0, "expiration": 0.0, "nodelist": [nodes]}}
print(json.dumps(r))
PYEOF
mkdir -p /tmp/fluxcfg
printf '[resource]\npath = "/tmp/R.json"\nnoverify = true\n' > /tmp/fluxcfg/resource.toml
FLUX_CONF_DIR=/tmp/fluxcfg flux start ${flux_cmd_quoted}
SCRIPT_EOF
  chmod 700 "$container_script"
  trap 'rm -f "$container_script"' EXIT
  exec srun -N "$nnodes" -n "$nnodes" --cpu-bind=none --mpi=pmi2 \
    podman-hpc run --rm -e 'SLURM_*' -e 'PALS_*' -e 'PMI_*' \
      --ipc=host --network=host --pid=host --privileged \
      -v "$SCRATCH:$SCRATCH" -v "$PWD:$PWD" -w "$PWD" \
      "$MATENSEMBLE_IMAGE" bash "$container_script"
}

case "${1:-}" in
  set-image) [[ $# -eq 2 ]] || { usage; exit 2; }; save_image "$2" ;;
  run) [[ $# -ge 2 ]] || { usage; exit 2; }; shift; run_workflow "$@" ;;
  -h|--help|help|"") usage ;;
  *) usage >&2; exit 2 ;;
esac
'''


def site_cli_script(system: str) -> str:
    key = system.strip().lower().replace("-", "_")
    if key in {"frontier", "pathfinder"}:
        return FRONTIER_CLI.replace("frontier", key)
    if key == "perlmutter":
        return PERLMUTTER_CLI
    raise ValueError(f"site CLI is only available for frontier, pathfinder, or perlmutter: {system!r}")
