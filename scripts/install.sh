#!/usr/bin/env bash
set -euo pipefail

APP_NAME="matensemble"
BIN_DIR="${HOME}/.local/bin"
APP_PATH="${BIN_DIR}/${APP_NAME}"

mkdir -p "$BIN_DIR"

cat >"$APP_PATH" <<'MATENSEMBLE_EOF'
#!/usr/bin/env bash
set -euo pipefail

MAT_DIR=".matensemble"

die() {
  echo "matensemble: error: $*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage:
  matensemble setup-run <podman-hpc-image> <script.py>
  matensemble run
  matensemble salloc <nodes>

Examples:
  matensemble setup-run ghcr.io/freddude2004/matensemble:perlmutter-dev lammps_mace_calculator.py
  matensemble salloc 2
  matensemble run
EOF
}

setup_run() {
  IMAGE="${1:-}"
  SCRIPT="${2:-}"

  [[ -n "$IMAGE" ]] || die "missing image"
  [[ -n "$SCRIPT" ]] || die "missing python script"
  [[ -f "$SCRIPT" ]] || die "script not found: $SCRIPT"

  mkdir -p "$MAT_DIR"

  cat > "$MAT_DIR/run.env" <<EOF
IMAGE="$IMAGE"
FLUX_CMD="python $SCRIPT"
NGPUS=4
EOF

  echo "Configured run in $MAT_DIR/run.env"
  echo "Next: matensemble run"
}

run_workflow() {
  [[ -f "$MAT_DIR/run.env" ]] || die "no run configured. Run: matensemble setup-run <image> <script.py>"

  # shellcheck disable=SC1091
  source "$MAT_DIR/run.env"

  [[ -n "${SLURM_JOB_ID:-}" ]] || die "not inside a Slurm allocation. Run matensemble salloc <nodes> first."
  [[ -n "${SCRATCH:-}" ]] || die "SCRATCH is not set."
  command -v srun >/dev/null || die "srun not found"
  command -v podman-hpc >/dev/null || die "podman-hpc not found"

  export NNODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
  export NGPUS="${NGPUS:-4}"
  export NRANK=$((NGPUS * NNODES))
  export RUNSTRING="${NNODES}node-${NGPUS}gpu-J${SLURM_JOB_ID}"

  export OMP_NUM_THREADS=1
  export OMP_PROC_BIND=spread
  export OMP_PLACES=threads
  export SLURM_CPU_BIND=cores
  export CRAY_ACCEL_TARGET=nvidia80
  export SUPPRESS_NVIDIA_HEADER=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  LD_PATH="/opt/basic/lib/python3.12/site-packages/nvidia/nccl/lib"
  LD_PATH="${LD_PATH}:/usr/lib64"
  LD_PATH="${LD_PATH}:/opt/lammps/build/kim_build-prefix/lib"
  LD_PATH="${LD_PATH}:/usr/local/cuda/lib64:/usr/local/cuda/compat"
  LD_PATH="${LD_PATH}:/opt/fftw/install/lib"
  LD_PATH="${LD_PATH}:/opt/hdf5/install/lib"
  LD_PATH="${LD_PATH}:/opt/lammps/build/plumed_build-prefix/lib"
  LD_PATH="${LD_PATH}:/opt/lammps/install/lib"
  LD_PATH="${LD_PATH}:/usr/local/blas"
  LD_PATH="${LD_PATH}:/opt/basic/lib/python3.12/site-packages/torch/lib"

  CONTAINER_SCRIPT="${SCRATCH}/flux_container_${SLURM_JOB_ID}.sh"
  GPU_RANGE="0-$((NGPUS - 1))"

  cat > "$CONTAINER_SCRIPT" <<EOF
python3 << 'PYEOF' > /tmp/R.json
import json, socket, os, re

def expand_nodelist(s):
    nodes = []
    for part in re.split(r",(?![^\\[]*\\])", s):
        m = re.match(r"^(.*?)\\[(.+)\\]$", part)
        if m:
            prefix, ranges = m.group(1), m.group(2)
            for r in ranges.split(","):
                if "-" in r:
                    a, b = r.split("-")
                    for i in range(int(a), int(b) + 1):
                        nodes.append(f"{prefix}{str(i).zfill(len(a))}")
                else:
                    nodes.append(f"{prefix}{r}")
        else:
            nodes.append(part)
    return nodes

ncores = os.cpu_count()
nodes = expand_nodelist(os.environ.get("SLURM_JOB_NODELIST", socket.gethostname()))
r = {
  "version": 1,
  "execution": {
    "R_lite": [
      {"rank": str(i), "children": {"core": f"0-{ncores-1}", "gpu": "$GPU_RANGE"}}
      for i in range(len(nodes))
    ],
    "starttime": 0.0,
    "expiration": 0.0,
    "nodelist": nodes,
  },
}
print(json.dumps(r))
PYEOF

mkdir -p /tmp/fluxcfg
cat > /tmp/fluxcfg/resource.toml << 'TOML'
[resource]
path = "/tmp/R.json"
noverify = true
TOML

FLUX_CONF_DIR=/tmp/fluxcfg flux start $FLUX_CMD
EOF

  PODMAN_ARGS="--rm \
    -e SLURM_* -e PALS_* -e PMI_* \
    --ipc=host --network=host --pid=host --privileged \
    -v /dev/shm:/dev/shm \
    -v /dev/cxi0:/dev/cxi0 -v /dev/cxi1:/dev/cxi1 \
    -v /dev/cxi2:/dev/cxi2 -v /dev/cxi3:/dev/cxi3 \
    -v /dev/xpmem:/dev/xpmem \
    -v /var/spool/slurmd:/var/spool/slurmd \
    -v /run/munge:/run/munge -v /run/nscd:/run/nscd \
    -v /dev/nvidia-caps-imex-channels:/dev/nvidia-caps-imex-channels \
    --device /dev/nvidia0 --device /dev/nvidia1 \
    --device /dev/nvidia2 --device /dev/nvidia3 \
    --device /dev/nvidiactl \
    --device /dev/nvidia-uvm \
    --device /dev/nvidia-uvm-tools \
    -v /usr/lib64:/usr/local/nvidia/lib64 \
    -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi \
    -e LD_PRELOAD=/opt/basic/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2 \
    -e MPICH_GPU_SUPPORT_ENABLED=1 \
    -e SUPPRESS_NVIDIA_HEADER \
    -e CUDA_VISIBLE_DEVICES \
    -e KOKKOS_PRINT_CONFIGURATION=1 \
    -e OMP_NUM_THREADS -e OMP_PROC_BIND -e OMP_PLACES \
    -e LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_PATH} \
    -v $SCRATCH:$SCRATCH -w $PWD"

  srun -N "$NNODES" -n "$NNODES" --cpu-bind=none --mpi=pmi2 \
    podman-hpc run ${PODMAN_ARGS} "$IMAGE" bash "$CONTAINER_SCRIPT"

  rm -f "$CONTAINER_SCRIPT"
}

start_allocation() {
  NODES="${1:-2}"

  salloc \
    -A "${MATENSEMBLE_ACCOUNT:-nstaff}" \
    -C gpu \
    --qos="${MATENSEMBLE_QOS:-debug}" \
    -t "${MATENSEMBLE_TIME:-0:30:00}" \
    -N "$NODES" \
    --ntasks-per-node=1 \
    --gpus-per-node=4 \
    --gpu-bind=closest
}

case "${1:-}" in
  setup-run)
    shift
    setup_run "$@"
    ;;
  run)
    shift
    run_workflow "$@"
    ;;
  salloc)
    shift
    start_allocation "$@"
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    die "unknown command: $1"
    ;;
esac
MATENSEMBLE_EOF

chmod +x "$APP_PATH"

add_path_line='export PATH="$HOME/.local/bin:$PATH"'

if ! command -v "$APP_NAME" >/dev/null 2>&1; then
	for rc in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile"; do
		if [[ -f "$rc" ]] && ! grep -Fq '.local/bin' "$rc"; then
			{
				echo ""
				echo "# Added by MatEnsemble installer"
				echo "$add_path_line"
			} >>"$rc"
		fi
	done
fi

echo "Installed matensemble to $APP_PATH"

case ":$PATH:" in
*":$BIN_DIR:"*)
	echo "You can now run: matensemble --help"
	;;
*)
	echo "Added $BIN_DIR to your shell startup files."
	echo "Run this now:"
	echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
	echo ""
	echo "Then try:"
	echo "  matensemble --help"
	;;
esac
