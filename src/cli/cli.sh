#!/usr/bin/env bash
set -euo pipefail

VERSION="0.1.0"

CONFIG_DIR="${MATENSEMBLE_CONFIG_DIR:-$HOME/.config/matensemble}"
CONFIG_FILE="${CONFIG_DIR}/config"

PROFILE="${MATENSEMBLE_PROFILE:-default}"
DRY_RUN=0
KEEP_SCRIPT=0

usage() {
	cat <<'EOF'
MatEnsemble CLI

Usage:
  matensemble [--profile PROFILE] set-image <image-or-sif>
  matensemble [--profile PROFILE] run <script.py> [--dry-run] [--keep-script]
  matensemble [--profile PROFILE] test run <script.py> [--size N] [--dry-run]
  matensemble --help
  matensemble --version

Profiles:
  default      Generic Apptainer profile
  frontier     OLCF Frontier Apptainer profile
  perlmutter   NERSC Perlmutter podman-hpc profile

Environment:
  MATENSEMBLE_PROFILE       Default profile
  MATENSEMBLE_CONFIG_DIR    Config directory, default ~/.config/matensemble

Examples:
  matensemble --profile perlmutter set-image ghcr.io/q-cad/matensemble:perlmutter-vX.Y.Z
  matensemble --profile perlmutter run run_workflow.py

  matensemble --profile frontier set-image /path/to/matensemble_frontier.sif
  matensemble --profile frontier run run_workflow.py
  matensemble --profile frontier test run run_workflow.py --size 2
EOF
}

die() {
	echo "matensemble: error: $*" >&2
	exit 1
}

info() {
	echo "matensemble: $*" >&2
}

need_cmd() {
	command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

shell_quote() {
	printf "%q" "$1"
}

ensure_config_dir() {
	mkdir -p "${CONFIG_DIR}"
	touch "${CONFIG_FILE}"
}

config_key() {
	local profile="$1"
	local key="$2"
	echo "${profile}.${key}"
}

config_get() {
	local key="$1"
	if [[ ! -f "${CONFIG_FILE}" ]]; then
		return 1
	fi

	grep -E "^${key}=" "${CONFIG_FILE}" | tail -n 1 | cut -d '=' -f 2-
}

config_set() {
	local key="$1"
	local value="$2"

	ensure_config_dir

	local tmp
	tmp="$(mktemp)"

	if [[ -f "${CONFIG_FILE}" ]]; then
		grep -v -E "^${key}=" "${CONFIG_FILE}" >"${tmp}" || true
	fi

	echo "${key}=${value}" >>"${tmp}"
	mv "${tmp}" "${CONFIG_FILE}"
}

get_image() {
	local key
	key="$(config_key "${PROFILE}" image)"

	local image
	image="$(config_get "${key}" || true)"

	[[ -n "${image}" ]] || die "no image configured for profile '${PROFILE}'. Run: matensemble --profile ${PROFILE} set-image <image-or-sif>"

	echo "${image}"
}

in_slurm_allocation() {
	[[ -n "${SLURM_JOB_ID:-}" ]] && { [[ -n "${SLURM_NNODES:-}" ]] || [[ -n "${SLURM_JOB_NUM_NODES:-}" ]]; }
}

slurm_nnodes() {
	if [[ -n "${SLURM_NNODES:-}" ]]; then
		echo "${SLURM_NNODES}"
	elif [[ -n "${SLURM_JOB_NUM_NODES:-}" ]]; then
		echo "${SLURM_JOB_NUM_NODES}"
	else
		die "could not determine node count; SLURM_NNODES and SLURM_JOB_NUM_NODES are unset"
	fi
}

require_slurm_allocation() {
	if ! in_slurm_allocation; then
		cat >&2 <<EOF
matensemble: error: run must be executed inside a Slurm allocation.

Try something like:
  salloc -A <project> -N 2 -t 00:30:00

For a login-node smoke test on Apptainer-based systems:
  matensemble --profile frontier test run run_workflow.py --size 2
EOF
		exit 1
	fi
}

run_cmd() {
	if [[ "${DRY_RUN}" == "1" ]]; then
		printf 'DRY RUN:'
		for arg in "$@"; do
			printf ' %q' "$arg"
		done
		printf '\n'
		return 0
	fi

	"$@"
}

validate_script() {
	local script="$1"

	[[ -f "${script}" ]] || die "script does not exist: ${script}"

	case "${script}" in
	*.py) ;;
	*) info "warning: script does not end in .py: ${script}" ;;
	esac
}

validate_image_for_profile() {
	local image="$1"

	case "${PROFILE}" in
	perlmutter)
		need_cmd podman-hpc
		podman-hpc images | grep -Fq "${image}" || {
			cat >&2 <<EOF
matensemble: error: image not found in podman-hpc images:

  ${image}

Try:
  podman-hpc pull ${image}
EOF
			exit 1
		}
		;;
	frontier | default)
		need_cmd apptainer

		if [[ "${image}" == docker://* || "${image}" == oras://* || "${image}" == library://* ]]; then
			info "using remote Apptainer image URI: ${image}"
		else
			[[ -f "${image}" ]] || die "Apptainer image does not exist: ${image}"
		fi
		;;
	*)
		die "unknown profile: ${PROFILE}"
		;;
	esac
}

cmd_set_image() {
	local image="${1:-}"
	[[ -n "${image}" ]] || die "missing image"

	validate_image_for_profile "${image}"

	local key
	key="$(config_key "${PROFILE}" image)"
	config_set "${key}" "${image}"

	info "saved image for profile '${PROFILE}': ${image}"
}

apptainer_common_args() {
	# Keep this conservative. Sites can extend later.
	echo "--cleanenv"
}

cmd_run_default() {
	local script="$1"
	local image
	image="$(get_image)"

	need_cmd srun
	need_cmd apptainer
	require_slurm_allocation

	local nnodes
	nnodes="$(slurm_nnodes)"

	run_cmd \
		srun -N "${nnodes}" -n "${nnodes}" \
		--mpi=pmi2 \
		apptainer exec $(apptainer_common_args) "${image}" \
		flux start python "${script}"
}

cmd_run_frontier() {
	local script="$1"
	local image
	image="$(get_image)"

	need_cmd srun
	need_cmd apptainer
	require_slurm_allocation

	local nnodes
	nnodes="$(slurm_nnodes)"

	run_cmd \
		srun -N "${nnodes}" -n "${nnodes}" \
		--external-launcher \
		--mpi=pmi2 \
		--gpu-bind=closest \
		apptainer exec $(apptainer_common_args) "${image}" \
		flux start python "${script}"
}

write_perlmutter_container_script() {
	local script="$1"

	[[ -n "${SCRATCH:-}" ]] || die "SCRATCH is not set; needed for Perlmutter generated container script"
	[[ -n "${SLURM_JOB_ID:-}" ]] || die "SLURM_JOB_ID is not set"

	local gpus_per_node="${MATENSEMBLE_GPUS_PER_NODE:-4}"
	local gpu_range="0-$((gpus_per_node - 1))"

	local container_script
	container_script="${SCRATCH}/matensemble_flux_container_${SLURM_JOB_ID}_$$.sh"

	cat >"${container_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

python3 << 'PYEOF' > /tmp/R.json
import json
import socket
import os
import re

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
nodelist_str = os.environ.get("SLURM_JOB_NODELIST", socket.gethostname())
nodes = expand_nodelist(nodelist_str)

r_lite = [
    {"rank": str(i), "children": {"core": f"0-{ncores-1}", "gpu": "${gpu_range}"}}
    for i in range(len(nodes))
]

r = {
    "version": 1,
    "execution": {
        "R_lite": r_lite,
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

FLUX_CONF_DIR=/tmp/fluxcfg flux start python $(shell_quote "${script}")
EOF

	chmod +x "${container_script}"
	echo "${container_script}"
}

perlmutter_podman_args() {
	local ld_path
	ld_path="/opt/basic/lib/python3.12/site-packages/nvidia/nccl/lib"
	ld_path="${ld_path}:/usr/lib64"
	ld_path="${ld_path}:/opt/lammps/build/kim_build-prefix/lib"
	ld_path="${ld_path}:/usr/local/cuda/lib64:/usr/local/cuda/compat"
	ld_path="${ld_path}:/opt/fftw/install/lib"
	ld_path="${ld_path}:/opt/hdf5/install/lib"
	ld_path="${ld_path}:/opt/lammps/build/plumed_build-prefix/lib"
	ld_path="${ld_path}:/opt/lammps/install/lib"
	ld_path="${ld_path}:/usr/local/blas"
	ld_path="${ld_path}:/opt/basic/lib/python3.12/site-packages/torch/lib"

	local args=(
		--rm
		-e "SLURM_*"
		-e "PALS_*"
		-e "PMI_*"
		--ipc=host
		--network=host
		--pid=host
		--privileged
		-v /dev/shm:/dev/shm
		-v /dev/cxi0:/dev/cxi0
		-v /dev/cxi1:/dev/cxi1
		-v /dev/cxi2:/dev/cxi2
		-v /dev/cxi3:/dev/cxi3
		-v /dev/xpmem:/dev/xpmem
		-v /var/spool/slurmd:/var/spool/slurmd
		-v /run/munge:/run/munge
		-v /run/nscd:/run/nscd
		-v /dev/nvidia-caps-imex-channels:/dev/nvidia-caps-imex-channels
		--device /dev/nvidia0
		--device /dev/nvidia1
		--device /dev/nvidia2
		--device /dev/nvidia3
		--device /dev/nvidiactl
		--device /dev/nvidia-uvm
		--device /dev/nvidia-uvm-tools
		-v /usr/lib64:/usr/local/nvidia/lib64
		-v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi
		-e LD_PRELOAD=/opt/basic/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
		-e MPICH_GPU_SUPPORT_ENABLED=1
		-e SUPPRESS_NVIDIA_HEADER
		-e CUDA_VISIBLE_DEVICES
		-e KOKKOS_PRINT_CONFIGURATION=1
		-e OMP_NUM_THREADS
		-e OMP_PROC_BIND
		-e OMP_PLACES
		-e "LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${ld_path}"
		-w "$PWD"
	)

	if [[ -n "${SCRATCH:-}" ]]; then
		args+=(-v "${SCRATCH}:${SCRATCH}")
	fi

	printf '%s\n' "${args[@]}"
}

cmd_run_perlmutter() {
	local script="$1"
	local image
	image="$(get_image)"

	need_cmd srun
	need_cmd podman-hpc
	require_slurm_allocation

	export NNODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}}"
	export NGPUS="${MATENSEMBLE_GPUS_PER_NODE:-4}"
	export NRANK="$((NGPUS * NNODES))"
	export RUNSTRING="${NNODES}node-${NGPUS}gpu-J${SLURM_JOB_ID}"

	export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
	export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
	export OMP_PLACES="${OMP_PLACES:-threads}"
	export SLURM_CPU_BIND="${SLURM_CPU_BIND:-cores}"
	export CRAY_ACCEL_TARGET="${CRAY_ACCEL_TARGET:-nvidia80}"
	export SUPPRESS_NVIDIA_HEADER=1
	export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

	local container_script
	container_script="$(write_perlmutter_container_script "${script}")"

	if [[ "${KEEP_SCRIPT}" == "1" ]]; then
		info "keeping generated container script: ${container_script}"
	else
		trap 'rm -f "${container_script}"' EXIT
	fi

	local nnodes
	nnodes="$(slurm_nnodes)"

	local podman_args=()
	while IFS= read -r arg; do
		podman_args+=("${arg}")
	done < <(perlmutter_podman_args)

	local cmd=(
		srun
		-N "${nnodes}"
		-n "${nnodes}"
		--cpu-bind=none
		--mpi=pmi2
		podman-hpc run
	)

	cmd+=("${podman_args[@]}")
	cmd+=("${image}" bash "${container_script}")

	run_cmd "${cmd[@]}"
}

cmd_run() {
	local script="${1:-}"
	[[ -n "${script}" ]] || die "missing script"
	validate_script "${script}"

	case "${PROFILE}" in
	perlmutter)
		cmd_run_perlmutter "${script}"
		;;
	frontier)
		cmd_run_frontier "${script}"
		;;
	default)
		cmd_run_default "${script}"
		;;
	*)
		die "unknown profile: ${PROFILE}"
		;;
	esac
}

cmd_test_run() {
	local script=""
	local size="2"

	[[ "${1:-}" == "run" ]] || die "expected: matensemble test run <script.py>"
	shift

	script="${1:-}"
	[[ -n "${script}" ]] || die "missing script"
	shift || true

	while [[ "$#" -gt 0 ]]; do
		case "$1" in
		--size)
			size="${2:-}"
			[[ -n "${size}" ]] || die "--size requires a value"
			shift 2
			;;
		--dry-run)
			DRY_RUN=1
			shift
			;;
		*)
			die "unknown test option: $1"
			;;
		esac
	done

	validate_script "${script}"

	case "${PROFILE}" in
	frontier | default)
		;;
	perlmutter)
		die "test run is currently only implemented for Apptainer-based profiles"
		;;
	*)
		die "unknown profile: ${PROFILE}"
		;;
	esac

	local image
	image="$(get_image)"

	need_cmd apptainer

	run_cmd \
		apptainer exec $(apptainer_common_args) "${image}" \
		flux start --test-size="${size}" python "${script}"
}

parse_global_flags() {
	while [[ "$#" -gt 0 ]]; do
		case "$1" in
		--profile)
			PROFILE="${2:-}"
			[[ -n "${PROFILE}" ]] || die "--profile requires a value"
			shift 2
			;;
		--dry-run)
			DRY_RUN=1
			shift
			;;
		--keep-script)
			KEEP_SCRIPT=1
			shift
			;;
		--help | -h)
			usage
			exit 0
			;;
		--version)
			echo "matensemble ${VERSION}"
			exit 0
			;;
		*)
			break
			;;
		esac
	done

	REMAINING_ARGS=("$@")
}

main() {
	local REMAINING_ARGS=()
	parse_global_flags "$@"
	set -- "${REMAINING_ARGS[@]}"

	local command="${1:-}"
	[[ -n "${command}" ]] || {
		usage
		exit 1
	}

	shift || true

	case "${command}" in
	set-image)
		cmd_set_image "$@"
		;;
	run)
		# Parse run-local flags after script too.
		local script="${1:-}"
		[[ -n "${script}" ]] || die "missing script"
		shift || true

		while [[ "$#" -gt 0 ]]; do
			case "$1" in
			--dry-run)
				DRY_RUN=1
				shift
				;;
			--keep-script)
				KEEP_SCRIPT=1
				shift
				;;
			*)
				die "unknown run option: $1"
				;;
			esac
		done

		cmd_run "${script}"
		;;
	test)
		cmd_test_run "$@"
		;;
	--help | -h)
		usage
		;;
	--version)
		echo "matensemble ${VERSION}"
		;;
	*)
		die "unknown command: ${command}"
		;;
	esac
}

main "$@"
