#!/usr/bin/bash

#SBATCH -A nstaff
#SBATCH -C gpu              # Specifically request the 40gb GPUs; this is purely for throughput reasons
#SBATCH --qos debug         # Update as appropriate
#SBATCH -t 0:30:00          # One hour runtime
#SBATCH -N 2
#SBATCH --ntasks-per-node=1 # 1 Flux broker per node
#SBATCH --gpus-per-node=4   # 4 GPUs per node
#SBATCH --gpu-bind=closest

# Command I was using to test the images
# salloc -A m5014_g -C gpu --qos=debug -t 0:30:00 -N 2 --ntasks-per-node=1 --gpus-per-node=4 --gpu-bind=closest

export NNODES=${SLURM_JOB_NUM_NODES}
export NGPUS=4
export NRANK=$((NGPUS * NNODES))
export RUNSTRING="${NNODES}node-${NGPUS}gpu-J${SLURM_JOB_ID}"

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export SLURM_CPU_BIND=cores
export CRAY_ACCEL_TARGET=nvidia80
export SUPPRESS_NVIDIA_HEADER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# name of image and input to give after 'flux start'
IMAGE="ghcr.io/freddude2004/matensemble:perlmutter-dev"
INPUT="example_matensemble.py"

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

# --- Flux configuration (update per application) --------
GPU_RANGE="0-$((NGPUS - 1))"
FLUX_CMD="python $INPUT"
# ---------------------------------------------------------

# This will create a script that will allow flux to see all the correct resources
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

# Write container script to $SCRATCH (mounted inside the container) so it
# can be passed as a path argument to bash — avoids stdin piping through srun/podman.
CONTAINER_SCRIPT="${SCRATCH}/flux_container_${SLURM_JOB_ID}.sh"
cat >"${CONTAINER_SCRIPT}" <<SCRIPT_EOF
python3 << 'PYEOF' > /tmp/R.json
import json, socket, os, re
def expand_nodelist(s):
    nodes = []
    for part in re.split(r',(?![^\[]*\])', s):
        m = re.match(r'^(.*?)\[(.+)\]$', part)
        if m:
            prefix, ranges = m.group(1), m.group(2)
            for r in ranges.split(','):
                if '-' in r:
                    a, b = r.split('-')
                    for i in range(int(a), int(b)+1):
                        nodes.append(f"{prefix}{str(i).zfill(len(a))}")
                else:
                    nodes.append(f"{prefix}{r}")
        else:
            nodes.append(part)
    return nodes
ncores = os.cpu_count()
nodelist_str = os.environ.get('SLURM_JOB_NODELIST', socket.gethostname())
nodes = expand_nodelist(nodelist_str)
r_lite = [{"rank": str(i), "children": {"core": f"0-{ncores-1}", "gpu": "${GPU_RANGE}"}}
          for i in range(len(nodes))]
r = {"version": 1, "execution": {
    "R_lite": r_lite,
    "starttime": 0.0, "expiration": 0.0,
    "nodelist": nodes}}
print(json.dumps(r))
PYEOF

mkdir -p /tmp/fluxcfg
cat > /tmp/fluxcfg/resource.toml << 'TOML'
[resource]
path = "/tmp/R.json"
noverify = true
TOML

FLUX_CONF_DIR=/tmp/fluxcfg flux start ${FLUX_CMD}
SCRIPT_EOF

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

srun --cpu-bind=none --mpi=pmi2 podman-hpc run ${PODMAN_ARGS} ${IMAGE} bash ${CONTAINER_SCRIPT}

rm -f "${CONTAINER_SCRIPT}"
