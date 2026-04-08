#! /usr/bin/bash

#SBATCH -A mXXXX
#SBATCH -C gpu              # Specifically request the 40gb GPUs; this is purely for throughput reasons
#SBATCH --qos debug         # Update as appropriate
#SBATCH -t 0:30:00          # One hour runtime
#SBATCH -N 2                # This is the total number of nodes
#SBATCH --ntasks-per-node=4 # Number of tasks per node
#SBATCH --gpus-per-task=1   # 1 GPU per MPI task
#SBATCH --gpu-bind=none     # This is necessary to let all 4 ranks on a node access all 4 GPUs.

export NNODES=${SLURM_JOB_NUM_NODES}
export NGPUS=${SLURM_NTASKS_PER_NODE}
export NRANK=$((${NGPU} * ${NNODES}))

### Cray/Slurm runtime flags
export OMP_NUM_THREADS=16
export SLURM_CPU_BIND=cores
export CRAY_ACCEL_TARGET=nvidia80

export RUNSTRING="${NNODES}node-${NGPUS}gpu-J${SLURM_JOB_ID}"
binding=" --mpi=pmi2 podman-hpc run --rm \
           -e SLURM_* -e PALS_* -e PMI_* --ipc=host --network=host --pid=host \
           --privileged -v /dev/shm:/dev/shm -v /dev/cxi0:/dev/cxi0 \
           -v /dev/cxi1:/dev/cxi1 -v /dev/cxi2:/dev/cxi2 -v /dev/cxi3:/dev/cxi3 \
           -v /dev/xpmem:/dev/xpmem -v /var/spool/slurmd:/var/spool/slurmd \
           -v /run/munge:/run/munge -v /run/nscd:/run/nscd  -e MPICH_GPU_SUPPORT_ENABLED=1 \
           -v $SCRATCH:$SCRATCH -w $PWD"
image="docker://docker.io/nersc/lammps_flux:25.07"
input="-k on g 4 -sf kk -pk kokkos newton off neigh half -in in.rhodo"

srun -v -N $NNODES --cpu-bind none --ntasks-per-node $NGPUS --gpus-per-task 1 --gpu-bind none \
     $binding $image lmp $input




srun -N $NNODES --cpu-bind --mpi=pmi2 podman-hpc run --rm 

srun -l -N2 -n2 --mpi=pmi2 \
    podman-hpc run --rm --gpu --net host \
    -e SLURM_* -e PMI_* -e PALS_* \
    ghcr.io/freddude2004/matensemble:perlmutter-dev \
    flux start flux resource list




# get the allocation
salloc -N 2 -t 60 -C gpu --qos interactive -A m5014_g -G 8

srun -N2 -n2 --mpi=pmi2 --gpus-per-node=4		      \
    podman-hpc run --rm --gpu --net host                      \
    -v "$PWD:$PWD" -w "$PWD"                                  \
    -e SLURM_* -e PMI_* -e PALS_*                             \
    ghcr.io/freddude2004/matensemble:perlmutter-dev-test      \
    flux start python run_mace_workflow.py


srun -N2 -n2 --mpi=pmi2 --gpus-per-node=4                                      \
    podman-hpc run --rm --gpu --net host                                       \
    -v "$PWD:$PWD" -w "$PWD"                                                   \
    -e SLURM_* -e PMI_* -e PALS_*                                              \
    ghcr.io/freddude2004/matensemble:perlmutter-dev-apr                        \
    flux start python run_mace_workflow.py


# run the workflow
srun -N2 -n2 --mpi=pmi2 --gpus-per-node=4                                      \		
    podman-hpc run --rm --gpu --net host                                       \ 		
    -v "$PWD:$PWD" -w "$PWD"                                                   \ 				
    -e SLURM_* -e PMI_* -e PALS_*                                              \			
    ghcr.io/freddude2004/matensemble:perlmutter-dev                            \	
    flux start python exec_workflow.py			

srun -N2 -n2 --mpi=pmi2 --gpus-per-node=4                                      \
    podman-hpc run --rm --gpu --net host                                       \
    -v "$PWD:$PWD" -w "$PWD"                                                   \
    -e SLURM_* -e PMI_* -e PALS_*                                              \
    ghcr.io/freddude2004/matensemble:perlmutter-dev                            \
    flux start flux resource list

#
# run the container with gpu support and clean up after finished 
# bind mount the CWD
# copy slurm and mpi vars to container 
# which container to run
# start flux and run workflow 							

