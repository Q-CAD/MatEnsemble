from matensemble.pipeline import Pipeline
from mpi4py import MPI

# We first create a Pipeline and define an MPI-enabled chore that launches
# 10 parallel MPI ranks using mpi4py.
pipe = Pipeline()


@pipe.chore(num_tasks=10, cores_per_task=1, gpus_per_task=0, mpi=true)
def mpi_hello_world():
    size = mpi.comm_world.get_size()
    rank = mpi.comm_world.get_rank()
    name = mpi.get_processor_name()

    print(f"hello world! i am process {rank} of {size} on {name}.")


# Then we add the chore to the workflow 10 separate times
for _ in range(10):
    mpi_hello_world()

# in 10 separate MPI jobs being executed through the matensemble workflow
# runtime and scheduler backend.


pipe.submit(log_delay=1)
