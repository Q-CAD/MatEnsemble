from matensemble.pipeline import Pipeline
from mpi4py import MPI

# We first create a Pipeline and define an MPI-enabled chore that launches
# 10 parallel MPI ranks using mpi4py.
pipe = Pipeline()


@pipe.chore(num_tasks=10, cores_per_task=1, gpus_per_task=0, mpi=True)
def mpi_hello_world():
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()

    print(f"Hello World! I am process {rank} of {size} on {name}.")


# Then we add the chore to the workflow 10 seperate times
for _ in range(10):
    mpi_hello_world()

# in 10 separate MPI jobs being executed through the matensemble workflow
# runtime and scheduler backend.


pipe.submit(log_delay=1)
