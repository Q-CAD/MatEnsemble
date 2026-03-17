from mpi4py import MPI
from matensemble.pipeline import Pipeline

pipe = Pipeline()


@pipe.job()
def run_mpi_hello(task_id: int):
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()

    with open(f"task_{task_id}_rank_{rank}.txt", "w") as f:
        f.write(f"Hello from rank {rank}/{size} on {name}, task={task_id}\n")

    return rank
