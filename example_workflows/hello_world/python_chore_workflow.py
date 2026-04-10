# python_chore_workflow.py
from mpi4py import MPI

from matensemble.pipeline import Pipeline

N_CHORES = 10
RANKS_PER_CHORE = 50

pipe = Pipeline()


@pipe.chore(
    name="mpi-hello",
    num_tasks=RANKS_PER_CHORE,
    cores_per_task=1,
    gpus_per_task=0,
    mpi=True,
)
def run_mpi_hello(task_id: int):
    """
    One Python chore that runs as a 50-rank MPI job.
    Every MPI rank enters this same function.
    """

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    out_file = f"task_{task_id}_rank_{rank}.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(
            f"Hello, World! I am process {rank} of {size} on {name}. "
            f"task_id={task_id}\n"
        )


def build_workflow() -> None:
    for i in range(1, N_CHORES + 1):
        run_mpi_hello(i)
