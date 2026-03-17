import numpy as np
import time
import os

from matensemble.manager import SuperFluxManager

__author__ = "Soumendu Bagchi"

N_task = 10

# Task IDs in order: 1, 2, 3, ... N_task
task_list = list(range(1, N_task + 1))

# Command / executable path
task_command = os.path.abspath("mpi_helloworld.py")

# Constant tasks_per_job for every task (match original behavior)
# (Use 56 if you want the same as your earlier benchmark)
TASKS_PER_JOB = 50
tasks_per_job = TASKS_PER_JOB * np.ones(N_task, dtype=int)

master = SuperFluxManager(
    task_list,
    task_command,
    write_restart_freq=5,
    tasks_per_job=tasks_per_job,
    cores_per_task=1,
    gpus_per_task=0,
)

# Deterministic args, aligned with task order: 1..N_task
task_arg_list = list(range(1, N_task + 1))

# Run
start_time = time.perf_counter()
master.poolexecutor(task_arg_list=task_arg_list, buffer_time=0.1, task_dir_list=None)
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"Workflow took {elapsed_time:.4f} seconds to run.")
