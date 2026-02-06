from matensemble.manager import SuperFluxManager
import numpy as np
import os

__author__ = "Soumendu Bagchi"


def get_random_task_args(N_task):
    from random import choice
    from string import ascii_lowercase, digits

    chars = ascii_lowercase + digits
    lst = ["".join(choice(chars) for _ in range(2)) for _ in range(N_task)]
    return lst


# create a list of task indicators; In the following I use integers as task-IDs.

N_task = 10
task_list = list(np.arange(N_task))

# spceify the basic command/executable-filepath used to execute the task (you can skip any mpirun/srun prefixes, and also any *args, **kwargs at this point)

task_command = os.path.abspath(
    "gpu_hello"
)  #'sample_amd' #'mpi_helloworld.py' #make sure to make it executable by `chmod u+x <file.py>`

# task_command = os.path.abspath("sample_amd")

# Now instatiate a task_manager object which is a Superflux Manager sitting on top of evey smaller Fluxlets

master = SuperFluxManager(
    task_list,
    task_command,
    write_restart_freq=5,
    tasks_per_job=1 * np.ones((N_task, 1)),
    cores_per_task=1,
    gpus_per_task=1,
)


# Input argument list specific to each task in the sorted as task_ids
# Generated random strings to serve as arguments

task_arg_list = list(
    np.random.randint(N_task, size=N_task)
)  # get_random_task_args(N_task)

# For multiple args per task each if the elements could be a list i.e. task_args_list = [['x0f','x14'],['xa9','xf3'],[]...]
# finally execute the whole pool of tasks
master.poolexecutor(task_arg_list=task_arg_list, buffer_time=1, task_dir_list=None)
