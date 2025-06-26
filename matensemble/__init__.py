

import os
import numpy as np
import flux.job

class Fluxlet:
    def __init__(self, handle, tasks_per_job, cores_per_task, gpus_per_task):
        self.flux_handle = handle
        self.future = []
        self.tasks_per_job = tasks_per_job
        self.cores_per_task = cores_per_task
        self.gpus_per_task = gpus_per_task

    def job_submit(self, executor, command, task, task_args, \
                        task_directory=None, set_gpu_affinity=False,\
                        set_cpu_affinity=True, set_mpi=None):

        launch_dir = os.getcwd()
        cmd_list = command.split(" ")

        if task_directory is not None:
            try:
                os.chdir(os.path.abspath(task_directory))
            except:
                print(f"Could not find task directory {task_directory}: So, creating one instead . . .")
                os.mkdir(os.path.abspath(task_directory))
                os.chdir(task_directory)
        else:
            print("No directories are specified for the task. Task-list will serve as directory tree.")
            try:
                os.chdir(str(task))
            except:
                os.mkdir(str(task))
                os.chdir(str(task))

        print(os.getcwd())
        if isinstance(task_args, list):
            str_args = [str(arg) for arg in task_args]
        elif task_args is None:
            str_args = []
        elif isinstance(task_args, (str, int, float, np.int64, np.float64, dict)):
            str_args = [str(task_args)]
        else:
            raise TypeError(f"ERROR: Task argument can not be {type(task_args)}. Currently supports `list`, `str`, `int`, `float`, `np.int64`, `np.float64`, and `dict` types")

        cmd_list.extend(str_args)

        jobspec = flux.job.JobspecV1.from_command(
            cmd_list,
            num_tasks=int(self.tasks_per_job),
            cores_per_task=self.cores_per_task,
            gpus_per_task=self.gpus_per_task
        )

        jobspec.cwd = os.getcwd()

        if set_mpi is not None:
                jobspec.setattr_shell_option("mpi", "pmi2")
        if set_cpu_affinity:
                jobspec.setattr_shell_option("cpu-affinity", "per-task")
        if set_gpu_affinity and self.gpus_per_task > 0:
            jobspec.setattr_shell_option("gpu-affinity", "per-task")
        jobspec.environment = dict(os.environ)

        jobspec.stdout = os.path.join(os.getcwd(), 'stdout')
        jobspec.stderr = os.path.join(os.getcwd(), 'stderr')

        self.resources = getattr(jobspec, 'resources', None)
        self.future = executor.submit(jobspec)
        self.future.task_ = task
        os.chdir(launch_dir)