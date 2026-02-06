import numpy as np
import flux.job
import shlex
import os

from pathlib import Path


def normalize_task_args(task_args):
    """
    Ensure that the task_args are of the supported types

    Parameters
    ----------
    task_args: list[int | str | float] | int | str | float | np.int64 | dict
        The arguments for the task
    """

    if isinstance(task_args, list):
        return [str(arg) for arg in task_args]
    if task_args is None:
        return []
    if isinstance(task_args, (str, int, float, np.int64, np.float64, dict)):
        return [str(task_args)]
    raise TypeError(
        f"ERROR: Task argument can not be {type(task_args)}. "
        "Currently supports `list`, `str`, `int`, `float`, `np.int64`, `np.float64`, and `dict` types"
    )


def resolve_workdir(
    task,
    task_directory=None,
    base_out_dir=None,
    launch_dir=None,
) -> Path:
    """
    Decide where the task should run and where stdout/stderr land.
    No cwd changes, just returns an absolute Path that exists.
    """

    launch_dir = Path(launch_dir or os.getcwd())

    if task_directory is not None:
        p = Path(task_directory)
        if not p.is_absolute():
            root = Path(base_out_dir) if base_out_dir is not None else launch_dir
            p = root / p
    else:
        root = Path(base_out_dir) if base_out_dir is not None else launch_dir
        p = root / str(task)

    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


class Fluxlet:
    """
    Wrapper around Flux job submission for a single 'task'.

    A Fluxlet builds a Flux Jobspec for a task,
    applies resource requirements (tasks_per_job, cores_per_task, gpus_per_task),
    and submits it through a FluxExecutor.

    For convenience/debugging, the submitted Future is annotated with metadata
    (e.g., task id, jobspec, workdir), so higher-level orchestration code
    (SuperFluxManager / strategy implementations) can track progress and locate
    task outputs.

    Attributes
    ----------
    flux_handle: flux.Flux()
        A reference to the flux instance that is managing the resources
    future: list[FluxExecutorFuture]
        The future objects representing the completion of tasks
    tasks_per_job: int
        The number of sub-tasks that will go along with a task
    cores_per_task: int
        The number of CPU cores that a task requires
    gpus_per_task: int
        The number of GPUs that a task requires
    """

    def __init__(self, handle, tasks_per_job, cores_per_task, gpus_per_task) -> None:
        """
        Parameters
        ----------
        handle: flux.Flux()
            A reference to the SuperFluxManager's flux handle
        tasks_per_job: int
            The number of sub-tasks that will go along with a task
        cores_per_task: int
            The number of CPU cores that a task requires
        gpus_per_task: int
            The number of GPUs that a task requires

        Return
        ------
        None
        """

        self.flux_handle = handle
        self.future = []
        self.tasks_per_job = tasks_per_job
        self.cores_per_task = cores_per_task
        self.gpus_per_task = gpus_per_task

    def job_submit(
        self,
        executor,
        command,
        task,
        task_args,
        task_directory=None,
        base_out_dir=None,
        set_gpu_affinity=False,
        set_cpu_affinity=True,
        set_mpi=None,
        env=None,
    ) -> flux.job.FluxExecutorFuture:
        """
        Submits a job to a with the given FluxExecutor.

        Parameters
        ----------
        executor: FluxExecutor
            The executor that the task will be submitted to
        command: str
            The command to run the task
        task: str | int
            The task ID
        task_args: list[str | int]
            The arguments of the task
        task_directory: str
            The directory where the results of the task will be placed
        base_out_dir: Path
            The base dir of where the entire workflow's output is
        set_gpu_affinity: bool
            Whether the task can-be/prefer computed on GPUs
        set_cpu_affinity:
            Whether the task perfers the CPU
        set_mpi: bool | None
            Whether the task will use a message passing interface
        env: i really don't know on this one
            I forgot what this is ¯\\(ツ)/¯
        """

        workdir = resolve_workdir(
            task=task,
            task_directory=task_directory,
            base_out_dir=base_out_dir,
            launch_dir=os.getcwd(),
        )

        cmd_list = shlex.split(command)
        cmd_list.extend(normalize_task_args(task_args))

        jobspec = flux.job.JobspecV1.from_command(
            cmd_list,
            num_tasks=int(self.tasks_per_job),
            cores_per_task=self.cores_per_task,
            gpus_per_task=self.gpus_per_task,
        )

        jobspec.cwd = str(workdir)

        if set_mpi is not None:
            jobspec.setattr_shell_option("mpi", "pmi2")
        if set_cpu_affinity:
            jobspec.setattr_shell_option("cpu-affinity", "per-task")
        if set_gpu_affinity and self.gpus_per_task > 0:
            jobspec.setattr_shell_option("gpu-affinity", "per-task")

        job_env = dict(os.environ) if env is None else dict(env)
        jobspec.environment = job_env

        jobspec.stdout = str(workdir / "stdout")
        jobspec.stderr = str(workdir / "stderr")

        self.resources = getattr(jobspec, "resources", None)
        future = executor.submit(jobspec)

        # compatibility with existing code
        future.task_ = task
        future.task = task
        future.job_spec = jobspec
        future.workdir = str(workdir)  # NEW: convenient for debugging/reporting

        self.future = future
        return future

    def hetero_job_submit(
        self,
        executor,
        nnodes,
        gpus_per_node,
        command,
        task,
        task_args,
        task_directory=None,
        base_out_dir=None,  # NEW
        set_gpu_affinity=False,
        set_cpu_affinity=True,
        set_mpi=None,
        env=None,  # NEW
    ):
        """
        Same as job_submit but uses the GPU for redis streaming I think
        """

        workdir = resolve_workdir(
            task=task,
            task_directory=task_directory,
            base_out_dir=base_out_dir,
            launch_dir=os.getcwd(),
        )

        cmd_list = shlex.split(command)
        cmd_list.extend(normalize_task_args(task_args))

        jobspec = flux.job.JobspecV1.per_resource(
            cmd_list,
            ncores=int(self.tasks_per_job),
            nnodes=nnodes,
            gpus_per_node=gpus_per_node,
            per_resource_type="core",
            per_resource_count=1,
        )

        jobspec.cwd = str(workdir)

        if set_mpi is not None:
            jobspec.setattr_shell_option("mpi", "pmi2")
        if set_cpu_affinity:
            jobspec.setattr_shell_option("cpu-affinity", "per-task")
        if set_gpu_affinity and self.gpus_per_task > 0:
            jobspec.setattr_shell_option("gpu-affinity", "per-task")

        job_env = dict(os.environ) if env is None else dict(env)
        # avoid global os.environ mutation (current code does this)
        job_env["SLURM_GPUS_PER_NODE"] = str(gpus_per_node)
        jobspec.environment = job_env

        jobspec.stdout = str(workdir / "stdout")
        jobspec.stderr = str(workdir / "stderr")

        self.resources = getattr(jobspec, "resources", None)
        future = executor.submit(jobspec)

        future.task_ = task
        future.task = task
        future.job_spec = jobspec
        future.workdir = str(workdir)

        self.future = future
        return future
