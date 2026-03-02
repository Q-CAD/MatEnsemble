from pathlib import Path
import flux
import flux.job

from matensemble.pipeline.PIPELINE import Job


class Fluxlet:
    def __init__(self, handle: flux.Flux) -> None:
        self.handle = handle

    def submit(
        self,
        executor: flux.job.FluxExecutor,
        job: Job,
        workdir: Path,
        *,
        num_nodes: int | None = None,
        set_cpu_affinity: bool = True,
        set_gpu_affinity: bool = False,
    ) -> flux.job.FluxExecutorFuture:

        jobspec = flux.job.JobspecV1.from_command(
            job.command,
            job.resources.num_tasks,
            job.resources.cores_per_task,
            job.resources.gpus_per_task,
        )

        jobspec.cwd = str(workdir)
        jobspec.stdout = str(workdir / "stdout")
        jobspec.stderr = str(workdir / "stderr")

        if job.resources.mpi:
            jobspec.setattr_shell_option("mpi", "pmi2")
        if set_cpu_affinity:
            jobspec.setattr_shell_option("cpu-affinity", "per-task")
        if set_gpu_affinity and job.resources.gpus_per_task > 0:
            jobspec.setattr_shell_option("gpu-affinity", "per-task")

        if job.resources.env:
            jobspec.env = job.resources.env

        if num_nodes:
            jobspec.num_nodes = num_nodes

        fut = executor.submit(jobspec)

        fut.job_obj = job
        fut.job_spec = jobspec
        fut.workdir = str(workdir)

        return fut


"""
 classmethod from_command(command, num_tasks=1, cores_per_task=1, gpus_per_task=None, num_nodes=None, exclusive=False, duration=None, environment=None, env_expand=None, cwd=None, rlimits=None, name=None, input=None, output=None, error=None, label_io=False, unbuffered=False, queue=None, bank=None) Factory function that builds the minimum legal v1 jobspec. Parameters command (iterable of str) -- command to execute num_tasks (int) -- number of tasks to create cores_per_task (int) -- number of cores to allocate per task gpus_per_task (int) -- number of GPUs to allocate per task
            num_nodes (int) -- distribute allocated tasks across N individual nodes.
            exclusive (bool) -- always allocate nodes exclusively
            duration (Number, str) -- assign a time limit to the job in Flux Standard Duration (if str), datetime.timedelta or Number in seconds. If not provided then the duration will unlimited unless set via the duration setter.
            environment (Mapping) -- Set the environment for the job via a mapping of environment variable name to value. If not provided then the environment will be initialized using os.environ.
            env_expand (Mapping) -- A mapping of environment variables that contain mustache templates to be expanded by the job shell at runtime. (See the flux-run(1) MUSTACHE TEMPLATES section for more info)
            rlimits (Mapping) -- Set process resource limits for the job via a mapping of limit name to value, where a value of -1 is taken as unlimited. E.g. {"nofile": 12000}.
            cwd (str) -- Set the current working directory for the job. If unset, then a working directory may be set using the cwd setter.
            name (str) -- Set a job name.
            input (str, os.PathLike) -- Set job input to a file path.
            output (str, os.PathLike) -- Set job output to a file path. stderr will be copied to the same path as stdout by default unless it is set separately.
            error -- (str, os.PathLike): Set job stderr to a file path.
            label_io (bool) -- For file output, label output with the source task ids. Default is False.
            unbuffered (bool) -- Disable output buffering as much as practical.
            queue (str) -- Set the queue for the job.
            bank (str) -- Set the bank for the job.
"""
