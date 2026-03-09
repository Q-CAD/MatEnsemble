import os
import json
import tempfile

import flux
import flux.job

from pathlib import Path
from matensemble.pipeline.PIPELINE import Job


class Fluxlet:
    def __init__(
        self,
        handle: flux.Flux,
    ) -> None:
        self.handle = handle

    def submit(
        self,
        executor: flux.job.FluxExecutor,
        job: Job,
        set_cpu_affinity: bool | None = None,
        set_gpu_affinity: bool | None = None,
        nnodes: int | None = None,
    ) -> flux.job.FluxExecutorFuture:

        jobspec = flux.job.JobspecV1.from_command(
            job.command,
            job.resources.num_tasks,
            job.resources.cores_per_task,
            job.resources.gpus_per_task,
        )

        # create workflow dir only when needed
        job.spec_file.parent.mkdir(parents=True, exist_ok=True)
        job.spec_file.touch()

        # atomically write the spec file to avoid partial writes
        with tempfile.NamedTemporaryFile(
            "w", dir=job.spec_file.parent, delete=False
        ) as tf:
            tf.write(job.__str__())
            temp_name = tf.name
        os.replace(temp_name, job.spec_file)

        jobspec.cwd = str(job.spec_file.parent)
        jobspec.stdout = str(job.spec_file.parent / "stdout")
        jobspec.stderr = str(job.spec_file.parent / "stderr")

        if job.resources.mpi:
            jobspec.setattr_shell_option("mpi", "pmi2")
        if set_cpu_affinity:
            jobspec.setattr_shell_option("cpu-affinity", "per-task")
        if set_gpu_affinity and job.resources.gpus_per_task > 0:
            jobspec.setattr_shell_option("gpu-affinity", "per-task")

        if job.resources.env:
            jobspec.env = job.resources.env

        if nnodes:
            jobspec.num_nodes = nnodes

        fut = executor.submit(jobspec)

        fut.job_obj = job
        fut.job_spec = jobspec
        fut.workdir = str(job.spec_file)

        return fut
