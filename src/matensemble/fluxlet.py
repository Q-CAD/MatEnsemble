import os
import pickle
import json
import tempfile

import flux
import flux.job

from pathlib import Path
from matensemble.job import Job


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

        job.workdir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            "wb", dir=job.spec_path.parent, delete=False
        ) as tf:
            pickle.dump(job, tf)
            temp_name = tf.name
        os.replace(temp_name, job.spec_path)

        # helpful for debugging
        job._write_debug_json()

        jobspec.cwd = str(job.workdir)
        jobspec.stdout = str(job.workdir / "stdout")
        jobspec.stderr = str(job.workdir / "stderr")

        if job.resources.mpi:
            jobspec.setattr_shell_option("mpi", "pmi2")
        if set_cpu_affinity:
            jobspec.setattr_shell_option("cpu-affinity", "per-task")
        if set_gpu_affinity and job.resources.gpus_per_task > 0:
            jobspec.setattr_shell_option("gpu-affinity", "per-task")

        if job.resources.env:
            jobspec.env = job.resources.env

        # only set this if you truly want every job to span a fixed node count
        if nnodes is not None:
            jobspec.num_nodes = nnodes

        fut = executor.submit(jobspec)
        fut.job_id = job.id
        fut.job_obj = job
        fut.job_spec = jobspec
        fut.workdir = str(job.workdir)
        return fut
