# matensemble/fluxlet.py

from __future__ import annotations

import os
import flux.job
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matensemble.pipeline.compile import TaskSpec


class Fluxlet:
    """
    Flux job submission helper.

    MVP: submit pre-compiled TaskSpec objects (final cmd list + per-task resources).
    """

    def __init__(self, handle) -> None:
        self.flux_handle = handle

    def submit_spec(
        self,
        executor,
        spec: "TaskSpec",
        *,
        set_cpu_affinity: bool = True,
        set_gpu_affinity: bool = True,
    ) -> flux.job.FluxExecutorFuture:
        """
        Submit a compiled TaskSpec to Flux.

        - Uses spec.command directly (already normalized/compiled by Pipeline)
        - Uses spec.resources.{num_tasks, cores_per_task, gpus_per_task}
        - Uses spec.workdir for cwd/stdout/stderr
        - Applies mpi/cpu-affinity/gpu-affinity based on spec.resources and flags
        """
        workdir = Path(spec.workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        jobspec = flux.job.JobspecV1.from_command(
            list(spec.command),
            num_tasks=int(spec.resources.num_tasks),
            cores_per_task=int(spec.resources.cores_per_task),
            gpus_per_task=int(spec.resources.gpus_per_task),
        )

        jobspec.cwd = str(workdir)
        jobspec.stdout = str(workdir / "stdout")
        jobspec.stderr = str(workdir / "stderr")

        # shell options
        if getattr(spec.resources, "mpi", False):
            jobspec.setattr_shell_option("mpi", "pmi2")
        if set_cpu_affinity:
            jobspec.setattr_shell_option("cpu-affinity", "per-task")
        if set_gpu_affinity and int(spec.resources.gpus_per_task) > 0:
            jobspec.setattr_shell_option("gpu-affinity", "per-task")

        # env
        env = dict(os.environ)
        if spec.resources.env:
            env.update(spec.resources.env)
        jobspec.environment = env

        fut = executor.submit(jobspec)

        # annotate for processing strategies
        fut.task = spec.id
        fut.task_ = spec.id
        fut.task_spec = spec
        fut.job_spec = jobspec
        fut.workdir = str(workdir)

        return fut
