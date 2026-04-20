import os
import pickle
import json
import tempfile

import flux
import flux.job

from pathlib import Path
from matensemble.chore import Chore
from matensemble.model import ChoreType


class Fluxlet:
    """
    A class that encapsulates the launching of flux jobs.

    Attributes
    ----------
    handle : flux.Flux
        A flux handle to be used to submit chores
    """

    def __init__(
        self,
        handle: flux.Flux,
    ) -> None:
        self.handle = handle
        self.gpus_per_node = self.get_gpus_per_node()

    def get_gpus_per_node(self) -> tuple[int, int]:
        """
        Get the available nodes and gpus and calculate the number of
        GPUs per node.
        """

        # drain broker rank first, then measure what is actually usable
        self.handle.rpc("resource.drain", {"targets": "0"}).get()

        resources = flux.resource.list.resource_list(self.handle).get()
        nnodes = len(resources.free.ranks)
        total_gpus = resources.free.ngpus

        gpus_per_node = total_gpus // nnodes
        return nnodes, gpus_per_node

    def submit(
        self,
        executor: flux.job.FluxExecutor,
        chore: Chore,
        set_cpu_affinity: bool | None = None,
        set_gpu_affinity: bool | None = None,
        nnodes: int | None = None,
        dynopro: bool | None = None,
    ) -> flux.job.FluxExecutorFuture:
        """
        Creates a :obj:`Jobspec` useing the a :obj:`Job`. Submits the :obj:`Jobspec`
        to flux and adds some metadata to the future object that is returned.

        Parameters
        ----------
        executor : flux.job.FluxExecutor
            The :obj:`FluxExecutor` to use to submit the flux job
        chore : Chore
            The :obj:`Chore` to be submitted to flux.
        set_cpu_affinity : bool, optional
            Whether cpu-affinity should be set in the :obj:`Jobspec`. Defaults
            to None.
        set_gpu_affinity : bool, optional
            Whether gpu-affinity should be set in the :obj:`Jobspec`. Defaults
            to None.
        nnodes : int, optional
            The number of nodes that the given chore needs to be able to run. Defaults
            to None.

        Returns
        -------
        flux.job.FluxExecutorFuture
        """

        if dynopro:
            jobspec = flux.job.JobspecV1.per_resource(
                chore.command,
                ncores=chore.resources.num_tasks,
                nnodes=nnodes,
                gpus_per_node=self.gpus_per_node,
                per_resource_type="core",
                per_resource_count=1,
            )

            chore.workdir.mkdir(parents=True, exist_ok=True)

            jobspec.cwd = str(chore.workdir)
            jobspec.stdout = str(chore.workdir / "stdout")
            jobspec.stderr = str(chore.workdir / "stderr")

            if chore.resources.mpi:
                jobspec.setattr_shell_option("mpi", "pmi2")
            if set_cpu_affinity:
                jobspec.setattr_shell_option("cpu-affinity", "per-task")
            if set_gpu_affinity and chore.resources.gpus_per_task > 0:
                jobspec.setattr_shell_option("gpu-affinity", "per-task")

            base_env = os.environ.copy() if chore.resources.inherit_env else {}
            base_env.update(chore.resources.env or {})
            base_env["SLURM_GPUS_PER_NODE"] = str(self.gpus_per_node)
            jobspec.environment = base_env

            fut = executor.submit(jobspec)

            fut = executor.submit(jobspec)
            fut.chore_id = chore.id
            fut.chore_obj = chore
            fut.chore_spec = jobspec
            fut.workdir = str(chore.workdir)
            return fut
        else:
            jobspec = flux.job.JobspecV1.from_command(
                chore.command,
                chore.resources.num_tasks,
                chore.resources.cores_per_task,
                chore.resources.gpus_per_task,
            )

            chore.workdir.mkdir(parents=True, exist_ok=True)

            if chore.chore_type is ChoreType.PYTHON:
                with tempfile.NamedTemporaryFile(
                    "wb", dir=chore.spec_path.parent, delete=False
                ) as tf:
                    pickle.dump(chore, tf)
                    temp_name = tf.name
                os.replace(temp_name, chore.spec_path)

            jobspec.cwd = str(chore.workdir)
            jobspec.stdout = str(chore.workdir / "stdout")
            jobspec.stderr = str(chore.workdir / "stderr")

            if chore.resources.mpi:
                jobspec.setattr_shell_option("mpi", "pmi2")
            if set_cpu_affinity:
                jobspec.setattr_shell_option("cpu-affinity", "per-task")
            if set_gpu_affinity and chore.resources.gpus_per_task > 0:
                jobspec.setattr_shell_option("gpu-affinity", "per-task")

            base_env = os.environ.copy() if chore.resources.inherit_env else {}
            base_env.update(chore.resources.env or {})
            jobspec.environment = base_env

            # helpful for debugging
            chore._write_debug_json()

            # only set this if you truly want every chore to span a fixed node count
            if nnodes is not None:
                jobspec.num_nodes = nnodes

            fut = executor.submit(jobspec)
            fut.chore_id = chore.id
            fut.chore_obj = chore
            fut.chore_spec = jobspec
            fut.workdir = str(chore.workdir)
            return fut
