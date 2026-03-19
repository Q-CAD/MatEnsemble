from __future__ import annotations

import copy
import functools
import datetime
import sys

import networkx as nx

from typing import Callable, Any
from pathlib import Path

from matensemble.manager import FluxManager
from matensemble.strategy import FutureProcessingStrategy
from matensemble.job import Job
from matensemble.model import OutputReference, Resources, JobFlavor
from matensemble.utils import _collect_dep_ids


class Pipeline:
    def __init__(self, basedir: str | None = None) -> None:
        """
        Parameters
        ----------
        basedir : str, optional
            The root directory of the workflow. Defaults to the current working
            directory
        """

        self._counter = 0
        self._job_list: list[Job] = []

        root = Path.cwd() if basedir is None else Path(basedir)
        self._base_dir = (
            root / f"matensemble_workflow-{datetime.datetime.now():%Y%m%d_%H%M%S}"
        )
        self._out_dir = self._base_dir / "out"

    def job(
        self,
        name: str | None = None,
        num_tasks: int = 1,
        cores_per_task: int = 1,
        gpus_per_task: int = 0,
        mpi: bool = False,
        env: dict[str, str] | None = None,
        inherit_env: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., OutputReference]]:
        """
        Wrap a function to produce a :obj:`Job` and returns a :obj: `OutputReference`
        which other job definitions can use to define dependencies.

        :obj:`Job` objects are delayed function calls that are put into the
        :obj: `Pipeline` and are added into its graph when you run it.
        A PYTHON :obj: `Job` contains meta-data that is needed to reproduce a
        function. The :obj: `FluxManager` will create a :obj: `Fluxlet` and
        submit the job to flux which calls the module :py:mod: `matensemble.runtime_worker`.
        :py:mod: `matensemble.runtime_worker` takes in two command line
        arguments which are the :param: `job_id` and :param: `spec_file` which
        the module will use to find the *pickled* python object containing all
        of the data on the job, and it will use it to import the function and
        call it with its respective arguments. The result will then be stored
        in the flux KVS

        Parameters
        ----------
        name : str, optional
            The name  that will be assigned to the job_id, defaults to the name
            of the function.
        num_tasks : int, optional
            The number of tasks that will be launched with flux, defaults to 1
        cores_per_task : int, optional
            The number of CPU cores that are required to submit the job, defaults
            to 1
        gpus_per_task : int, optional
            The number of GPUs that are required to submit the job, defaults to 0
        mpi : bool, optional
            Whether  or not the job will use the Message Passing Interface I think
            defaults to False
        env : dict[str, str], optional
            The environment varaibles that will be set on job submission, defaults
            to None


        Examples
        --------
        ...
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., OutputReference]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> OutputReference:
                if "<locals>" in func.__qualname__:
                    raise ValueError(
                        "MatEnsemble jobs must wrap importable top-level callables, not nested/local functions."
                    )

                self._counter += 1

                source_root = str(self._base_dir.parent.resolve())
                merged_env = dict(env or {})
                old_pythonpath = merged_env.get("PYTHONPATH")

                if old_pythonpath:
                    merged_env["PYTHONPATH"] = f"{source_root}:{old_pythonpath}"
                else:
                    merged_env["PYTHONPATH"] = source_root

                res = Resources(
                    num_tasks=num_tasks,
                    cores_per_task=cores_per_task,
                    gpus_per_task=gpus_per_task,
                    mpi=mpi,
                    env=merged_env,
                    inherit_env=inherit_env,
                )

                job_id = (
                    f"job-{name}-{self._counter:04d}"
                    if name
                    else f"job-{func.__name__}-{self._counter:04d}"
                )

                workdir = self._out_dir / job_id
                spec_file = workdir / "job.pkl"
                deps = _collect_dep_ids(args, kwargs)

                cmd = [
                    sys.executable,
                    "-m",
                    "matensemble.runtime_worker",
                    "--job-id",
                    job_id,
                    "--spec-file",
                    str(spec_file),
                ]

                job = Job(
                    id=job_id,
                    command=cmd,
                    flavor=JobFlavor.PYTHON,
                    resources=res,
                    workdir=workdir,
                    func_module=func.__module__,
                    func_qualname=func.__qualname__,
                    deps=deps,
                    args=copy.deepcopy(args),
                    kwargs=copy.deepcopy(kwargs),
                )
                self._job_list.append(job)
                return OutputReference(job_id)

            return wrapper

        return decorator

    def exec(
        self,
        command: str | list[str],
        name: str | None = None,
        num_tasks: int = 1,
        cores_per_task: int = 1,
        gpus_per_task: int = 0,
        mpi: bool = False,
        env: dict[str, str] | None = None,
        inherit_env: bool = False,
    ) -> Job:
        """
        Create a :obj:`Job` with a path to an executable rather than a delayed
        python function call.

        Parameters
        ----------
        command : str | list[str]
            The command to be run when the :obj:`Job` runs
        name : str, optional
            The name  that will be assigned to the job_id, defaults to the name
            of the function.
        num_tasks : int, optional
            The number of tasks that will be launched with flux, defaults to 1
        cores_per_task : int, optional
            The number of CPU cores that are required to submit the job, defaults
            to 1
        gpus_per_task : int, optional
            The number of GPUs that are required to submit the job, defaults to 0
        mpi : bool, optional
            Whether  or not the job will use the Message Passing Interface I think
            defaults to False
        env : dict[str, str], optional
            The environment varaibles that will be set on job submission, defaults
            to None
        """

        res = Resources(
            num_tasks=num_tasks,
            cores_per_task=cores_per_task,
            gpus_per_task=gpus_per_task,
            mpi=mpi,
            env=env,
            inherit_env=inherit_env,
        )

        self._counter += 1
        job_id = (
            f"job-{name}-{self._counter:04d}"
            if name
            else f"job-exec-{self._counter:04d}"
        )
        workdir = self._out_dir / job_id

        job = Job(
            id=job_id,
            command=command,
            flavor=JobFlavor.EXECUTABLE,
            resources=res,
            workdir=workdir,
        )
        self._job_list.append(job)
        return job

    def _create_graph(self) -> nx.DiGraph:
        """
        Build the graph to of :obj:`Job`'s with edges representing dependencies

        Return
        ------
        nx.DiGraph
        """

        G = nx.DiGraph()
        known_ids = {job.id for job in self._job_list}

        for job in self._job_list:
            G.add_node(job.id, job=job)

        for job in self._job_list:
            missing = [dep_id for dep_id in job.deps if dep_id not in known_ids]
            if missing:
                raise ValueError(f"Job {job.id} has unknown dependencies: {missing}")
            for dep_id in job.deps:
                G.add_edge(dep_id, job.id)

        return G

    def _sort_graph(self, G: nx.DiGraph) -> list:
        """
        Topologically sorts the graph into a list of :obj:`Job`'s to have the
        least dependent jobs in the front of the list.
        """

        if not nx.is_directed_acyclic_graph(G):
            raise Exception(
                "Error: MatEnsemble workflow graph cannot contain cycles, \
                        topological sort not possible"
            )
        else:
            try:
                return list(nx.topological_sort(G))
            except nx.NetworkXUnfeasible:
                raise Exception(
                    "Error: MatEnsemble workflow graph cannot contain cycles, \
                            cycle found by topological sort"
                )

    def submit(
        self,
        write_restart_freq: int | None = 100,
        buffer_time: float = 1.0,
        set_cpu_affinity: bool = True,
        set_gpu_affinity: bool = False,
        adaptive: bool = True,
        dynopro: bool = False,
        processing_strategy: FutureProcessingStrategy | None = None,
    ) -> None:
        """
        Submit the current number of jobs. Builds the graph, sorts the graph, and
        creates a :obj:`FluxManager` and runs the workflow with that.
        """

        dag = self._create_graph()
        ordered_ids = self._sort_graph(dag)
        ordered_jobs = [dag.nodes[job_id]["job"] for job_id in ordered_ids]

        self._out_dir.mkdir(parents=True, exist_ok=True)

        manager = FluxManager(
            job_list=ordered_jobs,
            base_dir=self._base_dir,
            write_restart_freq=write_restart_freq,
            set_cpu_affinity=set_cpu_affinity,
            set_gpu_affinity=set_gpu_affinity,
        )
        manager.run(
            buffer_time=buffer_time,
            adaptive=adaptive,
            dynopro=dynopro,
            processing_strategy=processing_strategy,
        )

    def graph(self) -> nx.DiGraph:
        return self._create_graph()
