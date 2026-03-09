from __future__ import annotations

import sys
import json
import functools
import datetime

import networkx as nx
import shlex

from typing import Callable, Any
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import StrEnum, auto
from pathlib import Path

# TODO: Make these imports work when everything is working
from matensemble.MANAGER import Manager
from matensemble.strategy.process_futures_strategy_base import FutureProcessingStrategy


@dataclass(frozen=True)
class OutputReference:
    """
    An object to encapsulate the result of a job as the input to another job
    """

    job_id: str


@dataclass()
class Resources:
    """
    The resources that a wrapped flux job needs
    """

    num_tasks: int = 1
    cores_per_task: int = 1
    gpus_per_task: int = 0
    mpi: bool = False
    env: dict[str, str] | None = None


class JobFlavor(StrEnum):
    """
    The different flavors that a job can be. As of right now there are two types
    of Jobs, Python jobs and Executable jobs. Python jobs are delayed function
    calls and executable jobs are paths to executable files.
    """

    PYTHON = auto()
    EXECUTABLE = auto()


def _find_refs(args, kwargs):
    """
    Find all OutReference objects found anywhere inside args or kwargs
    """

    seen = set()

    def _walk(x):
        obj_id = id(x)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if isinstance(x, OutputReference):
            yield x
            return

        # do not iterate into strings or bytes
        if isinstance(x, (str, bytes, bytearray)):
            return

        # dict like things
        if isinstance(x, Mapping):
            for k, v in x.items():
                yield from _walk(k)
                yield from _walk(v)
            return

        if is_dataclass(x):
            for field_name in getattr(x, "__dataclass_fields__", {}):
                yield from _walk(getattr(x, field_name))
            return

        # iterables
        if isinstance(x, Iterable):
            for item in x:
                yield from _walk(item)
            return

        return

    # walk args and kwargs
    yield from _walk(args)
    yield from _walk(kwargs)


class Job:
    def __init__(
        self,
        id: str,
        command: str,
        flavor: JobFlavor,
        resources: Resources,
        spec_file: Path,
        func_module: str | None = None,
        func_qualname: str | None = None,
        deps: tuple = (),
        args: tuple = (),
        kwargs: dict = {},
    ) -> None:
        self.id = id
        self.command = (
            shlex.split(command) if isinstance(command, str) else list(command)
        )
        self.flavor = flavor
        self.resources = resources
        self.spec_file = spec_file.resolve()
        self.func_module = func_module
        self.func_qualname = func_qualname
        self.deps = deps
        self.args = args
        self.kwargs = kwargs

    def graph(self) -> nx.DiGraph:
        return nx.DiGraph()

    def __str__(self) -> str:
        data = {}
        if self.flavor == JobFlavor.PYTHON:
            data["id"] = self.id
            data["command"] = self.command
            data["flavor"] = self.flavor
            data["resources"] = asdict(self.resources)
            data["spec_file"] = str(self.spec_file)
            data["func_module"] = self.func_module
            data["func_qualname"] = self.func_qualname
            data["deps"] = self.deps
            data["args"] = self.args
            data["kwargs"] = self.kwargs
        else:
            data["id"] = self.id
            data["command"] = self.command
            data["flavor"] = self.flavor
            data["resources"] = asdict(self.resources)
            data["spec_file"] = str(self.spec_file)
            data["deps"] = self.deps
        return json.dumps(data, indent=4)


class Pipeline:
    def __init__(self, basedir: str | None = None) -> None:
        self._counter = 0
        self._job_list: list[Job] = []

        if basedir is None:
            self._base_dir = (
                Path.cwd()
                / f"matensemble_workflow-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        else:
            self._base_dir = (
                Path(basedir)
                / f"matensemble_workflow-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    ) -> Callable[[Callable[..., Any]], Callable[..., OutputReference]]:
        """
        Wrap a function to produce a :obj:`Job` and returns a :obj: `OutReference`
        which other job definitions can use to define dependencies.

        :obj:`Job` objects are delayed function calls that are put into the
        :obj: `Pipeline` and are added into its graph when you run it.
        A PYTHON :obj: `Job` contains meta-data that is needed to reproduce a function.
        The :obj: `SuperFluxManager` will create a :obj: `Fluxlet` and submit
        the job to flux which calls the module :py:mod: `matensemble.piepline.runtime_worker`.
        :py:mod: `matensemble.piepline.runtime_worker` takes in two command line
        arguments which are the :param: `job_id` and :param: `spec_file` which
        the module will use to find the JSON file containing all of the data
        on the job, and it will use it to import the function and call it with
        its respective arguments. The result will then be stored in the flux KVS

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
        >>> @job
        ... def print_message():
        ...     print("I am a Job")
        >>> print_job = print_message()
        >>> type(print_job)
        <class 'jobflow.core.job.Job'>
        >>> print_job.function
        <function print_message at 0x7ff72bdf6af0>

        . . .

        /home/fred/Desktop/github.com/materialsproject/jobflow/core/job.py
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., OutputReference]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> OutputReference:
                self._counter += 1

                res = Resources(
                    num_tasks=num_tasks,
                    cores_per_task=cores_per_task,
                    gpus_per_task=gpus_per_task,
                    mpi=mpi,
                    env=env,
                )

                job_id = (
                    f"job-{name}-{self._counter:04d}"
                    if name
                    else f"job-{func.__qualname__}-{self._counter:04d}"
                )

                spec_file = self._out_dir / job_id
                cmd = f"python -m matensemble.pipeline.runtime_worker --job-id {job_id} --job-dir {spec_file}"

                job = Job(
                    id=job_id,
                    command=cmd,
                    flavor=JobFlavor.PYTHON,
                    resources=res,
                    spec_file=spec_file,
                    func_module=func.__module__,
                    func_qualname=func.__qualname__,
                    args=args,
                    kwargs=kwargs,
                )
                self._job_list.append(job)

                return OutputReference(job_id)

            return wrapper

        return decorator

    def exec(
        self,
        command: str,
        name: str | None = None,
        num_tasks: int = 1,
        cores_per_task: int = 1,
        gpus_per_task: int = 0,
        mpi: bool = False,
        env: dict[str, str] | None = None,
    ) -> Job:
        """
        Wrap a function to produce a :obj:`Job` and returns a :obj: `OutReference`
        which other job definitions can use to define dependencies.

        :obj:`Job` objects are delayed function calls that are put into the
        :obj: `Pipeline` and are added into its graph when you run it.
        A PYTHON :obj: `Job` contains meta-data that is needed to reproduce a function.
        The :obj: `SuperFluxManager` will create a :obj: `Fluxlet` and submit
        the job to flux which calls the module :py:mod: `matensemble.piepline.runtime_worker`.
        :py:mod: `matensemble.piepline.runtime_worker` takes in two command line
        arguments which are the :param: `job_id` and :param: `spec_file` which
        the module will use to find the JSON file containing all of the data
        on the job, and it will use it to import the function and call it with
        its respective arguments. The result will then be stored in the flux KVS

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
        """

        res = Resources(
            num_tasks=num_tasks,
            cores_per_task=cores_per_task,
            gpus_per_task=gpus_per_task,
            mpi=mpi,
            env=env,
        )

        self._counter += 1  # you probably want exec jobs counted too
        job_id = (
            f"job-{name}-{self._counter:04d}"
            if name
            else f"job-exec-{self._counter:04d}"
        )

        job = Job(
            id=job_id,
            command=command,
            flavor=JobFlavor.EXECUTABLE,
            resources=res,
            spec_file=self._out_dir / job_id,
        )
        self._job_list.append(job)  # optional: include exec jobs in DAG
        return job

    def _create_graph(self):
        G = nx.DiGraph()
        for job in self._job_list:
            G.add_node(job.id, job=job)

        for job in self._job_list:
            for ref in _find_refs(job.args, job.kwargs):
                G.add_edge(ref.job_id, job.id)
        return G

    def _sort_graph(self, G: nx.DiGraph) -> list:
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

    def _creat_out_structure(self) -> None:
        pass

    def submit(
        self,
        write_restart_freq: int | None = 100,
        buffer_time: int | None = 1,
        set_cpu_affinity: bool = True,
        set_gpu_affinity: bool = False,
        adaptive: bool = True,
        dynopro: bool = False,
        processing_strategy: FutureProcessingStrategy | None = None,
    ) -> None:
        """
        - Builds the workflow DAG and topologically sorts it
        - Creates the output structure and writes all of the Jobspecs as json
          into thier directories
        - Sends the sorted workflow graph to the manager which runs the workflow
        """
        dag = self._create_graph()
        try:
            workflow_dag = self._sort_graph(dag)
        except Exception as e:
            print(f"Exiting due to Exception: {e}")
            sys.exit(1)
        self._creat_out_structure()

        manager = Manager(
            self._base_dir, write_restart_freq, set_cpu_affinity, set_gpu_affinity
        )
        manager.poolexecutor(
            workflow_dag,
            buffer_time=buffer_time,
            adaptive=adaptive,
            dynopro=dynopro,
            processing_strategy=processing_strategy,
        )

    def graph(self) -> nx.DiGraph:
        return nx.DiGraph()
