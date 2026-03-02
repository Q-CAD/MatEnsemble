import os
import functools
import networkx as nx
import flux.job

from pathlib import Path
from typing import Callable
from dataclasses import dataclass
from enum import Enum
from collections.abc import Iterable, Mapping
from dataclasses import is_dataclass


@dataclass(frozen=True)
class OutReference:
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


class JobFlavor(Enum):
    """
    The different flavors that a job can be. As of right now there are two types
    of Jobs, Python jobs and Executable jobs. Python jobs are delayed function
    calls and executable jobs are paths to executable files.
    """

    PYTHON = 1
    EXECUTABLE = 2


def find_refs(args, kwargs):
    """
    Find all OutReference objects found anywhere inside args or kwargs
    """

    seen = set()

    def _walk(x):
        obj_id = id(x)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if isinstance(x, OutReference):
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
        workdir: Path,
        func_module: str | None = None,
        func_qualname: str | None = None,
        deps: tuple | None = None,
        args: tuple | None = None,
        kwargs: dict | None = None,
    ) -> None:
        self.id = id
        self.command = command.split()
        self.flavor = flavor
        self.resources = resources
        self.workdir = str(workdir.resolve())
        self.func_module = func_module
        self.func_qualname = func_qualname
        self.deps = deps
        self.args = args
        self.kwargs = kwargs

    def graph(self) -> nx.DiGraph:
        return nx.DiGraph()


class Pipeline:
    def __init__(self, basedir: str) -> None:
        self._counter = 1
        self._job_list = []

    def job(
        self,
        name: str | None,
        num_tasks: int = 1,
        cores_per_task: int = 1,
        gpus_per_task: int = 0,
        mpi: bool = False,
        env: dict[str, str] | None = None,
    ) -> Callable[..., OutReference]:
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # grab original (unwrapped) function right now
                original = wrapper.__wrapped__  # == func
                self._counter += 1

                res = Resources(
                    num_tasks=num_tasks,
                    cores_per_task=cores_per_task,
                    gpus_per_task=gpus_per_task,
                    mpi=mpi,
                    env=env,
                )

                if name:
                    job_id = f"job-{name}-{self._counter:04d}"
                else:
                    job_id = f"job-{original.__qualname__}-{self._counter:04d}"

                # create a Job instead of running func
                job = Job(
                    id=job_id,
                    flavor=JobFlavor.PYTHON,
                    resources=res,
                    func_module=original.__module__,
                    func_qualname=original.__qualname__,
                    args=args,
                    kwargs=kwargs,
                )
                self._job_list.append(job)

                # return a reference
                return OutReference(job_id)

            return wrapper

        return decorator

    def exec(
        self,
        command: str,
        name: str | None,
        num_tasks: int = 1,
        cores_per_task: int = 1,
        gpus_per_task: int = 0,
        mpi: bool = False,
        env: dict[str, str] | None = None,
    ) -> Job:
        res = Resources(
            num_tasks=num_tasks,
            cores_per_task=cores_per_task,
            gpus_per_task=gpus_per_task,
            mpi=mpi,
            env=env,
        )
        if name:
            job_id = f"job-{name}-{self._counter:04d}"
        else:
            job_id = f"job-exec-{self._counter:04d}"
        return Job(
            id=job_id, command=command, flavor=JobFlavor.EXECUTABLE, resources=res
        )

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
                "Error: The graph contains a cycle. A topological sort is not possible. MatEnsemble Workflow cannot contain cycles"
            )
        else:
            # Perform the topological sort
            # nx.topological_sort returns a generator, so convert to a list to view the order
            try:
                return list(nx.topological_sort(G))
            except nx.NetworkXUnfeasible:
                # This exception is also raised by topological_sort if a cycle is found
                raise Exception(
                    "Error: The graph contains a cycle (caught by topological_sort). MatEnsemble workflow cannot contain cycles"
                )

    def submit(self) -> None:
        pass

    def graph(self) -> nx.DiGraph:
        pass
