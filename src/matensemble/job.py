from __future__ import annotations

import json

import networkx as nx
import shlex

from pathlib import Path

from matensemble.model import JobFlavor, Resources
from matensemble.utils import _json_safe


class Job:
    """
    A :obj:`Job` is what MatEnsemble is built around. :obj:`Job`'s can have two
    different flavors. ``PYTHON`` or ``EXECUTABLE``

    Python jobs are delayed function calls that will be submitted to the
    runtime-worker when the :obj:`Job`'s dependencies are resolved and they are
    schduled in the queue.

    Executable jobs are simply commands that will usually call an Executable script
    when the job is scheduled.

    """

    def __init__(
        self,
        id: str,
        command: str | list[str],
        flavor: JobFlavor,
        resources: Resources,
        workdir: Path,
        func_module: str | None = None,
        func_qualname: str | None = None,
        deps: tuple[str, ...] = (),
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> None:
        """
        The constructor for a :obj:`Job`

        Parameters
        ----------
        id : str
            The ID for the :obj:`Job`
        command : str, list[str]
            The command that will be run when the :obj:`Job` is submitted
        flavor : JobFlavor
            Either PYTHON or EXECUTABLE
        resources : Resources
            An instance of :obj:`Resources` that holds all the information about
            what resources are needed to run the :obj:`Job`
        workdir : Path
            The Path to the directory where the output of the :obj:`Job` will be
            handled
        func_module : str
            The module where the function definition is if the flavor of the :obj:`Job`
            is PYTHON
        func_qualname : str
            The name of the function if the flavor of the :obj:`Job` is PYTHON
        deps : tuple[str, ...]
            A tupele of job-id's which results this :obj:`Job` depends on
        args : tuple
            The arguments to give the function if flavor is PYTHON
        kwargs : dict
            The key-word arguments to give the function if flaovr is PYTHON
        """

        self.id = id
        self.command = (
            shlex.split(command) if isinstance(command, str) else list(command)
        )

        if flavor == JobFlavor.PYTHON:
            if not func_module:
                raise ValueError("Python jobs require importable func_module")
            if not func_qualname:
                raise ValueError("Python jobs require func_qualname")

        self.flavor = flavor
        self.resources = resources
        self.workdir = workdir.resolve()
        self.spec_path = self.workdir / "job.pkl"
        self.func_module = func_module
        self.func_qualname = func_qualname
        self.deps = deps
        self.args = args
        self.kwargs = {} if kwargs is None else kwargs

    def graph(self) -> nx.DiGraph:
        return nx.DiGraph()

    def _to_debug_dict(self) -> dict:
        return {
            "id": self.id,
            "command": self.command,
            "flavor": _json_safe(self.flavor),
            "resources": {
                "num_tasks": self.resources.num_tasks,
                "cores_per_task": self.resources.cores_per_task,
                "gpus_per_task": self.resources.gpus_per_task,
                "mpi": self.resources.mpi,
                "env": _json_safe(self.resources.env),
                "inherit_env": self.resources.inherit_env,
            },
            "spec_file": str(self.spec_path),
            "func_module": self.func_module,
            "func_qualname": self.func_qualname,
            "deps": list(self.deps),
            "args": _json_safe(self.args),
            "kwargs": _json_safe(self.kwargs),
        }

    def _write_debug_json(self) -> None:
        """
        The :obj:`Job` is pickled at runtime to be used later on, but it is also
        written as json for debugging.
        """

        debug_file = self.spec_path.parent / "job.json"
        debug_file.parent.mkdir(parents=True, exist_ok=True)
        with debug_file.open("w") as f:
            json.dump(self._to_debug_dict(), f, indent=2)

    def __str__(self) -> str:
        """
        Return the :obj:`Job` as a JSON string.
        """

        return json.dumps(self._to_debug_dict(), indent=2)
