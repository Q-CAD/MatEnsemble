from __future__ import annotations

import json

import networkx as nx
import shlex

from pathlib import Path

from matensemble.model import ChoreType, Resources
from matensemble.utils import _json_safe


class Chore:
    """
    A :obj:`Chore` is what MatEnsemble is built around. :obj:`Job`'s can have two
    different types. ``PYTHON`` or ``EXECUTABLE``

    Python chores are delayed function calls that will be submitted to the
    runtime-worker when the :obj:`Chore`'s dependencies are resolved and they are
    schduled in the queue.

    Executable chores are simply commands that will usually call an Executable script
    when the chore is scheduled.

    """

    def __init__(
        self,
        id: str,
        command: str | list[str],
        chore_type: ChoreType,
        resources: Resources,
        workdir: Path,
        func_module: str | None = None,
        func_qualname: str | None = None,
        serialized_callable: bytes | None = None,
        deps: tuple[str, ...] = (),
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> None:
        """
        The constructor for a :obj:`Chore`

        Parameters
        ----------
        id : str
            The ID for the :obj:`Chore`
        command : str, list[str]
            The command that will be run when the :obj:`Chore` is submitted
        chore_type: ChoreType
            Either PYTHON or EXECUTABLE
        resources : Resources
            An instance of :obj:`Resources` that holds all the information about
            what resources are needed to run the :obj:`Chore`
        workdir : Path
            The Path to the directory where the output of the :obj:`Chore` will be
            handled
        func_module : str
            The module where the function definition is if the type of the :obj:`Chore`
            is PYTHON
        func_qualname : str
            The name of the function if the type of the :obj:`Chore` is PYTHON
        serialized_callable : bytes
            The original function that was wrapped stored as bytes
        deps : tuple[str, ...]
            A tupele of chore-id's which results this :obj:`Chore
        args : tuple
            The arguments to give the function if type is PYTHON
        kwargs : dict
            The key-word arguments to give the function if flaovr is PYTHON
        """

        self.id = id
        self.command = (
            shlex.split(command) if isinstance(command, str) else list(command)
        )

        if chore_type == ChoreType.PYTHON:
            if serialized_callable is None and not (func_module and func_qualname):
                raise ValueError(
                    "Python chores require either serialized_callable or func_module+func_qualname"
                )

        self.chore_type = chore_type
        self.resources = resources
        self.workdir = workdir.resolve()
        self.spec_path = self.workdir / "chore.pkl"

        self.func_module = func_module
        self.func_qualname = func_qualname
        self.serialized_callable = serialized_callable
        
        self.deps = deps
        self.args = args
        self.kwargs = {} if kwargs is None else kwargs

    def graph(self) -> nx.DiGraph:
        return nx.DiGraph()

    def _to_debug_dict(self) -> dict:
        return {
            "id": self.id,
            "command": self.command,
            "chore_type": _json_safe(self.chore_type),
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
            "has_serialized_callable": self.serialized_callable is not None,
            "deps": list(self.deps),
            "args": _json_safe(self.args),
            "kwargs": _json_safe(self.kwargs),
        }

    def _write_debug_json(self) -> None:
        """
        The :obj:`Chore` is pickled at runtime to be used later on, but it is also
        written as json for debugging.
        """

        debug_file = self.spec_path.parent / "chore.json"
        debug_file.parent.mkdir(parents=True, exist_ok=True)
        with debug_file.open("w") as f:
            json.dump(self._to_debug_dict(), f, indent=2)

    def __str__(self) -> str:
        """
        Return the :obj:`Chore` as a JSON string.
        """

        return json.dumps(self._to_debug_dict(), indent=2)
