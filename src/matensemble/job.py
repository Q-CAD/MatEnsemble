from __future__ import annotations

import json

import networkx as nx
import shlex

from pathlib import Path

from matensemble.model import JobFlavor, Resources
from matensemble.utils import _json_safe


class Job:
    """
    as;dlkfjas;dlkfj;x
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
        debug_file = self.spec_path.parent / "job.json"
        debug_file.parent.mkdir(parents=True, exist_ok=True)
        with debug_file.open("w") as f:
            json.dump(self._to_debug_dict(), f, indent=2)

    def __str__(self) -> str:
        return json.dumps(self._to_debug_dict(), indent=2)
