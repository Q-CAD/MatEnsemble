from __future__ import annotations

import cloudpickle
import json

import networkx as nx
import shlex

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable

from matensemble.model import ChoreType, Resources
from matensemble.utils import _json_safe


def registry_entry_filename(key: str) -> str:
    """Return *key* as a basename-only registry filename or raise ValueError."""
    if key in (".", "..") or not key:
        raise ValueError(f"invalid registry key: {key!r}")
    if "\x00" in key:
        raise ValueError(f"registry key contains null byte: {key!r}")
    if "/" in key or "\\" in key:
        raise ValueError(f"registry key must not contain path separators: {key!r}")
    safe = Path(key).name
    if safe != key:
        raise ValueError(f"registry key must be a single path segment: {key!r}")
    return safe


@dataclass(frozen=True)
class ChoreRegistration:
    """
    A registered Python chore callable and its default execution resources.
    """

    qualname: str
    func: Callable[..., Any]
    resources: Resources
    id_name: str


class ChoreRegistry:
    """
    Registry of Python callable chores.

    The registry is intentionally small: it stores callable implementations,
    validates registry names, and writes the callables to the runtime-worker
    registry directory. :class:`~matensemble.pipeline.Pipeline` uses this
    internally, and advanced users can pass a pre-populated registry to a
    pipeline and create delayed work with ``Pipeline.call(...)``.
    """

    def __init__(self) -> None:
        self._entries: dict[str, ChoreRegistration] = {}

    def register(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        resources: Resources | None = None,
        id_name: str | None = None,
    ) -> ChoreRegistration:
        qualname = name or str(func.__qualname__)
        registry_entry_filename(qualname)
        if qualname in self._entries:
            raise ValueError(f"chore {qualname!r} is already registered")

        entry = ChoreRegistration(
            qualname=qualname,
            func=func,
            resources=resources if resources is not None else Resources(),
            id_name=id_name or name or func.__name__,
        )
        self._entries[qualname] = entry
        return entry

    def chore(
        self,
        name: str | None = None,
        resources: Resources | None = None,
        num_tasks: int = 1,
        cores_per_task: int = 1,
        gpus_per_task: int = 0,
        mpi: bool = False,
        env: dict[str, str] | None = None,
        inherit_env: bool = True,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        if resources is None:
            resources = Resources(
                num_tasks=num_tasks,
                cores_per_task=cores_per_task,
                gpus_per_task=gpus_per_task,
                mpi=mpi,
                env=env,
                inherit_env=inherit_env,
            )

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register(func, name=name, resources=resources)
            return func

        return decorator

    def get(self, name: str) -> ChoreRegistration:
        try:
            return self._entries[name]
        except KeyError as e:
            raise KeyError(f"chore {name!r} is not registered") from e

    def write(self, registry_dir: Path) -> None:
        registry_dir.mkdir(parents=True, exist_ok=True)
        for key, entry in self._entries.items():
            safe_name = registry_entry_filename(key)
            with open(registry_dir / safe_name, "wb") as file:
                cloudpickle.dump(entry.func, file)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __getitem__(self, name: str) -> Callable[..., Any]:
        return self.get(name).func

    def __iter__(self):
        return iter(self._entries)

    def keys(self):
        return self._entries.keys()

    def items(self):
        return ((key, entry.func) for key, entry in self._entries.items())

    def values(self):
        return (entry.func for entry in self._entries.values())


class ChoreSpec:
    """
    The specification of a :obj:`Chore`

    Holds the arguments, keyword
    arguments and the name of the chore that you want those arguments to be
    passed to. This is class is used by the user when creating a UserStrategy
    that does processing on completed chores and can spawn new chores.
    """

    def __init__(
        self,
        args,
        kwargs,
        qualname,
        resources: Resources,
        nice: int = 0,
    ) -> None:
        self.args = args
        self.kwargs = kwargs
        self.qualname = qualname
        self.resources = resources
        if not isinstance(nice, int):
            raise TypeError("nice must be an int")
        self.nice = nice


class Chore:
    """
    A :obj:`Chore` is what MatEnsemble is built around. :obj:`Job`'s can have two
    different types. ``PYTHON`` or ``EXECUTABLE``

    Python chores are delayed function calls that will be submitted to the
    runtime-worker when the :obj:`Chore`'s dependencies are resolved and they are
    scheduled in the queue.

    Executable chores are simply commands that will usually call an Executable script
    when the chore is scheduled.

    """

    def __init__(
        self,
        id: str,
        workdir: Path,
        command: str | list[str],
        chore_type: ChoreType | int,
        resources: Resources,
        chore_qualname: str | None = None,
        deps: tuple[str, ...] = (),
        args: tuple = (),
        kwargs: dict | None = None,
        dynopro_args: dict[str, tuple] | None = None,
        dynopro_kwargs: dict[str, dict] | None = None,
        nnodes: int | None = None,
        nice: int = 0,
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
            A tuple of chore IDs whose results this :obj:`Chore` depends on
        args : tuple
            The arguments to give the function if type is PYTHON
        kwargs : dict
            The keyword arguments to give the function if flavor is PYTHON
        dynopro_args : dict, optional
            Per-registered-subprocess positional arguments for dynopro chores.
        dynopro_kwargs : dict, optional
            Per-registered-subprocess keyword arguments for dynopro chores.
        nnodes : int, optional
            When set, this chore will be scheduled via ``per_resource`` and will
            occupy *nnodes* whole nodes (all cores and all GPUs on each node).
            The manager uses this to compute the true resource footprint instead
            of ``num_tasks * cores_per_task`` / ``num_tasks * gpus_per_task``.
            Leave as ``None`` for normal ``from_command`` chores.
        """

        self.id = id
        self.command = (
            shlex.split(command) if isinstance(command, str) else list(command)
        )

        self.chore_type = chore_type
        self.resources = resources
        self.workdir = workdir.resolve()
        self.spec_path = self.workdir / "chore.pickle"

        self.chore_qualname = chore_qualname
        self.deps = deps
        self.args = args
        self.kwargs = {} if kwargs is None else kwargs
        self.dynopro_args = {} if dynopro_args is None else dynopro_args
        self.dynopro_kwargs = {} if dynopro_kwargs is None else dynopro_kwargs
        self.nnodes = nnodes
        if not isinstance(nice, int):
            raise TypeError("nice must be an int")
        self.nice = nice

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
            "chore_qualname": self.chore_qualname,
            "deps": list(self.deps),
            "args": _json_safe(self.args),
            "kwargs": _json_safe(self.kwargs),
            "dynopro_args": _json_safe(self.dynopro_args),
            "dynopro_kwargs": _json_safe(self.dynopro_kwargs),
            "nnodes": self.nnodes,
            "nice": self.nice,
        }

    def _write_metadata(self) -> None:
        """
        The :obj:`Chore` is pickled at runtime to be used later on, but it is also
        written as json for debugging.
        """

        debug_file = self.spec_path.parent / "metadata.json"
        debug_file.parent.mkdir(parents=True, exist_ok=True)
        with debug_file.open("w") as f:
            json.dump(self._to_debug_dict(), f, indent=2)

    def __str__(self) -> str:
        """
        Return the :obj:`Chore` as a JSON string.
        """

        return json.dumps(self._to_debug_dict(), indent=2)
