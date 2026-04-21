from __future__ import annotations

import copy
import functools
import datetime
import sys

import cloudpickle
import networkx as nx

from typing import Callable, Any
from pathlib import Path

from matensemble.manager import FluxManager
from matensemble.strategy import FutureProcessingStrategy
from matensemble.chore import Chore
from matensemble.model import OutputReference, Resources, ChoreType
from matensemble.utils import _collect_dep_ids


class Pipeline:
    """
    Build and submit a MatEnsemble workflow as a directed acyclic graph (DAG)
    of delayed chores.

    A :obj:`Pipeline` is the main user-facing workflow builder in MatEnsemble.
    It collects chores without executing them immediately, tracks dependencies
    between them, and later submits the fully constructed workflow to Flux
    through a :obj:`FluxManager`.

    There are two main ways to add work to a pipeline:

    #. ``Pipeline.chore(...)``
       Wraps a top-level Python function and turns each call to that function
       into a delayed ``Chore`` object instead of executing it immediately.
       The call returns an ``OutputReference`` placeholder that can be passed
       into later chores to define dependencies between tasks.

    #. ``Pipeline.exec(...)``
       Adds an executable or shell-style command as a ``Chore`` for non-Python
       tasks.

    For Python chores, the pipeline records enough metadata to reproduce the
    original function call later, including:

    - the module where the function is defined
    - the function's qualified name
    - the positional and keyword arguments
    - the chore's resource requirements
    - any upstream dependencies discovered from ``OutputReference`` objects

    When ``submit()`` is called, the pipeline:

    - builds a DAG from the collected chores
    - validates that all dependencies refer to known chores
    - checks that the workflow is acyclic
    - topologically sorts the chores into dependency order
    - creates a :obj:`FluxManager` to launch the workflow

    For Python chores, the manager will submit a Flux job whose command runs
    ``matensemble.runtime_worker``. That worker loads the serialized ``Chore``
    specification from disk, imports the recorded module, resolves the target
    function by qualified name, replaces dependency references with their
    concrete upstream results, and calls the function
    """

    def __init__(self, basedir: str | None = None) -> None:
        """
        Parameters
        ----------
        basedir : str, optional
            The root directory of the workflow. Defaults to the current working
            directory
        """

        self._counter = 0
        self._chore_list: list[Chore] = []

        root = Path.cwd() if basedir is None else Path(basedir)
        self._base_dir = (
            root / f"matensemble_workflow-{datetime.datetime.now():%Y%m%d_%H%M%S}"
        )
        self._out_dir = self._base_dir / "out"

    def chore(
        self,
        name: str | None = None,
        num_tasks: int = 1,
        cores_per_task: int = 1,
        gpus_per_task: int = 0,
        mpi: bool = False,
        env: dict[str, str] | None = None,
        inherit_env: bool = True,
    ) -> Callable[[Callable[..., Any]], Callable[..., OutputReference]]:
        """
        Wrap a function to produce a :obj:`Chore` and returns a :obj: `OutputReference`
        which other chore definitions can use to define dependencies.

        :obj:`Chore` objects are delayed function calls that are put into the
        :obj: `Pipeline` and are added into its graph when you run it.
        A PYTHON :obj: `Chore` contains meta-data that is needed to reproduce a
        function. The :obj: `FluxManager` will create a :obj: `Fluxlet` and
        submit the chore to flux which calls the module :py:mod: `matensemble.runtime_worker`.
        :py:mod: `matensemble.runtime_worker` takes in two command line
        arguments which are the :param: `chore_id` and :param: `spec_file` which
        the module will use to find the *pickled* python object containing all
        of the data on the chore, and it will use it to import the function and
        call it with its respective arguments. The result will then be stored
        in the flux KVS

        Parameters
        ----------
        name : str, optional
            The name  that will be assigned to the chore_id, defaults to the name
            of the function.
        num_tasks : int, optional
            The number of tasks that will be launched with flux, defaults to 1
        cores_per_task : int, optional
            The number of CPU cores that are required to submit the chore, defaults
            to 1
        gpus_per_task : int, optional
            The number of GPUs that are required to submit the chore, defaults to 0
        mpi : bool, optional
            When True, sets Flux shell option ``mpi=pmi2`` on the chorespec (default False).
        env : dict[str, str], optional
            Extra environment variables for the task. For Python chores,
            ``PYTHONPATH`` is merged to include the workflow parent directory.
        inherit_env : bool
            If True (default), the Flux jobspec starts from the submitting process
            environment and applies ``env`` overrides.

        Returns
        -------
        Callable
            A decorator that returns a wrapped function; each call to the wrapper
            enqueues a Python chore and returns :class:`~matensemble.model.OutputReference`.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., OutputReference]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> OutputReference:
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

                chore_id = (
                    f"chore-{name}-{self._counter:04d}"
                    if name
                    else f"chore-{func.__name__}-{self._counter:04d}"
                )

                workdir = self._out_dir / chore_id
                spec_file = workdir / "chore.pkl"
                deps = _collect_dep_ids(args, kwargs)

                cmd = [
                    sys.executable,
                    "-m",
                    "matensemble.runtime_worker",
                    "--chore-id",
                    chore_id,
                    "--spec-file",
                    str(spec_file),
                ]

                needs_serialization = (
                    func.__module__ == "__main__" or "<locals>" in func.__qualname__
                )

                serialized_callable = (
                    cloudpickle.dumps(func) if needs_serialization else None
                )

                chore = Chore(
                    id=chore_id,
                    command=cmd,
                    chore_type=ChoreType.PYTHON,
                    resources=res,
                    workdir=workdir,
                    func_module=None if needs_serialization else func.__module__,
                    func_qualname=None if needs_serialization else func.__qualname__,
                    serialized_callable=serialized_callable,
                    deps=deps,
                    args=copy.deepcopy(args),
                    kwargs=copy.deepcopy(kwargs),
                )
                self._chore_list.append(chore)
                return OutputReference(chore_id, workdir)

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
        inherit_env: bool = True,
    ) -> Chore:
        """
        Create a :obj:`Chore` with a path to an executable rather than a delayed
        python function call.

        Parameters
        ----------
        command : str | list[str]
            The command to be run when the :obj:`Chore` runs
        name : str, optional
            The name  that will be assigned to the chore_id, defaults to the name
            of the function.
        num_tasks : int, optional
            The number of tasks that will be launched with flux, defaults to 1
        cores_per_task : int, optional
            The number of CPU cores that are required to submit the chore, defaults
            to 1
        gpus_per_task : int, optional
            The number of GPUs that are required to submit the chore, defaults to 0
        mpi : bool, optional
            When True, sets Flux shell option ``mpi=pmi2`` on the chorespec (default False).
        env : dict[str, str], optional
            Extra environment variables for the task (default None).
        inherit_env : bool
            If True (default), the Flux jobspec starts from the submitting process
            environment and applies ``env`` overrides.

        Returns
        -------
        Chore
            The executable chore object (already appended to this pipeline).
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
        chore_id = (
            f"chore-{name}-{self._counter:04d}"
            if name
            else f"chore-exec-{self._counter:04d}"
        )
        workdir = self._out_dir / chore_id

        # TODO: make sure that the command paths are absolute paths
        chore = Chore(
            id=chore_id,
            command=command,
            chore_type=ChoreType.EXECUTABLE,
            resources=res,
            workdir=workdir,
        )
        self._chore_list.append(chore)
        return chore

    def _create_graph(self) -> nx.DiGraph:
        """
        Build the graph to of :obj:`Chore`'s with edges representing dependencies

        Return
        ------
        nx.DiGraph
        """

        G = nx.DiGraph()
        known_ids = {chore.id for chore in self._chore_list}

        for chore in self._chore_list:
            G.add_node(chore.id, chore=chore)

        for chore in self._chore_list:
            missing = [dep_id for dep_id in chore.deps if dep_id not in known_ids]
            if missing:
                raise ValueError(
                    f"Chore {chore.id} has unknown dependencies: {missing}"
                )
            for dep_id in chore.deps:
                G.add_edge(dep_id, chore.id)

        return G

    def _sort_graph(self, G: nx.DiGraph) -> list:
        """
        Topologically sorts the graph into a list of :obj:`Chore`'s to have the
        least dependent chores in the front of the list.

        Parameters
        ----------
        G : ns.DiGraph
            A directed graph representing the workflow

        Return
        ------
        list, The sorted graph in topological order

        Raises
        ------

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
        dashboard: bool = False,
    ) -> None:
        """
        Submit the current number of chores. Builds the graph, sorts the graph, and
        creates a :obj:`FluxManager` and runs the workflow with that.

        Parameters
        ----------
        write_restart_freq : int or None
            If an integer *N*, the completion strategy tries to checkpoint after each *N*
            successful chores. **Checkpointing is not implemented yet**; leave ``None`` (or
            pass ``None`` explicitly) for production runs until restart files are supported.
            The default remains ``100`` for historical reasons only.
        buffer_time : float
            The amount of seconds that the :obj:`FluxManager` should wait between
            submission of chores, defaults to 1.0s.
        set_cpu_affinity : bool
            Whether CPU affinity should be set for flux jobspecs, defaults to True.
        set_gpu_affinity : bool
            Whether GPU affinity should be set for flux jobspecs, defaults to False.
        adaptive : bool
            Whether the :obj:`FluxManager` should adaptively submit other chores
            as resources become available, defaults to True.
        dynopro : bool
            Reserved flag forwarded to :meth:`FluxManager.run`. The core manager loop does
            not read it yet; it exists for experiments integrating the in-tree dynopro stack.
        processing_strategy : FutureProcessingStrategy
            The strategy that should be used to process the future objects as :obj:`Chore`'s
            complete.
        dashboard : bool
            Whether or not MatEnsemble will server a GUI Dashboard on port 8000
            as the workflow runs. Defaults to False.

        """

        dag = self._create_graph()
        ordered_ids = self._sort_graph(dag)
        ordered_chores = [dag.nodes[chore_id]["chore"] for chore_id in ordered_ids]

        self._out_dir.mkdir(parents=True, exist_ok=True)

        manager = FluxManager(
            chore_list=ordered_chores,
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
            dashboard=dashboard,
        )

    def graph(self) -> nx.DiGraph:
        return self._create_graph()
