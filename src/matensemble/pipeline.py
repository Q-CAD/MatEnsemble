from __future__ import annotations

import atexit
import threading
import time
import copy
import functools
import datetime
import sys

import cloudpickle
import networkx as nx

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Any
from pathlib import Path

from matensemble.manager import FluxManager
from matensemble.strategy import FutureProcessingStrategy, UserStrategy
from matensemble.chore import Chore, ChoreSpec
from matensemble.model import OutputReference, Resources, ChoreType
from matensemble.utils import _collect_dep_ids


def _registry_entry_filename(key: str) -> str:
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


class Pipeline:
    """
    Object to build and submit a MatEnsemble workflow as a directed acyclic graph (DAG)
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
    specification from disk, deserializes the function from MatEnsemble's internal
    function registry, replaces dependency references with their
    concrete upstream results, and calls the function.
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
        self._registry: dict[str, Callable] = {}
        self._output_reference_list: list[OutputReference] = []

        root = Path.cwd() if basedir is None else Path(basedir)
        self._base_dir = (
            root / f"matensemble_workflow-{datetime.datetime.now():%Y%m%d_%H%M%S}"
        )
        self._out_dir = self._base_dir / "out"

        self._strategy_spec = None
        self._finished = False
        self._submission_exception: Exception | None = None
        self._submission_state_lock = threading.Lock()
        self._submission_executor: ThreadPoolExecutor | None = None
        self._submission_future: Future | None = None

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
        Wrap a function to produce a :obj:`Chore` and returns an
        :class:`~matensemble.model.OutputReference`.
        which other chore definitions can use to define dependencies.

        :obj:`Chore` objects are delayed function calls that are put into the
        :obj:`Pipeline` and are added into its graph when you run it.
        A PYTHON :obj:`Chore` contains metadata that is needed to reproduce a
        function. The :obj:`FluxManager` will create a :obj:`Fluxlet` and
        submit the chore to flux which calls the module :py:mod: `matensemble.runtime_worker`.
        :py:mod: `matensemble.runtime_worker` takes in two command line
        arguments which are the :param: `chore_id` and :param: `spec_file` which
        the module will use to load the serialized function from MatEnsemble's
        internal function registry, call the function with the given arguements
        and key-word arguments and store the results in the chores respective directory.

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
            registry_key = name or str(func.__qualname__)
            self._registry[registry_key] = func

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

                chore_id = (
                    f"chore-{name}-{self._counter:04d}"
                    if name
                    else f"chore-{func.__name__}-{self._counter:04d}"
                )
                workdir = self._out_dir / chore_id
                cmd = [
                    sys.executable,
                    "-m",
                    "matensemble.runtime_worker",
                    "--chore-id",
                    chore_id,
                    "--spec-file",
                    str(workdir / "chore.pickle"),
                ]
                res = Resources(
                    num_tasks=num_tasks,
                    cores_per_task=cores_per_task,
                    gpus_per_task=gpus_per_task,
                    mpi=mpi,
                    env=merged_env,
                    inherit_env=inherit_env,
                )
                chore_qualname = registry_key
                deps = _collect_dep_ids(args, kwargs)

                chore = Chore(
                    id=chore_id,
                    workdir=workdir,
                    command=cmd,
                    chore_type=ChoreType.PYTHON,
                    resources=res,
                    chore_qualname=chore_qualname,
                    deps=deps,
                    args=copy.deepcopy(args),
                    kwargs=copy.deepcopy(kwargs),
                )
                out_ref = OutputReference(chore_id, workdir)

                self._chore_list.append(chore)
                self._output_reference_list.append(out_ref)

                return out_ref

            return wrapper

        return decorator

    def strategy(
        self,
        bolo_list: list[str],
        name: str | None = None,
        num_tasks: int = 1,
        cores_per_task: int = 1,
        gpus_per_task: int = 0,
        mpi: bool = False,
        env: dict[str, str] | None = None,
        inherit_env: bool = True,
    ):
        """
        Creates a strategy, which is essentially a callback function to another chore.
        The callback function itself is a chore. This function is expected to
        return an :obj:`ChoreSpec` which will then dynamically spawn a new chore
        into the queue based on the specification that is returned.

        Parameters
        ----------
        bolo_list : list[str]
            The names of the chores that you want this to callback on
        name : str, optional
            The name  that will be assigned to the chore_id, defaults to the
            name of the function.
        num_tasks : int, optional
            The number of tasks that will be launched with flux, defaults to 1
        cores_per_task : int, optional
            The number of CPU cores that are required to submit the chore,
            defaults to 1
        gpus_per_task : int, optional
            The number of GPUs that are required to submit the chore, defaults
            to 0
        mpi : bool, optional
            When True, sets Flux shell option ``mpi=pmi2`` on the chorespec
            (default False).
        env : dict[str, str], optional
            Extra environment variables for the task. For Python chores,
            ``PYTHONPATH`` is merged to include the workflow parent directory.
        inherit_env : bool
            If True (default), the Flux jobspec starts from the submitting
            process environment and applies ``env`` overrides.

        Returns
        -------
        Callable
            A dummy function that just prints a warning to the stdout. The
            actual function is stored in the registry
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            for chore in bolo_list:
                if chore not in self._registry:
                    raise Exception(
                        f"Error: The chore '{chore}' was not found in the registry"
                    )

            registry_key = name or str(func.__qualname__)
            self._registry[registry_key] = func

            self._strategy_spec = {
                "name": registry_key,
                "resources": Resources(
                    num_tasks=num_tasks,
                    cores_per_task=cores_per_task,
                    gpus_per_task=gpus_per_task,
                    mpi=mpi,
                    env=env,
                    inherit_env=inherit_env,
                ),
                "bolo_list": bolo_list,
            }

            def disabled_wrapper(*args: Any, **kwargs: Any) -> None:
                raise RuntimeError(
                    f"Do not call '{registry_key}' directly. "
                    "This strategy is managed internally by the workflow engine."
                )

            return disabled_wrapper

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
        Create a :obj:`Chore` with an argv style command rather than a delayed
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
            workdir=workdir,
            command=command,
            chore_type=ChoreType.EXECUTABLE,
            resources=res,
        )
        self._chore_list.append(chore)
        return chore

    # TODO: So this needs to call the driver, and the driver will split it into two groups
    #       one group on the gpus and one on the cpus, but I need to read the code to figure
    #       out where it is actually splitting to, so like does each GPU get a cpu? or how
    #       does that work, and we need to make a way to then add thi sto the piplien as
    #       its own chore, which should be simple. but we also need a way to standardize
    #       the functions so that they have a communicator and I need to figure out how
    #       important it is that we give them that.
    def dynopro(
        self,
        gpu_subprocess: str,
        cpu_subprocess: str,
        nnodes: int,
        gpus_per_node: int,
        cores_per_node: int,
        name: str | None = None,
        num_tasks: int | None = None,
        cores_per_task: int = 1,
        gpus_per_task: int = 0,
        env: dict[str, str] | None = None,
        inherit_env: bool = True,
        subprocess_args: tuple[Any, ...] = (),
        subprocess_kwargs: dict[str, Any] | None = None,
        gpu_args: tuple[Any, ...] = (),
        gpu_kwargs: dict[str, Any] | None = None,
        cpu_args: tuple[Any, ...] = (),
        cpu_kwargs: dict[str, Any] | None = None,
    ) -> Chore:
        """
        Register a dynopro wrapper chore that runs GPU and CPU subprocess chores.

        Prefer ``gpu_args`` / ``gpu_kwargs`` and ``cpu_args`` / ``cpu_kwargs`` to
        pass arguments to each registered subprocess independently. The legacy
        ``subprocess_args`` / ``subprocess_kwargs`` parameters still apply the
        same payload to both subprocesses, but cannot be mixed with per-chore
        payloads.
        """

        for subprocess_name in (gpu_subprocess, cpu_subprocess):
            if subprocess_name not in self._registry:
                raise ValueError(
                    f"dynopro subprocess {subprocess_name!r} is not registered. "
                    "Register it first with Pipeline.chore(...)."
                )

        if not isinstance(nnodes, int) or nnodes < 1:
            raise ValueError("nnodes must be an integer >= 1")
        if not isinstance(gpus_per_node, int) or gpus_per_node < 1:
            raise ValueError("gpus_per_node must be an integer >= 1")
        if not isinstance(cores_per_node, int) or cores_per_node < 1:
            raise ValueError("cores_per_node must be an integer >= 1")
        if gpus_per_node > cores_per_node:
            raise ValueError("gpus_per_node cannot exceed cores_per_node")

        res = Resources(
            num_tasks=num_tasks if num_tasks is not None else nnodes * cores_per_node,
            cores_per_task=cores_per_task,
            gpus_per_task=gpus_per_task,
            mpi=True,
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

        subprocess_kwargs = {} if subprocess_kwargs is None else subprocess_kwargs
        shared_payload_provided = bool(subprocess_args) or bool(subprocess_kwargs)
        per_chore_payload_provided = (
            bool(gpu_args)
            or bool(cpu_args)
            or bool(gpu_kwargs)
            or bool(cpu_kwargs)
        )
        if shared_payload_provided and per_chore_payload_provided:
            raise ValueError(
                "dynopro shared subprocess_args/subprocess_kwargs cannot be mixed "
                "with gpu_args/gpu_kwargs or cpu_args/cpu_kwargs"
            )
        if gpu_subprocess == cpu_subprocess and per_chore_payload_provided:
            raise ValueError(
                "dynopro per-chore arguments require distinct gpu_subprocess "
                "and cpu_subprocess names"
            )

        gpu_kwargs = {} if gpu_kwargs is None else gpu_kwargs
        cpu_kwargs = {} if cpu_kwargs is None else cpu_kwargs

        if shared_payload_provided:
            dynopro_args = {
                gpu_subprocess: subprocess_args,
                cpu_subprocess: subprocess_args,
            }
            dynopro_kwargs = {
                gpu_subprocess: subprocess_kwargs,
                cpu_subprocess: subprocess_kwargs,
            }
        else:
            dynopro_args = {
                gpu_subprocess: gpu_args,
                cpu_subprocess: cpu_args,
            }
            dynopro_kwargs = {
                gpu_subprocess: gpu_kwargs,
                cpu_subprocess: cpu_kwargs,
            }

        deps = _collect_dep_ids(dynopro_args, dynopro_kwargs)

        chore = Chore(
            id=chore_id,
            workdir=workdir,
            command=[
                sys.executable,
                "-m",
                "matensemble.dynopro.driver",
                f"--gpus-per-node={gpus_per_node}",
                f"--cores-per-node={cores_per_node}",
                f"--gpu-subprocess={gpu_subprocess}",
                f"--cpu-subprocess={cpu_subprocess}",
                f"--chore-dir={workdir}",
            ],
            chore_type=ChoreType.EXECUTABLE,
            resources=res,
            deps=deps,
            args=copy.deepcopy(subprocess_args),
            kwargs=copy.deepcopy(subprocess_kwargs),
            dynopro_args=copy.deepcopy(dynopro_args),
            dynopro_kwargs=copy.deepcopy(dynopro_kwargs),
            nnodes=nnodes,
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
            raise Exception("Error: MatEnsemble workflow graph cannot contain cycles, \
                        topological sort not possible")
        else:
            try:
                return list(nx.topological_sort(G))
            except nx.NetworkXUnfeasible:
                raise Exception(
                    "Error: MatEnsemble workflow graph cannot contain cycles, \
                            cycle found by topological sort"
                )

    def _spawn_chore_from_name(
        self,
        chore_name: str,
        resources: Resources | None = None,
        dependent: OutputReference | None = None,
    ) -> tuple[Chore, OutputReference]:
        self._counter += 1

        chore_id = f"chore-{chore_name}-{self._counter:04d}"
        workdir = self._out_dir / chore_id
        cmd = [
            sys.executable,
            "-m",
            "matensemble.runtime_worker",
            "--chore-id",
            chore_id,
            "--spec-file",
            str(workdir / "chore.pickle"),
        ]

        args = (dependent,) if dependent is not None else ()
        deps = _collect_dep_ids(args, {})

        chore = Chore(
            id=chore_id,
            workdir=workdir,
            command=cmd,
            chore_type=ChoreType.PYTHON,
            resources=resources if resources else Resources(),
            chore_qualname=chore_name,
            deps=deps,
            args=args,
        )
        out_ref = OutputReference(chore_id, workdir)

        return chore, out_ref

    def _spawn_chore_from_spec(self, spec: ChoreSpec) -> tuple[Chore, OutputReference]:
        self._counter += 1

        chore_id = f"chore-{spec.qualname}-{self._counter:04d}"
        workdir = self._out_dir / chore_id
        cmd = [
            sys.executable,
            "-m",
            "matensemble.runtime_worker",
            "--chore-id",
            chore_id,
            "--spec-file",
            str(workdir / "chore.pickle"),
        ]

        args = copy.deepcopy(spec.args)
        kwargs = copy.deepcopy(spec.kwargs)
        deps = _collect_dep_ids(args, kwargs)

        chore = Chore(
            id=chore_id,
            workdir=workdir,
            command=cmd,
            chore_type=ChoreType.PYTHON,
            resources=spec.resources,
            chore_qualname=spec.qualname,
            deps=deps,
            args=args,
            kwargs=kwargs,
        )
        out_ref = OutputReference(chore_id, workdir)

        return chore, out_ref

    def _admit_spawned_chore(
        self, chore: Chore, out_ref: OutputReference, manager: FluxManager
    ) -> None:
        """
        Validate registry membership, admit *chore* to *manager*, then record it
        on this pipeline. Raises if the spawn cannot be admitted.
        """
        if chore.chore_qualname not in self._registry:
            raise ValueError(
                f"spawned chore qualname {chore.chore_qualname!r} is not in the pipeline registry"
            )
        if not manager._add_chore(chore):
            raise RuntimeError(
                f"FluxManager rejected spawned chore {chore.id!r} (duplicate id, "
                "dependency error, or allocation)"
            )
        self._chore_list.append(chore)
        self._output_reference_list.append(out_ref)

    def close(self) -> None:
        if self._submission_executor is not None:
            self._submission_executor.shutdown(wait=True)
            self._submission_executor = None

    def __enter__(self) -> Pipeline:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _submit(
        self,
        write_restart_freq: int | None = None,
        buffer_time: float = 1.0,
        log_delay: float = 5.0,
        set_cpu_affinity: bool = True,
        set_gpu_affinity: bool = False,
        adaptive: bool = True,
        dynopro: bool = False,
        processing_strategy: FutureProcessingStrategy | None = None,
    ):
        """
        The actual submit under the API hood

        Returns
        -------
        dict
            The results of the workflow with each key being the chores ID and
            the value being the results of the chore.
        """
        if write_restart_freq is not None:
            raise NotImplementedError(
                "MatEnsemble restart/checkpoint files are not supported yet. "
                "Leave write_restart_freq=None."
            )

        with self._submission_state_lock:
            self._finished = False
            self._submission_exception = None
        try:
            dag = self._create_graph()
            ordered_ids = self._sort_graph(dag)
            ordered_chores = [dag.nodes[chore_id]["chore"] for chore_id in ordered_ids]

            self._out_dir.mkdir(parents=True, exist_ok=True)
            registry_dir = self._out_dir / "registry"
            registry_dir.mkdir(parents=True, exist_ok=True)
            for key in self._registry:
                safe_name = _registry_entry_filename(key)
                with open(registry_dir / safe_name, "wb") as file:
                    cloudpickle.dump(self._registry[key], file)

            manager = FluxManager(
                chore_list=ordered_chores,
                base_dir=self._base_dir,
                write_restart_freq=write_restart_freq,
                set_cpu_affinity=set_cpu_affinity,
                set_gpu_affinity=set_gpu_affinity,
            )

            if self._strategy_spec:
                strat = UserStrategy(
                    manager=manager,
                    pipeline=self,
                    processing_chore=self._strategy_spec["name"],
                    processing_chore_resources=self._strategy_spec["resources"],
                    bolo_list=self._strategy_spec["bolo_list"],
                )
            else:
                strat = processing_strategy

            manager.run(
                buffer_time=buffer_time,
                log_delay=log_delay,
                adaptive=adaptive,
                dynopro=dynopro,
                processing_strategy=strat,
            )
        except Exception as exc:
            with self._submission_state_lock:
                self._submission_exception = exc
                self._finished = True
            raise
        else:
            with self._submission_state_lock:
                self._finished = True
            return self._collect_results()

    def graph(self) -> nx.DiGraph:
        return self._create_graph()

    def _collect_results(self) -> dict[str, Any]:
        results = {}
        for out_ref in self._output_reference_list:
            try:
                results[f"{out_ref.chore_id}"] = out_ref.result()
            except Exception as e:
                results[f"{out_ref.chore_id}"] = (
                    f"Error: Could not access the results of {out_ref.chore_id} due to exception -> {e}"
                )
        return results

    def submit(
        self,
        write_restart_freq: int | None = None,
        buffer_time: float = 1.0,
        log_delay: float = 5.0,
        set_cpu_affinity: bool = True,
        set_gpu_affinity: bool = False,
        adaptive: bool = True,
        dynopro: bool = False,
        processing_strategy: FutureProcessingStrategy | None = None,
    ) -> Future:
        """
        Submit the current number of chores. Builds the graph, sorts the graph, and
        creates a :obj:`FluxManager` and runs the workflow with that.

        Parameters
        ----------
        write_restart_freq : int or None
            Restart/checkpoint files are not supported yet. Leave this as
            ``None``. Passing an integer raises :exc:`NotImplementedError`.
        buffer_time : float
            The amount of seconds that the :obj:`FluxManager` should wait between
            submission of chores, defaults to 1.0s.
        log_delay : float
            The amount delay in seconds between the writing of logs
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

        Returns
        -------
        Future
            A Future object which represents the run of the workflow. The results will be a
            dictionary with the keys being the ID's of the chores and the values being the
            results of each respective chore.
        """

        with self._submission_state_lock:
            if (
                self._submission_future is not None
                and not self._submission_future.done()
            ):
                raise RuntimeError(
                    "submit() called while a workflow is already running"
                )
            self._submission_exception = None
            self._finished = False

        if self._submission_executor is None:
            self._submission_executor = ThreadPoolExecutor(max_workers=1)
            atexit.register(self.close)

        # Returns a concurrent.futures.Future object
        fut = self._submission_executor.submit(
            self._submit,
            write_restart_freq=write_restart_freq,
            buffer_time=buffer_time,
            log_delay=log_delay,
            set_cpu_affinity=set_cpu_affinity,
            set_gpu_affinity=set_gpu_affinity,
            adaptive=adaptive,
            dynopro=dynopro,
            processing_strategy=processing_strategy,
        )
        with self._submission_state_lock:
            self._submission_future = fut
        return fut

    def results(self, timeout=100):
        """
        Returns a dictionary of each chore to its results
        """
        deadline = time.monotonic() + timeout

        # if the workflow is not finished spin wait until it is or you reach a timeout
        while True:
            with self._submission_state_lock:
                finished = self._finished
            if finished or time.monotonic() >= deadline:
                break
            time.sleep(0.1)

        with self._submission_state_lock:
            exc = self._submission_exception
            done = self._finished
        if exc is not None:
            raise exc
        if done:
            return self._collect_results()
        return "Error: The results are not ready (timeout reached)."
