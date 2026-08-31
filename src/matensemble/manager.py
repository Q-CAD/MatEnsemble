import time
import threading

import flux
import flux.job

from pathlib import Path
from collections import deque

from matensemble.logger import _setup_logger, _setup_status_writer
from matensemble.chore import Chore
from matensemble.strategy import (
    AdaptiveStrategy,
    NonAdaptiveStrategy,
    FutureProcessingStrategy,
)
from matensemble.fluxlet import Fluxlet


class FluxManager:
    """
    The :obj:`FluxManager` takes a list of :obj:`Chore`'s and manages their submission
    dependencies and output organization.

    Attributes
    ----------
    _base_dir : Path
        The base directory where the output will be placed.
    _chores_by_id : dict
        A dictionary of chore_id's to :obj:`Chore`
    _dependents : dict
        A dictionary of chore_id's to a list of chore_id's that they depend on
    _remaining_deps : dict
        A dictionary of chore_id's to integers (the number of dependencies remaining)
    _ready : collections.deque
        A double ended queue of :obj:`Chore`'s that are ready for submission
    _blocked : set
        A set of :obj:`Chore`'s that are waiting on their dependencies to resolve
    _running_chores : set
        A set of :obj:`Chore`'s that are currently running
    _completed_chores : list
        A list of :obj:`Chore`'s that have completed successfully
    _failed_chores : list
        A list of :obj:`Chore`'s that failed
    _futures : set
        A set of future objects representing the completion of running chores
    _flux_handle : flux.Flux
        A flux handle
    _fluxlet : matensemble.Fluxlet
        A :obj:`Fluxlet` that is where all the chores are submitted
    _write_restart_freq : int
        The number of chores to be completed before pickling a restart file
    _nnodes_on_allocation : int
        The number of nodes available to chores after applying the broker policy.
    _cores_per_node : int
        The number of cores that are on each node
    _gpus_per_node : int
        The number of gpus that are on each node
    _set_cpu_affinity : bool
        Whether or not CPU affinity will be set
    _set_gpu_affinity : bool
        Whether or not GPU affinity will be set
    _status_writer : StatusWriter
        A :obj:`StatusWriter` for logging the status of the workflow in JSON
    _logger : logging.Logger
        A :obj:`Logger` to log the progress of the workflow
    """

    def __init__(
        self,
        chore_list: list[Chore] | None = None,
        base_dir: Path | None = None,
        write_restart_freq: int | None = None,
        set_cpu_affinity: bool = True,
        set_gpu_affinity: bool = True,
        restart_file: str | None = None,
        reserve_broker_node: bool | None = None,
        controller_cores: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        chore_list : list
            A list of :obj:`Chore`'s that need to be submitted
        base_dir : Path
            The base directory of the workflow
        write_restart_freq : int
            The number of chores to be completed before pickling a restart file
        set_cpu_affinity : bool, optional
            Whther affinity to the CPU should be set, defaults to True
        set_gpu_affinity : bool, optional
            Whether affinity to the GPU should be set, default to True
        restart_file : str
            The path to a restart file which will be loaded and restart the work-
            flow from the save point, default to None.
        reserve_broker_node : bool or None, optional
            ``None`` shares rank 0 in a single-rank instance and reserves it in
            a multi-rank instance. ``True`` always reserves rank 0 and ``False``
            always keeps it available to chores.
        controller_cores : int or None, optional
            Chore capacity reserved for the controller in shared single-rank
            mode. ``None`` defaults to one core in that mode and zero otherwise.

        Return
        ------
        None
        """

        if restart_file:
            self._load_restart(restart_file)
            return None
        if write_restart_freq is not None:
            raise NotImplementedError(
                "MatEnsemble restart/checkpoint files are not supported yet. "
                "Leave write_restart_freq=None."
            )
        if not chore_list:
            raise Exception(
                f"Error: expected chore_list to be a `list[Chore]` instead got {chore_list}"
            )
        if not base_dir:
            raise Exception(
                f"Error: expected base_dir to be a `Path` instead got {base_dir}"
            )
        if reserve_broker_node is not None and not isinstance(
            reserve_broker_node, bool
        ):
            raise TypeError("reserve_broker_node must be a bool or None")
        if controller_cores is not None and (
            isinstance(controller_cores, bool)
            or not isinstance(controller_cores, int)
            or controller_cores < 0
        ):
            raise ValueError("controller_cores must be a non-negative integer or None")

        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

        # dictionary to reference chore objects by their chore-id
        self._chores_by_id = {chore.id: chore for chore in chore_list}
        self._dependents = {chore.id: [] for chore in chore_list}
        self._remaining_deps = {chore.id: len(chore.deps) for chore in chore_list}

        # ensuring that chores have their correct dependencies
        for chore in chore_list:
            for dep in chore.deps:
                self._dependents[dep].append(chore.id)

        self._ready_order = {}
        self._ready_order_counter = 0

        # queue for chores that are ready for submission
        self._ready = deque()
        for chore_id, num_deps in self._remaining_deps.items():
            if num_deps == 0:
                self._mark_ready(chore_id)

        # queue for chores that are waiting on their dependencies to finish
        self._blocked = set(self._chores_by_id.keys()) - set(self._ready)

        # main queues for running chores and completed chores
        self._running_chores = set()
        self._completed_chores = []
        self._failed_chores = []
        self._futures = set()
        self._state_lock = threading.RLock()

        # Apply the resource policy once before initializing Fluxlet.
        self._flux_handle = flux.Flux()
        self._requested_reserve_broker_node = reserve_broker_node
        self._requested_controller_cores = controller_cores
        self._reserve_broker_node = False
        self._controller_cores = 0
        self._drained_broker_node = False

        self._write_restart_freq = write_restart_freq

        try:
            # setup logging and the job launcher from one authoritative resource
            # policy. Fluxlet must not mutate the allocation independently.
            self._nnodes_on_allocation, self._cores_per_node, self._gpus_per_node = (
                self._get_allocation_info()
            )
            self._check_resources()
            self._fluxlet = Fluxlet(
                self._flux_handle,
                self._nnodes_on_allocation,
                self._gpus_per_node,
            )
            self._set_cpu_affinity = set_cpu_affinity
            self._set_gpu_affinity = set_gpu_affinity

            self._status_writer = _setup_status_writer(
                self._base_dir / "status.json",
                nnodes=self._nnodes_on_allocation,
                cores_per_node=self._cores_per_node,
                gpus_per_node=self._gpus_per_node,
            )
            self._logger = _setup_logger(self._base_dir)
        except Exception:
            self._restore_broker_node()
            raise

    # NOTE: The ready ordering is based on the nice score.
    #       The manager will constantly be sorting the ready
    #       list to give the user more control over what gets
    #       scheduled
    def _next_ready_order(self) -> int:
        order = getattr(self, "_ready_order_counter", 0)
        self._ready_order_counter = order + 1
        return order

    def _ready_sort_key(self, chore_id: str) -> tuple[int, int]:
        if not hasattr(self, "_ready_order"):
            self._ready_order = {}
        if chore_id not in self._ready_order:
            self._ready_order[chore_id] = self._next_ready_order()
        chore = self._chores_by_id[chore_id]
        return (getattr(chore, "nice", 0), self._ready_order[chore_id])

    def _sort_ready(self) -> None:
        self._ready = deque(sorted(self._ready, key=self._ready_sort_key))

    def _mark_ready(self, chore_id: str) -> None:
        if chore_id in self._ready:
            self._sort_ready()
            return
        if not hasattr(self, "_ready_order"):
            self._ready_order = {}
        self._ready_order[chore_id] = self._next_ready_order()
        self._ready.append(chore_id)
        self._sort_ready()

    def _make_restart(self) -> None:
        """
        Pickle the current state of the manager and dump it to a file
        """
        raise NotImplementedError(
            "MatEnsemble restart/checkpoint files are not supported yet."
        )

    def _load_restart(self, path: str) -> None:
        """
        Load the pickled restart file and pick up where it left off.

        Parameters
        ----------
        path : str, Path
            The path to the restart file.
        """

        raise NotImplementedError(
            "MatEnsemble restart/checkpoint files are not supported yet."
        )

    def _log_progress(self) -> None:
        """
        Update the status file and append a progress line in the log file
        """
        # The logging thread and submission loop share these fields. Hold one
        # lock through snapshot creation and writing so a record cannot combine
        # the pending count from before a submission with the running/resource
        # counts from after it.
        with self._state_lock:
            pending = len(self._ready) + len(self._blocked)
            ready = len(self._ready)
            blocked = len(self._blocked)
            running = len(self._running_chores)
            completed = len(self._completed_chores)
            failed = len(self._failed_chores)
            free_cores = self._free_cores
            free_gpus = self._free_gpus
            failures = list(self._failed_chores)

            self._status_writer.update(
                pending=pending,
                ready=ready,
                blocked=blocked,
                running=running,
                completed=completed,
                failed=failed,
                free_cores=free_cores,
                free_gpus=free_gpus,
                failures=failures,
            )

            self._logger.info(
                "CHORES: Pending=%d Running=%d Completed=%d Failed=%d | RESOURCES: Free_cores=%d Free_gpus=%d",
                pending,
                running,
                completed,
                failed,
                free_cores,
                free_gpus,
            )

    def _get_allocation_info(self) -> tuple[int, int, int]:
        """
        Get the available nodes, cpus and gpus and calculate the number of
        GPUs per node and number of CPUs per node.
        """

        resources = flux.resource.list.resource_list(self._flux_handle).get()
        if not hasattr(resources, "all"):
            self._initial_resources = resources
            nnodes = len(resources.free.ranks)
            if nnodes == 0:
                return 0, 0, 0
            return (
                nnodes,
                resources.free.ncores // nnodes,
                resources.free.ngpus // nnodes,
            )

        all_ranks = set(resources.all.ranks)
        free_ranks = set(resources.free.ranks)
        rank_count = len(all_ranks)

        if rank_count == 0:
            raise RuntimeError("Flux reported an empty resource inventory")

        self._reserve_broker_node = (
            rank_count > 1
            if self._requested_reserve_broker_node is None
            else self._requested_reserve_broker_node
        )

        if self._reserve_broker_node:
            if rank_count == 1:
                raise ValueError(
                    "reserve_broker_node=True cannot be used with a single-rank "
                    "Flux instance because it would leave no resources for chores; "
                    "use reserve_broker_node=False or the default automatic mode"
                )
            if self._requested_controller_cores not in (None, 0):
                raise ValueError(
                    "controller_cores cannot be reserved when rank 0 is dedicated"
                )
            self._controller_cores = 0
            self._flux_handle.rpc("resource.drain", {"targets": "0"}).get()
            self._drained_broker_node = True
            resources = flux.resource.list.resource_list(self._flux_handle).get()
        else:
            if 0 not in free_ranks:
                raise RuntimeError(
                    "shared broker mode requires rank 0 to be available; start a "
                    "fresh Flux instance or explicitly undrain rank 0"
                )
            if rank_count > 1:
                if self._requested_controller_cores not in (None, 0):
                    raise ValueError(
                        "nonzero controller_cores are only supported in a "
                        "single-rank Flux instance"
                    )
                self._controller_cores = 0
            else:
                self._controller_cores = (
                    1
                    if self._requested_controller_cores is None
                    else self._requested_controller_cores
                )

        nnodes = len(resources.free.ranks)
        physical_cores = resources.free.ncores
        total_gpus = resources.free.ngpus

        if nnodes == 0:
            return 0, 0, 0

        if self._controller_cores >= physical_cores:
            raise ValueError(
                f"controller_cores={self._controller_cores} leaves no cores for "
                f"chores on a {physical_cores}-core Flux allocation"
            )

        usable_cores = physical_cores - self._controller_cores
        self._initial_resources = resources
        cores_per_node = usable_cores // nnodes
        gpus_per_node = total_gpus // nnodes
        return nnodes, cores_per_node, gpus_per_node

    def _restore_broker_node(self) -> None:
        """Undo a rank-0 drain owned by this manager."""

        if not getattr(self, "_drained_broker_node", False):
            return
        self._flux_handle.rpc("resource.undrain", {"targets": "0"}).get()
        self._drained_broker_node = False

    def _chore_resource_footprint(self, chore: Chore) -> tuple[int, int]:
        """
        Return ``(needed_cores, needed_gpus)`` for a chore.

        For whole-node (dynopro) chores — those with ``chore.nnodes`` set — the
        footprint is ``nnodes * cores_per_node`` and ``nnodes * gpus_per_node``
        because ``per_resource`` allocates entire nodes and every core and GPU on
        them becomes unavailable.  For ordinary chores the footprint is the
        familiar ``num_tasks * cores_per_task`` / ``num_tasks * gpus_per_task``.
        """

        if chore.nnodes is not None:
            needed_cores = chore.nnodes * self._cores_per_node
            needed_gpus = chore.nnodes * self._gpus_per_node
        else:
            needed_cores = chore.resources.num_tasks * chore.resources.cores_per_task
            needed_gpus = chore.resources.num_tasks * chore.resources.gpus_per_task

        return needed_cores, needed_gpus

    def _chore_fits_allocation(self, chore: Chore) -> bool:
        """
        Checks whether the given chore is too big to be submitted

        Parameters
        ----------
        chore : Chore
            The :obj:`Chore` to check if it will fit in the allocation
        """

        needed_cores, needed_gpus = self._chore_resource_footprint(chore)

        total_cores = getattr(self, "_total_cores", None)
        if total_cores is None:
            total_cores = self._nnodes_on_allocation * self._cores_per_node
        total_gpus = getattr(self, "_total_gpus", None)
        if total_gpus is None:
            total_gpus = self._nnodes_on_allocation * self._gpus_per_node

        return needed_cores <= total_cores and needed_gpus <= total_gpus

    def _validate_chores(self) -> None:
        """
        Calls :meth:`_chore_fits_allocation()` on each chore given to the manager to make sure
        that they all fit. If a given chore does not fit it will be discarded.
        """

        for chore_id, chore in self._chores_by_id.items():
            if not self._chore_fits_allocation(chore):
                self._record_failure(
                    chore_id,
                    reason="chore_exceeds_allocation",
                )

                try:
                    self._ready.remove(chore_id)
                except ValueError:
                    pass
                self._blocked.discard(chore_id)

                self._logger.error(
                    "CHORE INVALID: chore=%s requires more resources than the allocation can provide",
                    chore_id,
                )
                self._fail_dependents(chore_id)

    def _check_resources(self):
        """
        Initialize resource counters from one Flux resource snapshot.

        Return
        ------
        flux.resource.ResourceList
            The Flux resource snapshot used to initialize the counters.
        """

        if not hasattr(self, "_state_lock"):
            self._state_lock = threading.RLock()

        with self._state_lock:
            if getattr(self, "_running_chores", set()):
                raise RuntimeError(
                    "Flux resources may only be polled before chores are submitted; "
                    "runtime capacity is tracked from submissions and completions"
                )

        resources = getattr(self, "_initial_resources", None)
        if resources is None:
            resources = flux.resource.list.resource_list(self._flux_handle).get()
        else:
            del self._initial_resources
        with self._state_lock:
            self._total_cores = max(0, resources.free.ncores - self._controller_cores)
            self._total_gpus = resources.free.ngpus
            self._free_cores = self._total_cores
            self._free_gpus = self._total_gpus
        return resources

    def _release_resources(self, chore: Chore) -> tuple[int, int]:
        """Return a finished chore's resources to the local available pool."""

        released_cores, released_gpus = self._chore_resource_footprint(chore)
        self._free_cores += released_cores
        self._free_gpus += released_gpus

        if self._free_cores > self._total_cores or self._free_gpus > self._total_gpus:
            raise RuntimeError(
                "resource accounting exceeded the initial Flux allocation after "
                f"finishing {chore.id}: cores={self._free_cores}/{self._total_cores}, "
                f"gpus={self._free_gpus}/{self._total_gpus}"
            )

        return released_cores, released_gpus

    def _finish_chore(
        self,
        chore: Chore,
        *,
        failure_reason: str | None = None,
        exception: str | None = None,
    ) -> tuple[int, int]:
        """Atomically record a terminal chore and release its resources."""

        chore_id = chore.id
        with self._state_lock:
            self._running_chores.remove(chore_id)
            released = self._release_resources(chore)

            if failure_reason is not None:
                self._record_failure(
                    chore_id,
                    reason=failure_reason,
                    exception=exception,
                )
                self._fail_dependents(chore_id)
                return released

            self._completed_chores.append(chore_id)
            for dep_id in self._dependents.get(chore_id, []):
                self._remaining_deps[dep_id] -= 1
                if self._remaining_deps[dep_id] == 0:
                    self._mark_ready(dep_id)
                    self._blocked.discard(dep_id)

            return released

    def _can_submit_now(self, chore: Chore) -> bool:
        """
        Checks to see if there are enough resources to submit the given :obj:`Chore`
        """

        needed_cores, needed_gpus = self._chore_resource_footprint(chore)
        return self._free_cores >= needed_cores and self._free_gpus >= needed_gpus

    def _has_failed(self, chore_id: str) -> bool:
        """
        Checks if a given chore_id has failed
        """

        return any(item["chore_id"] == chore_id for item in self._failed_chores)

    def _record_failure(
        self,
        chore_id: str,
        reason: str,
        *,
        upstream: str | None = None,
        exception: str | None = None,
    ) -> None:
        """
        Logs the failure of a chore with its reason
        """

        if self._has_failed(chore_id):
            return

        self._failed_chores.append(
            {
                "chore_id": chore_id,
                "reason": reason,
                "upstream": upstream,
                "exception": exception,
            }
        )

    def _fail_dependents(self, failed_chore_id: str) -> None:
        """
        Cascades the failure of one chore to all of it dependents to avoid
        deadlocks.
        """

        for dep_id in self._dependents.get(failed_chore_id, []):
            if dep_id in self._completed_chores or dep_id in self._running_chores:
                continue

            try:
                self._ready.remove(dep_id)
            except ValueError:
                pass
            self._blocked.discard(dep_id)

            if not self._has_failed(dep_id):
                self._record_failure(
                    dep_id,
                    reason="dependency_failed",
                    upstream=failed_chore_id,
                )
                self._logger.error(
                    "CHORE SKIPPED: chore=%s because dependency %s failed",
                    dep_id,
                    failed_chore_id,
                )

            self._fail_dependents(dep_id)

    def _submit_one(
        self, chore_id: str, buffer_time: float, dynopro: bool = False
    ) -> bool:
        """
        Submits a :obj:`Chore` and does book-keeping all the queues and resources
        count

        Parameters
        ----------
        buffer_time : float
            Retained for compatibility with custom strategies. Submission is
            intentionally not delayed; the value controls future waiting.
        """

        chore = self._chores_by_id[chore_id]

        try:
            fut = self._fluxlet.submit(
                self._executor,
                chore,
                set_cpu_affinity=self._set_cpu_affinity,
                set_gpu_affinity=self._set_gpu_affinity,
                dynopro=dynopro,
            )
        except Exception as e:
            self._logger.exception("CHORE SUBMIT FAILED: chore=%s", chore_id)
            self._record_failure(
                chore_id,
                reason="submit_exception",
                exception=repr(e),
            )
            self._fail_dependents(chore_id)
            self._blocked.discard(chore_id)
            return False

        self._blocked.discard(chore_id)
        fut.chore_id = chore_id
        fut.chore_obj = chore
        self._running_chores.add(chore_id)
        self._futures.add(fut)

        needed_cores, needed_gpus = self._chore_resource_footprint(chore)
        self._free_cores -= needed_cores
        self._free_gpus -= needed_gpus
        return True

    def _submit_until_ooresources(
        self, buffer_time: float, dynopro: bool = False
    ) -> bool:
        """
        Submit as many chores as possible until out-of-resources

        Parameters
        ----------
        buffer_time : float
            Retained for compatibility with custom strategies. Submission is
            immediate; the value controls future waiting.
        """

        submitted_any = False

        # Examine every chore that was ready when this fill phase began. Jobs
        # that do not fit are rotated to the back without ever disappearing
        # from status snapshots. Holding the state lock across the quick Flux
        # submission keeps queue, running, future, and resource transitions
        # atomic with respect to the logging thread.
        for _ in range(len(self._ready)):
            with self._state_lock:
                chore_id = self._ready[0]
                chore = self._chores_by_id[chore_id]

                if not self._can_submit_now(chore):
                    self._ready.rotate(-1)
                    continue

                self._ready.popleft()
                submitted_any = (
                    self._submit_one(chore_id, buffer_time, dynopro=dynopro)
                    or submitted_any
                )

        return submitted_any

    def _log_worker(self, delay: float, stop_event: threading.Event) -> None:
        """
        Function that updates the logs every so often
        """
        while not stop_event.wait(delay):
            self._log_progress()

    def _add_chore(self, chore: Chore) -> bool:
        """
        Add a UserStrategy spawned chore to the queue.

        Returns
        -------
        bool
            True if *chore* was admitted to the manager, False if it was rejected.
        """

        # Keep direct ``__new__`` construction used by lightweight custom
        # strategy tests compatible with the normal initialized manager.
        if not hasattr(self, "_state_lock"):
            self._state_lock = threading.RLock()
        with self._state_lock:
            return self._add_chore_locked(chore)

    def _add_chore_locked(self, chore: Chore) -> bool:
        """Implement :meth:`_add_chore` while the state lock is held."""

        if not self._chore_fits_allocation(chore):
            self._record_failure(chore.id, reason="chore_exceeds_allocation")
            self._logger.error(
                "CHORE INVALID: chore=%s requires more resources than the allocation can provide",
                chore.id,
            )
            self._fail_dependents(chore.id)
            return False

        if chore.id in self._chores_by_id:
            self._logger.error(
                "CHORE DUPLICATE: chore=%s already exists, rejecting spawn",
                chore.id,
            )
            return False

        for dep in chore.deps:
            if dep not in self._chores_by_id:
                self._record_failure(
                    chore.id, reason="unknown_dependency", upstream=dep
                )
                self._logger.error(
                    "CHORE INVALID: chore=%s has unknown dependency %s",
                    chore.id,
                    dep,
                )
                return False
            if self._has_failed(dep):
                self._record_failure(chore.id, reason="dependency_failed", upstream=dep)
                self._logger.error(
                    "CHORE SKIPPED: chore=%s because dependency %s already failed",
                    chore.id,
                    dep,
                )
                return False

        self._chores_by_id[chore.id] = chore
        self._dependents.setdefault(chore.id, [])

        remaining = sum(1 for dep in chore.deps if dep not in self._completed_chores)
        self._remaining_deps[chore.id] = remaining

        for dep in chore.deps:
            self._dependents.setdefault(dep, []).append(chore.id)

        if remaining == 0:
            self._mark_ready(chore.id)
            self._blocked.discard(chore.id)
        else:
            self._blocked.add(chore.id)

        return True

    def run(
        self,
        buffer_time: float = 1.0,
        log_delay: float = 5.0,
        adaptive: bool = True,
        dynopro: bool = False,
        processing_strategy: FutureProcessingStrategy | None = None,
        restarting: bool = False,
    ) -> None:
        """Run the workflow and restore any broker-node drain owned by it."""

        try:
            self._run_workflow(
                buffer_time=buffer_time,
                log_delay=log_delay,
                adaptive=adaptive,
                dynopro=dynopro,
                processing_strategy=processing_strategy,
                restarting=restarting,
            )
        finally:
            self._restore_broker_node()

    def _run_workflow(
        self,
        buffer_time: float = 1.0,
        log_delay: float = 5.0,
        adaptive: bool = True,
        dynopro: bool = False,
        processing_strategy: FutureProcessingStrategy | None = None,
        restarting: bool = False,
    ) -> None:
        """
        Runs the 'Super Loop' until there are no more ready, running or blocked
        :obj:`Chore`'s

        Parameters
        ----------
        buffer_time : float
            Maximum number of seconds adaptive strategies wait for a future
            completion before checking the workflow state again.
        log_delay : float
            The amount of time in seconds that the log files will be written to
        adaptive : bool
            Whether or not :obj:`Chore`'s should be submitted adaptively, defaults
            to True
        dynopro : bool
            Currently does nothing because I couldn't figure out what it did
            to begin with.
        restarting : bool
            Whether :meth:`run` is being invoked for the first time or after a
            restart file has been loaded


        Notes
        -----
        In adaptive mode, each completed future releases its locally tracked
        resources and can cause immediate backfilling before the next outer loop
        iteration.
        In non-adaptive mode, completion processing waits for the entire currently
        running wave before the outer loop submits another.

        Each loop iteration:

        #. Submits new chores until locally tracked resources are exhausted
        #. processes completed chores using a FutureProcessingStrategy:
            * User implementation if a processing_strategy is used
            * AdaptiveStrategy if adaptive=True
            * NonAdaptiveStrategy otherwise
        """

        if processing_strategy:
            proc_strat = processing_strategy
        elif adaptive:
            proc_strat = AdaptiveStrategy(self)
        else:
            proc_strat = NonAdaptiveStrategy(self)

        buffer_time = 0.0 if buffer_time is None else float(buffer_time)
        if buffer_time < 0:
            raise ValueError("buffer_time must be non-negative")
        if log_delay <= 0:
            raise ValueError("log_delay must be greater than zero")
        self._dynopro = dynopro

        with flux.job.FluxExecutor() as executor:
            self._executor = executor

            if restarting:
                self._logger.info("=== RESTARTING WORKFLOW ENVIRONMENT ===")
            else:
                self._logger.info("=== ENTERING WORKFLOW ENVIRONMENT ===")
                self._start_time = time.perf_counter()

            self._validate_chores()

            # Resource counters were initialized from Flux once during manager
            # construction. From this point on, submissions and completions are
            # the source of truth.
            logging_stop = threading.Event()
            logging_thread = threading.Thread(
                target=self._log_worker,
                args=(log_delay, logging_stop),
                daemon=True,
            )
            logging_thread.start()
            try:
                self._log_progress()

                ### Super Loop ###
                done = (
                    len(self._ready) == 0
                    and len(self._running_chores) == 0
                    and len(self._blocked) == 0
                )
                while not done:
                    self._submit_until_ooresources(
                        buffer_time=buffer_time, dynopro=dynopro
                    )
                    proc_strat.process_futures(buffer_time=buffer_time)

                    done = (
                        len(self._ready) == 0
                        and len(self._running_chores) == 0
                        and len(self._blocked) == 0
                    )
                ### Super Loop ###
            finally:
                logging_stop.set()
                logging_thread.join()

            end = time.perf_counter()
            self._log_progress()
            self._logger.info("=== EXITING WORKFLOW ENVIRONMENT ===")
            self._logger.info(
                "Workflow took %.4f seconds to run.", end - self._start_time
            )
