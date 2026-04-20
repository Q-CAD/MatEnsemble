import os
import copy
import time
import pickle
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
from matensemble.utils import setup_dashboard


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
        A list of :obj:`Chore`'s that have completed succesfully
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
        The number of nodes available on the allocaiton minus one for flux broker
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
        write_restart_freq: int | None = 100,
        set_cpu_affinity: bool = True,
        set_gpu_affinity: bool = True,
        restart_file: str | None = None,
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
            Whether affinity to the GPU should be set, defulat to True
        restart_file : str
            The path to a restart file which will be loaded and restart the work-
            flow from the save point, default to None.

        Return
        ------
        None
        """

        if restart_file:
            self._load_restart(restart_file)
            return None
        if not chore_list:
            raise Exception(
                f"Error: expected chore_list to be a `list[Chore]` instead got {chore_list}"
            )
        if not base_dir:
            raise Exception(
                f"Error: expected base_dir to be a `Path` instead got {base_dir}"
            )

        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

        # dictionary to referenece chore objects by their chore-id
        self._chores_by_id = {chore.id: chore for chore in chore_list}
        self._dependents = {chore.id: [] for chore in chore_list}
        self._remaining_deps = {chore.id: len(chore.deps) for chore in chore_list}

        # ensuring that chores have their correct dependencies
        for chore in chore_list:
            for dep in chore.deps:
                self._dependents[dep].append(chore.id)

        # queue for chores that are ready for submission
        self._ready = deque(
            [
                chore_id
                for chore_id, num_deps in self._remaining_deps.items()
                if num_deps == 0
            ]
        )

        # queue for chores that are waiting on their dependencies to finish
        self._blocked = set(self._chores_by_id.keys()) - set(self._ready)

        # main queues for running chores and completed chores
        self._running_chores = set()
        self._completed_chores = []
        self._failed_chores = []
        self._futures = set()

        # aquiring a flux handle
        self._flux_handle = flux.Flux()
        self._fluxlet = Fluxlet(self._flux_handle)

        self._write_restart_freq = write_restart_freq

        # setup logging to be able to communicate with dashboard
        self._nnodes_on_allocation, self._cores_per_node, self._gpus_per_node = (
            self._get_allocation_info()
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

    def _make_restart(self) -> None:
        """
        Pickle the current state of the manager and dump it to a file
        """

        fm = copy.deepcopy(self)
        fm._ready.extendleft(fm._running_chores)
        fm._running_chores = set()
        fm._futures = set()

        pickle.dump(fm, open(f"restart_{len(self._completed_chores)}.dat", "wb"))
        self._logger.info("=== CREATING RESTART FILE ===")

    def _load_restart(self, path: str) -> None:
        """
        Load the pickled restart file and pick up where it left off.

        Parameters
        ----------
        path : str, Path
            The path to the restart file.
        """

        if path and os.path.exists(path):
            try:
                fm = pickle.load(open(path, "rb"))
                self._base_dir = fm._base_dir
                self._chores_by_id = fm._chores_by_id

                self._dependents = fm._dependents
                self._remaining_deps = fm._remaining_deps

                self._ready = fm._ready
                self._blocked = fm._blocked
                self._running_chores = fm._running_chores
                self._completed_chores = fm._completed_chores
                self._failed_chores = fm._failed_chores
                self._futures = fm._futures

                self._flux_handle = flux.Flux()
                self._fluxlet = Fluxlet(self._flux_handle)

                self._write_restart_freq = fm._write_restart_freq
                self._nnodes_on_allocation = fm._nnodes_on_allocation
                self._cores_per_node = fm.cores_per_node
                self._gpus_per_node = fm.gpus_per_node
                self._set_cpu_affinity = fm.set_cpu_affinity
                self._set_gpu_affinity = fm.set_gpu_affinity

                self._logger = fm._logger
                self._status_writer = fm._status_writer

                self._start_time = fm._start_time
            except Exception as e:
                self._logger.error(e)
                raise e

        # TODO: Create a way to call the run method with all the previous
        #       arguments or expose a function that users can call that
        #       can restart a workflow with all of the args that they want
        # self.run(buffer_time)

    def _log_progress(self) -> None:
        """
        Update the status file and append a progress line in the log file
        """
        pending = len(self._ready) + len(self._blocked)
        self._status_writer.update(
            pending=pending,
            running=len(self._running_chores),
            completed=len(self._completed_chores),
            failed=len(self._failed_chores),
            free_cores=self._free_cores,
            free_gpus=self._free_gpus,
        )

        self._logger.info(
            "CHORES: Pending=%d Running=%d Completed=%d Failed=%d | RESOURCES: Free_cores=%d Free_gpus=%d",
            pending,
            len(self._running_chores),
            len(self._completed_chores),
            len(self._failed_chores),
            self._free_cores,
            self._free_gpus,
        )

    def _get_allocation_info(self) -> tuple[int, int, int]:
        """
        Get the available nodes, cpus and gpus and calculate the number of
        GPUs per node and number of CPUs per node.
        """

        # drain broker rank first, then measure what is actually usable
        self._flux_handle.rpc("resource.drain", {"targets": "0"}).get()

        resources = flux.resource.list.resource_list(self._flux_handle).get()
        nnodes = len(resources.free.ranks)
        total_cores = resources.free.ncores
        total_gpus = resources.free.ngpus

        if nnodes == 0:
            return 0, 0, 0

        cores_per_node = total_cores // nnodes
        gpus_per_node = total_gpus // nnodes
        return nnodes, cores_per_node, gpus_per_node

    def _chore_fits_allocation(self, chore: Chore) -> bool:
        """
        Checks whether the given chore is too big to be submitted

        Parameters
        ----------
        chore : Chore
            The :obj:`Chore` to check if it will fit in the allocation
        """

        needed_cores = chore.resources.num_tasks * chore.resources.cores_per_task
        needed_gpus = chore.resources.num_tasks * chore.resources.gpus_per_task

        total_cores = self._nnodes_on_allocation * self._cores_per_node
        total_gpus = self._nnodes_on_allocation * self._gpus_per_node

        return needed_cores <= total_cores and needed_gpus <= total_gpus

    def _validate_chores(self) -> None:
        """
        Calls :method:`_chore_fits_allocation()` on each chore given to the manager to make sure
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

    def _check_resources(self) -> None:
        """
        Gets the available resources from Flux's Remote Procedure Call (RPC)
        and sets them as private fields

        Return
        ------
        None
        """

        resources = flux.resource.list.resource_list(self._flux_handle).get()
        self._free_cores = resources.free.ncores
        self._free_gpus = resources.free.ngpus

    def _can_submit_now(self, chore: Chore) -> bool:
        """
        Checks to see if there are enough resources to submit the given :obj:`Chore`
        """

        needed_cores = chore.resources.num_tasks * chore.resources.cores_per_task
        needed_gpus = chore.resources.num_tasks * chore.resources.gpus_per_task
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

    def _submit_one(self, chore_id: str, buffer_time: float) -> None:
        """
        Submits a :obj:`Chore` and does book-keeping all the queues and resources
        count

        Parameters
        ----------
        buffer_time : float
            The amount of time in seconds buffer the submission of :obj:`Chore`'s
        """

        chore = self._chores_by_id[chore_id]

        try:
            fut = self._fluxlet.submit(
                self._executor,
                chore,
                set_cpu_affinity=self._set_cpu_affinity,
                set_gpu_affinity=self._set_gpu_affinity,
                nnodes=None,
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
            return

        self._blocked.discard(chore_id)
        fut.chore_id = chore_id
        fut.chore_obj = chore
        self._running_chores.add(chore_id)
        self._futures.add(fut)

        self._free_cores -= chore.resources.num_tasks * chore.resources.cores_per_task
        self._free_gpus -= chore.resources.num_tasks * chore.resources.gpus_per_task
        time.sleep(buffer_time)

    def _submit_until_ooresources(self, buffer_time: float) -> bool:
        """
        Submit as many chores as possible until out-of-resources

        Parameters
        ----------
        buffer_time : float
            The amount of time in seconds buffer the submission of :obj:`Chore`'s
        """

        deferred = deque()
        submitted_any = False

        while self._ready:
            chore_id = self._ready.popleft()
            chore = self._chores_by_id[chore_id]

            if self._can_submit_now(chore):
                self._submit_one(chore_id, buffer_time)
                submitted_any = True
            else:
                deferred.append(chore_id)

        self._ready = deferred
        return submitted_any

    def _log_worker(self, delay: float) -> None:
        """
        Function that updates the logs every so often
        """
        done = (
            len(self._ready) == 0
            and len(self._running_chores) == 0
            and len(self._blocked) == 0
        )
        while not done:
            self._log_progress()
            time.sleep(delay)
            done = (
                len(self._ready) == 0
                and len(self._running_chores) == 0
                and len(self._blocked) == 0
            )

    def run(
        self,
        buffer_time: float = 1.0,
        log_delay: float = 5.0,
        adaptive: bool = True,
        dynopro: bool = False,
        processing_strategy: FutureProcessingStrategy | None = None,
        dashboard: bool = False,
        restarting: bool = False,
    ) -> None:
        """
        Runs the 'Super Loop' until there are no more ready, running or blocked
        :obj:`Chore`'s

        Parameters
        ----------
        buffer_time : float
            The amount of time in seconds buffer the submission of :obj:`Chore`'s
        log_delay : float
            The amount of time in seconds that the log files will be written to
        adaptive : bool
            Whether or not :obj:`Chore`'s should be submitted adaptively, defaults
            to True
        dynopro : bool
            Currently does nothing because I couldn't figure out what it did
            to begin with.
        dashboard : bool
            Whether or not the dashboard should be started
        restarting : bool
            Whether :method:`run` is being invoked for the first time or after a
            restart file has been loaded


        Notes
        -----
        Each loop iteration:

        #. Refreshes available resources
        #. Prints a progress snapshot
        #. Submits new chores until resources are exhausted
        #. processes completed chores using a FutureProcessingStrategy:
            * User implementation if a processing_strategy is used
            * AdaptiveStrategy if adaptive=True
            * NonAdaptiveStrategy otherwise
        """

        if dashboard:
            status_file = self._base_dir / "status.json"
            setup_dashboard(str(status_file))

        if processing_strategy:
            proc_strat = processing_strategy
        elif adaptive:
            proc_strat = AdaptiveStrategy(self)
        else:
            proc_strat = NonAdaptiveStrategy(self)

        buffer_time = 0.0 if buffer_time is None else float(buffer_time)

        self._flux_handle.rpc("resource.drain", {"targets": "0"}).get()
        with flux.job.FluxExecutor() as executor:
            self._executor = executor

            if restarting:
                self._logger.info("=== RESTARTING WORKFLOW ENVIRONMENT ===")
            else:
                self._logger.info("=== ENTERING WORKFLOW ENVIRONMENT ===")
                self._start_time = time.perf_counter()

            # starting a thread to continueally log every {log_delay} seconds
            logging_thread = threading.Thread(
                target=self._log_worker,
                args=(log_delay,),
                daemon=True,
            )
            logging_thread.start()

            self._validate_chores()

            ### Super Loop ###
            done = (
                len(self._ready) == 0
                and len(self._running_chores) == 0
                and len(self._blocked) == 0
            )
            while not done:
                self._check_resources()
                self._submit_until_ooresources(buffer_time=buffer_time)
                proc_strat.process_futures(buffer_time=buffer_time)

                done = (
                    len(self._ready) == 0
                    and len(self._running_chores) == 0
                    and len(self._blocked) == 0
                )
            ### Super Loop ###

            end = time.perf_counter()
            logging_thread.join()
            self._log_progress()
            self._logger.info("=== EXITING WORKFLOW ENVIRONMENT ===")
            self._logger.info(
                "Workflow took %.4f seconds to run.", end - self._start_time
            )
