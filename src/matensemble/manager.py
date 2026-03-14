import time

import flux
import flux.job

from pathlib import Path
from collections import deque

from matensemble.logger import _setup_logger, _setup_status_writer
from matensemble.job import Job
from matensemble.strategy import (
    AdaptiveStrategy,
    NonAdaptiveStrategy,
    FutureProcessingStrategy,
)
from matensemble.fluxlet import Fluxlet


class FluxManager:
    """
    The :obj:`FluxManager` takes a list of :obj:`Job`'s and manages their submission
    dependencies and output organization.

    Attributes
    ----------
    _base_dir : Path
        The base directory where the output will be placed.
    _jobs_by_id : dict
        A dictionary of job_id's to :obj:`Job`
    _dependents : dict
        A dictionary of job_id's to a list of job_id's that they depend on
    _remaining_deps : dict
        A dictionary of job_id's to integers (the number of dependencies remaining)
    _ready : collections.deque
        A double ended queue of :obj:`Job`'s that are ready for submission
    _blocked : set
        A set of :obj:`Job`'s that are waiting on their dependencies to resolve
    _running_jobs : set
        A set of :obj:`Job`'s that are currently running
    _completed_jobs : list
        A list of :obj:`Job`'s that have completed succesfully
    _failed_jobs : list
        A list of :obj:`Job`'s that failed
    _futures : set
        A set of future objects representing the completion of running jobs
    _flux_handle : flux.Flux
        A flux handle
    _fluxlet : matensemble.Fluxlet
        A :obj:`Fluxlet` that is where all the jobs are submitted
    _write_restart_freq : int
        The number of jobs to be completed before pickling a restart file
    _nnodes : int
        The number of nodes available on the allocaiton minus one for flux broker
    _cores_per_node : int
        The number of cores that are on each node
    _gpus_per_node : int
        The number of gpus that are on each node
    _status_writer : StatusWriter
        A :obj:`StatusWriter` for logging the status of the workflow in JSON
    _logger : logging.Logger
        A :obj:`Logger` to log the progress of the workflow
    _set_cpu_affinity : bool
        Whether or not CPU affinity will be set
    _set_gpu_affinity : bool
        Whether or not GPU affinity will be set
    """

    def __init__(
        self,
        job_list: list[Job],
        base_dir: Path,
        write_restart_freq: int | None = 100,
        set_cpu_affinity: bool = True,
        set_gpu_affinity: bool = True,
        restart_file: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        job_list : list
            A list of :obj:`Job`'s that need to be submitted
        base_dir : Path
            The base directory of the workflow
        write_restart_freq : int
            The number of jobs to be completed before pickling a restart file
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

        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

        self._jobs_by_id = {job.id: job for job in job_list}
        self._dependents = {job.id: [] for job in job_list}
        self._remaining_deps = {job.id: len(job.deps) for job in job_list}

        for job in job_list:
            for dep in job.deps:
                self._dependents[dep].append(job.id)

        self._ready = deque(
            [
                job_id
                for job_id, num_deps in self._remaining_deps.items()
                if num_deps == 0
            ]
        )
        self._blocked = set(self._jobs_by_id.keys()) - set(self._ready)

        self._running_jobs = set()
        self._completed_jobs = []
        self._failed_jobs = []
        self._futures = set()

        self._flux_handle = flux.Flux()
        self._fluxlet = Fluxlet(self._flux_handle)

        self._write_restart_freq = write_restart_freq

        self._nnodes, self._cores_per_node, self._gpus_per_node = (
            self._get_allocation_info()
        )
        self._status_writer = _setup_status_writer(
            self._base_dir / "status.json",
            nnodes=self._nnodes,
            cores_per_node=self._cores_per_node,
            gpus_per_node=self._gpus_per_node,
        )
        self._logger = _setup_logger(self._base_dir)

        self._set_cpu_affinity = set_cpu_affinity
        self._set_gpu_affinity = set_gpu_affinity

        if restart_file:
            self._load_restart(restart_file)

    def _make_restart(self) -> None:
        """
        Pickle the current state of the manager and dump it to a file
        """
        raise NotImplementedError("Restart checkpoints are not implemented yet")

    def _load_restart(self, path: str | Path) -> None:
        """
        Load the pickled restart file and pick up where it left off.

        Parameters
        ----------
        path : str, Path
            The path to the restart file.
        """
        raise NotImplementedError("Restart checkpoints are not implemented yet")

    def _log_progress(self) -> None:
        """
        Update the status file and append a progress line in the log file
        """
        pending = len(self._ready) + len(self._blocked)
        self._status_writer.update(
            pending=pending,
            running=len(self._running_jobs),
            completed=len(self._completed_jobs),
            failed=len(self._failed_jobs),
            free_cores=self._free_cores,
            free_gpus=self._free_gpus,
        )

        self._logger.info(
            "JOBS: Pending=%d Running=%d Completed=%d Failed=%d | RESOURCES: Free_cores=%d Free_gpus=%d",
            pending,
            len(self._running_jobs),
            len(self._completed_jobs),
            len(self._failed_jobs),
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

    def _job_fits_allocation(self, job: Job) -> bool:
        """
        Checks whether the given job is too big to be submitted

        Parameters
        ----------
        job : Job
            The :obj:`Job` to check if it will fit in the allocation
        """

        needed_cores = job.resources.num_tasks * job.resources.cores_per_task
        needed_gpus = job.resources.num_tasks * job.resources.gpus_per_task

        total_cores = self._nnodes * self._cores_per_node
        total_gpus = self._nnodes * self._gpus_per_node

        return needed_cores <= total_cores and needed_gpus <= total_gpus

    def _validate_jobs(self) -> None:
        """
        Calls :method:`_job_fits_allocation()` on each job given to the manager to make sure
        that they all fit. If a given job does not fit it will be discarded.
        """

        for job_id, job in self._jobs_by_id.items():
            if not self._job_fits_allocation(job):
                self._record_failure(
                    job_id,
                    reason="job_exceeds_allocation",
                )

                try:
                    self._ready.remove(job_id)
                except ValueError:
                    pass
                self._blocked.discard(job_id)

                self._logger.error(
                    "JOB INVALID: job=%s requires more resources than the allocation can provide",
                    job_id,
                )
                self._fail_dependents(job_id)

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

    def _can_submit_now(self, job: Job) -> bool:
        """
        Checks to see if there are enough resources to submit the given :obj:`Job`
        """

        needed_cores = job.resources.num_tasks * job.resources.cores_per_task
        needed_gpus = job.resources.num_tasks * job.resources.gpus_per_task
        return self._free_cores >= needed_cores and self._free_gpus >= needed_gpus

    def _has_failed(self, job_id: str) -> bool:
        """
        Checks if a given job_id has failed
        """

        return any(item["job_id"] == job_id for item in self._failed_jobs)

    def _record_failure(
        self,
        job_id: str,
        reason: str,
        *,
        upstream: str | None = None,
        exception: str | None = None,
    ) -> None:
        """
        Logs the failure of a job with its reason
        """

        if self._has_failed(job_id):
            return

        self._failed_jobs.append(
            {
                "job_id": job_id,
                "reason": reason,
                "upstream": upstream,
                "exception": exception,
            }
        )

    def _fail_dependents(self, failed_job_id: str) -> None:
        """
        Cascades the failure of one job to all of it dependents to avoid
        deadlocks.
        """

        for dep_id in self._dependents.get(failed_job_id, []):
            if dep_id in self._completed_jobs or dep_id in self._running_jobs:
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
                    upstream=failed_job_id,
                )
                self._logger.error(
                    "JOB SKIPPED: job=%s because dependency %s failed",
                    dep_id,
                    failed_job_id,
                )

            self._fail_dependents(dep_id)

    def _submit_one(self, job_id: str, buffer_time: float) -> None:
        """
        Submits a :obj:`Job` and does book-keeping all the queues and resources
        count

        Parameters
        ----------
        buffer_time : float
            The amount of time in seconds buffer the submission of :obj:`Job`'s
        """

        job = self._jobs_by_id[job_id]

        try:
            fut = self._fluxlet.submit(
                self._executor,
                job,
                set_cpu_affinity=self._set_cpu_affinity,
                set_gpu_affinity=self._set_gpu_affinity,
                nnodes=None,
            )
        except Exception as e:
            self._logger.exception("JOB SUBMIT FAILED: job=%s", job_id)
            self._record_failure(
                job_id,
                reason="submit_exception",
                exception=repr(e),
            )
            self._fail_dependents(job_id)
            self._blocked.discard(job_id)
            return

        self._blocked.discard(job_id)
        fut.job_id = job_id
        fut.job_obj = job
        self._running_jobs.add(job_id)
        self._futures.add(fut)

        self._free_cores -= job.resources.num_tasks * job.resources.cores_per_task
        self._free_gpus -= job.resources.num_tasks * job.resources.gpus_per_task
        time.sleep(buffer_time)

    def _submit_until_ooresources(self, buffer_time: float) -> bool:
        """
        Submit as many jobs as possible until out-of-resources

        Parameters
        ----------
        buffer_time : float
            The amount of time in seconds buffer the submission of :obj:`Job`'s
        """

        deferred = deque()
        submitted_any = False

        while self._ready:
            job_id = self._ready.popleft()
            job = self._jobs_by_id[job_id]

            if self._can_submit_now(job):
                self._submit_one(job_id, buffer_time)
                submitted_any = True
            else:
                deferred.append(job_id)

        self._ready = deferred
        return submitted_any

    def run(
        self,
        buffer_time: float = 1.0,
        adaptive: bool = True,
        dynopro: bool = False,
        processing_strategy: FutureProcessingStrategy | None = None,
    ) -> None:
        """
        Runs the 'Super Loop' until there are no more ready, running or blocked
        :obj:`Job`'s

        Parameters
        ----------
        buffer_time : float
            The amount of time in seconds buffer the submission of :obj:`Job`'s
        adaptive : bool
            Whether or not :obj:`Job`'s should be submitted adaptively, defaults
            to True
        dynopro : bool
            Currently does nothing because I couldn't figure out what it did
            to begin with.

        Notes
        -----
        Each loop iteration:

        #. Refreshes available resources
        #. Prints a progress snapshot
        #. Submits new jobs until resources are exhausted
        #. processes completed jobs using a FutureProcessingStrategy:
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

        self._flux_handle.rpc("resource.drain", {"targets": "0"}).get()
        with flux.job.FluxExecutor() as executor:
            self._executor = executor

            self._logger.info("=== ENTERING WORKFLOW ENVIRONMENT ===")
            start = time.perf_counter()

            self._validate_jobs()

            done = (
                len(self._ready) == 0
                and len(self._running_jobs) == 0
                and len(self._blocked) == 0
            )

            while not done:
                self._check_resources()
                self._log_progress()
                self._submit_until_ooresources(buffer_time=buffer_time)
                proc_strat.process_futures(buffer_time=buffer_time)

                done = (
                    len(self._ready) == 0
                    and len(self._running_jobs) == 0
                    and len(self._blocked) == 0
                )

            end = time.perf_counter()
            self._logger.info("=== EXITING WORKFLOW ENVIRONMENT ===")
            self._logger.info("Workflow took %.4f seconds to run.", end - start)
