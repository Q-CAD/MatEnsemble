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
    def __init__(
        self,
        job_list: list[Job],
        base_dir: Path,
        write_restart_freq: int | None = 100,
        set_cpu_affinity: bool = True,
        set_gpu_affinity: bool = True,
        restart_file: str | None = None,
    ) -> None:
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
        pass

    def _load_restart(self, path: str | Path) -> None:
        """
        Load the pickled restart file and pick up where it left off.
        """
        pass

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

    def _get_allocation_info(self) -> tuple:
        """
        Gets the total number of nodes (minus one for the flux broker) and the
        number of cores and gpus per node.

        Return
        ------
        tuple
            Each of the computed values (nnodes, cores_per_node, gpus_per_node)
        """

        rpc = flux.resource.list.resource_list(self._flux_handle)
        resources = rpc.get()

        nnodes = len(resources.all.ranks)
        total_cores = resources.all.ncores
        total_gpus = resources.all.ngpus

        cores_per_node = total_cores // nnodes
        gpus_per_node = total_gpus // nnodes

        return (nnodes - 1, cores_per_node, gpus_per_node)

    def _record_failure(
        self,
        job_id: str,
        reason: str,
        *,
        upstream: str | None = None,
        exception: str | None = None,
    ) -> None:
        self._failed_jobs.append(
            {
                "job_id": job_id,
                "reason": reason,
                "upstream": upstream,
                "exception": exception,
            }
        )

    def _fail_dependents(self, failed_job_id: str) -> None:
        for dep_id in self._dependents.get(failed_job_id, []):
            if dep_id in self._completed_jobs or dep_id in self._running_jobs:
                continue

            # remove from pending structures
            try:
                self._ready.remove(dep_id)
            except ValueError:
                pass
            self._blocked.discard(dep_id)

            # avoid duplicate entries
            already_failed = any(item["job_id"] == dep_id for item in self._failed_jobs)
            if not already_failed:
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

            # recurse downward
            self._fail_dependents(dep_id)

    def _job_fits_allocation(self, job: Job) -> bool:
        needed_cores = job.resources.num_tasks * job.resources.cores_per_task
        needed_gpus = job.resources.num_tasks * job.resources.gpus_per_task

        total_cores = self._nnodes * self._cores_per_node
        total_gpus = self._nnodes * self._gpus_per_node

        return needed_cores <= total_cores and needed_gpus <= total_gpus

    def _validate_jobs(self) -> None:
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

    def _submit_one(self, job_id: str, buffer_time: float) -> None:
        spec = self._jobs_by_id[job_id]
        self._ready.popleft()
        self._blocked.discard(job_id)
        fut = self._fluxlet.submit(
            self._executor,
            self._jobs_by_id[job_id],
            set_cpu_affinity=self._set_cpu_affinity,
            set_gpu_affinity=self._set_gpu_affinity,
            nnodes=None,
        )
        fut.job_id = job_id
        self._running_jobs.add(job_id)
        self._futures.add(fut)
        self._free_cores -= spec.resources.num_tasks * spec.resources.cores_per_task
        self._free_gpus -= spec.resources.num_tasks * spec.resources.gpus_per_task
        time.sleep(buffer_time)

    def _can_submit_now(self, job: Job) -> bool:
        needed_cores = job.resources.num_tasks * job.resources.cores_per_task
        needed_gpus = job.resources.num_tasks * job.resources.gpus_per_task
        return self._free_cores >= needed_cores and self._free_gpus >= needed_gpus

    def _submit_until_ooresources(self, buffer_time: float) -> None:
        deferred = deque()
        submitted_any = False

        while self._ready:
            job_id = self._ready.popleft()
            job = self._jobs_by_id[job_id]

            if self._can_submit_now(job):
                self._blocked.discard(job_id)
                self._submit_one(job_id, buffer_time)
                submitted_any = True
            else:
                deferred.append(job_id)

        self._ready = deferred
        return submitted_any

    def run(
        self,
        buffer_time: int = 1,
        adaptive: bool = True,
        dynopro: bool = False,
        processing_strategy: FutureProcessingStrategy | None = None,
    ) -> None:
        """ """

        if processing_strategy:
            proc_strat = processing_strategy
        elif adaptive:
            proc_strat = AdaptiveStrategy(self)
        else:
            proc_strat = NonAdaptiveStrategy(self)

        self._flux_handle.rpc("resource.drain", {"targets": "0"}).get()
        with flux.job.FluxExecutor() as executor:
            # set executor in manager so that strategies can access it
            self._executor = executor

            self._logger.info("=== ENTERING WORKFLOW ENVIRONMENT ===")
            start = time.perf_counter()

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
            self._logger.info("=== EXITING WORKFLOW ENVIRONMENT  ===")
            self._logger.info(f"Workflow took {(end - start):.4f} seconds to run.")
