import time
import logging

import flux
import flux.job

from pathlib import Path
from collections import deque

from matensemble.LOGGER import StatusWriter
from matensemble.pipeline.PIPELINE import Job
from matensemble.strategy.adaptive_strategy import AdaptiveStrategy
from matensemble.strategy.process_futures_strategy_base import FutureProcessingStrategy
from matensemble.FLUXLET import Fluxlet


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
        self._base_dir = base_dir
        
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
        self._completed_tasks = []
        self._failed_tasks = []
        self._futures = set()

        self._flux_handle = flux.Flux()
        self._fluxlet = Fluxlet(self._flux_handle)

        self._write_restart_freq = write_restart_freq
        self._status_writer = self._setup_status_writer()
        self._logger = self._setup_logger()

    def _setup_status_writer(self)""" -> StatusWriter""":
        pass



    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("matensemble")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # Prevent duplicate handlers if setup is called twice
        if logger.handlers:
            logger.handlers.clear()

        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        log_file = self._base_dir / f"{matensemble-datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

        return logger

    def _make_restart(self) -> None:
        """
        Pickle the current state of the manager and dump it to a file
        """
        pass

    def _load_restart(self) -> None:
        """
        Load the pickled restart file and pick up where it left off.
        """
        pass

    def _log_progress(self) -> None:
        """
        Update the status file and append a progress line in the log file
        """
        pass

    def _get_nnodes(self) -> tuple:
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

    def run(
        self,
        buffer_time: int | None = None,
        adaptive: bool = True,
        dynopro: bool = False,
        processing_strategy: FutureProcessingStrategy | None = None,
    ) -> None:
        """ """

        if processing_strategy:
            proc_strat = processing_strategy
        else:
            # TODO: Update the adaptive strategy to work with the new API
            proc_strat = AdaptiveStrategy(self)

        self._flux_handle.rpc("resource.drain", {"targets": "0"}).get()
        with flux.job.FluxExecutor() as executor:
            # set executor in manager so that strategies can access it
            self._executor = executor

            self._logger.info("=== ENTERING WORKFLOW ENVIRONMENT ===")
            start = time.perf_counter()

            # done = len
            # while not done:
