import traceback

import concurrent.futures
import flux
import flux.job.executor

from datetime import datetime
from pathlib import Path

from abc import ABC, abstractmethod


class FutureProcessingStrategy(ABC):
    """
    The Base Class that all FutureProcessingStrategy's must extend in order to
    be compliant with how the :obj:`FluxManager` uses them
    """

    @abstractmethod
    def process_futures(self, buffer_time) -> None:
        pass


class AdaptiveStrategy(FutureProcessingStrategy):
    """
    An implementation of the :obj:`FutureProcessingStrategy` which will adaptively
    submit new :obj:`Job`'s as incoming jobs are completed.
    """

    def __init__(self, manager) -> None:
        """
        AdaptiveStrategy constructor

        Parameters
        ----------
        manager : FluxManager
            The :obj:`FluxManager` that holds all of the queues and functions
            to handle them.
        """

        self.manager = manager

    def process_futures(self, buffer_time: float) -> None:
        completed, self.manager._futures = concurrent.futures.wait(
            self.manager._futures, timeout=buffer_time
        )
        """
        Process the future objects as :obj:`Job`'s complete 

        Parameters
        ----------
        buffer_time : float 
            The amount of time to wait between jobs being completed. 
        """

        for fut in completed:
            job_id = fut.job_id
            job = fut.job_obj
            self.manager._running_jobs.remove(job_id)

            try:
                rc = fut.result()
            except Exception as e:
                tb = traceback.format_exc()
                stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                append_text(
                    job.workdir / "stderr",
                    (
                        f"\n\n===== MATENSEMBLE WRAPPER ERROR ({stamp}) =====\n"
                        f"job={job_id}\n"
                        f"workdir={job.workdir}\n"
                        f"{type(e).__name__}: {e}"
                        f"{tb}\n"
                    ),
                )
                self.manager._logger.exception("JOB FAILED: job=%s", job_id)
                self.manager._record_failure(
                    job_id,
                    reason="exception",
                    exception=f"{type(e).__name__}: {e}",
                )
                self.manager._fail_dependents(job_id)
                continue

            if rc != 0:
                append_text(
                    job.workdir / "stderr",
                    f"\n\n===== MATENSEMBLE: NONZERO EXIT =====\n"
                    f"job={job_id} rc={rc}\n",
                )
                self.manager._logger.error(
                    "JOB NONZERO EXIT: job=%s rc=%s | workdir=%s | stdout=%s | stderr=%s",
                    job_id,
                    rc,
                    job.workdir,
                    job.workdir / "stdout",
                    job.workdir / "stderr",
                )
                self.manager._record_failure(
                    job_id,
                    reason=f"nonzero_exit:{rc}",
                )
                self.manager._fail_dependents(job_id)
                continue

            self.manager._completed_jobs.append(job_id)

            for dep_id in self.manager._dependents.get(job_id, []):
                self.manager._remaining_deps[dep_id] -= 1
                if self.manager._remaining_deps[dep_id] == 0:
                    self.manager._ready.append(dep_id)
                    self.manager._blocked.discard(dep_id)

            # adaptively submit another job
            self.manager._submit_until_ooresources(buffer_time=buffer_time)

            if self.manager._write_restart_freq and (
                len(self.manager._completed_jobs) % self.manager._write_restart_freq
                == 0
            ):
                self.manager._make_restart()


class NonAdaptiveStrategy(FutureProcessingStrategy):
    """
    An implementation of the :obj:`FutureProcessingStrategy` which will not adaptively
    submit new :obj:`Job`'s as incoming jobs are completed.
    """

    def __init__(self, manager) -> None:
        self.manager = manager

    def process_futures(self, buffer_time) -> None:
        completed, self.manager._futures = concurrent.futures.wait(
            self.manager._futures, timeout=buffer_time
        )

        for fut in completed:
            job_id = fut.job_id
            job = fut.job_obj
            self.manager._running_jobs.remove(job_id)

            try:
                rc = fut.result()
            except Exception as e:
                tb = traceback.format_exc()
                stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                append_text(
                    job.workdir / "stderr",
                    (
                        f"\n\n===== MATENSEMBLE WRAPPER ERROR ({stamp}) =====\n"
                        f"job={job_id}\n"
                        f"workdir={job.workdir}\n"
                        f"{type(e).__name__}: {e}"
                        f"{tb}\n"
                    ),
                )
                self.manager._logger.exception("JOB FAILED: job=%s", job_id)
                self.manager._record_failure(
                    job_id,
                    reason="exception",
                    exception=f"{type(e).__name__}: {e}",
                )
                self.manager._fail_dependents(job_id)
                continue

            if rc != 0:
                append_text(
                    job.workdir / "stderr",
                    f"\n\n===== MATENSEMBLE: NONZERO EXIT =====\n"
                    f"job={job_id} rc={rc}\n",
                )
                self.manager._logger.error(
                    "JOB NONZERO EXIT: job=%s rc=%s | workdir=%s | stdout=%s | stderr=%s",
                    job_id,
                    rc,
                    job.workdir,
                    job.workdir / "stdout",
                    job.workdir / "stderr",
                )
                self.manager._record_failure(
                    job_id,
                    reason=f"nonzero_exit:{rc}",
                )
                self.manager._fail_dependents(job_id)
                continue

            self.manager._completed_jobs.append(job_id)

            for dep_id in self.manager._dependents.get(job_id, []):
                self.manager._remaining_deps[dep_id] -= 1
                if self.manager._remaining_deps[dep_id] == 0:
                    self.manager._ready.append(dep_id)
                    self.manager._blocked.discard(dep_id)

            if self.manager._write_restart_freq and (
                len(self.manager._completed_jobs) % self.manager._write_restart_freq
                == 0
            ):
                self.manager._make_restart()


def append_text(path: Path, text: str) -> None:
    """
    Append some text to the end of a given file. Used for writing error messages
    to stderr on a specific job

    Parameters
    ----------
    path : Path
        The path to the file to write to
    text : str
        The text to append to the file

    Return
    ------
    None
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
