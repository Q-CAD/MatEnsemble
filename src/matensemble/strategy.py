import traceback
import time

import concurrent.futures
import flux
import flux.job.executor

from datetime import datetime
from pathlib import Path

from matensemble.fluxlet import Fluxlet

from abc import ABC, abstractmethod


class FutureProcessingStrategy(ABC):
    @abstractmethod
    def process_futures(self, buffer_time) -> None:
        pass


# TODO: Update this to be consistent with new API
class AdaptiveStrategy(FutureProcessingStrategy):
    """
    Implements the FutureProcessingStrategy interface. Processes futures
    adaptively. Every time a future object is completed it will submit a job
    right then and there rather than waiting.
    """

    def __init__(self, manager) -> None:
        """
        Parameters
        ----------
        manager : FluxManager
            manages resources and calls this method based on its strategy

        Return
        ------
        None
        """

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
                        f"exception={repr(e)}\n"
                        f"{tb}\n"
                    ),
                )
                self.manager._logger.error("JOB FAILED: job=%s", job_id)

                self.manager._failed_jobs.append((job_id, fut.job_spec))
                self.manager._logger.exception("JOB FAILED: job=%s", job_id)
                self.manager._record_failure(
                    job_id,
                    reason="exception",
                    exception=repr(e),
                )
                self.manager._fail_dependents(job_id)
                continue

            if rc != 0:
                append_text(
                    job.workdir / "stderr",
                    f"\n\n===== MATENSEMBLE: NONZERO EXIT =====\n"
                    f"job={job_id} rc={rc}\n",
                )

                self.manager._failed_jobs.append((job_id, fut.job_spec))
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

            # unlock dependents
            for dep_id in self.manager._dependents.get(job_id, []):
                self.manager._remaining_deps[dep_id] -= 1
                if self.manager._remaining_deps[dep_id] == 0:
                    self.manager._ready.append(dep_id)
                    self.manager._blocked.discard(dep_id)

            # adaptively submit
            self.manager._submit_one(buffer_time)

            if self.manager._write_restart_freq and (
                len(self.manager._completed_jobs) % self.manager._write_restart_freq
                == 0
            ):
                self.manager._create_restart_file()


class NonAdaptiveStrategy(FutureProcessingStrategy):
    """
    Implements the FutureProcessingStrategy interface. Simply processes the
    futures and updates the five queues in the FluxManager
    """

    def __init__(self, manager) -> None:
        """
        Parameters
        ----------
        manager : FluxManager
            manages resources and calls this method based on its strategy

        Return
        ------
        None
        """

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
                        f"exception={repr(e)}\n"
                        f"{tb}\n"
                    ),
                )
                self.manager._logger.error("JOB FAILED: job=%s", job_id)

                self.manager._failed_jobs.append((job_id, fut.job_spec))
                self.manager._logger.exception("JOB FAILED: job=%s", job_id)
                continue

            if rc != 0:
                append_text(
                    job.workdir / "stderr",
                    f"\n\n===== MATENSEMBLE: NONZERO EXIT =====\n"
                    f"job={job_id} rc={rc}\n"
                    f"See workflow log for details: {self.manager.paths.verbose_log_file}\n",
                )

                self.manager._failed_jobs.append((job_id, fut.job_spec))
                self.manager._logger.error(
                    "JOB NONZERO EXIT: job=%s rc=%s | workdir=%s | stdout=%s | stderr=%s",
                    job_id,
                    rc,
                    job.workdir,
                    job.workdir / "stdout",
                    job.workdir / "stderr",
                )
                continue

            self.manager._completed_jobs.append(job_id)

            # unlock dependents
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
