import concurrent.futures

from matensemble.strategy.process_futures_strategy_base import FutureProcessingStrategy
from pathlib import Path


class NonAdaptiveStrategy(FutureProcessingStrategy):
    """
    Implements the FutureProcessingStrategy interface. Non adaptive, does not
    try and submit a new task after each new one is completed
    """

    def __init__(self, manager) -> None:
        """
        Parameters
        ----------
        manager: SuperFluxManager
            manages resources and calls this method based on its strategy

        Return
        ------
        None
        """

        self.manager = manager

    def process_futures(self, buffer_time) -> None:
        """
        Process the FluxExecutorFuture objects and update the status lists in
        the manager and adaptively submit the

        Parameters
        ----------
        buffer_time: int | float
            The amount of time in seconds to wait for the future objects to
            complete
        """

        completed, self.manager.futures = concurrent.futures.wait(
            self.manager.futures, timeout=buffer_time
        )
        for fut in completed:
            self.manager.running_tasks.remove(fut.task)

            task = getattr(fut, "task", getattr(fut, "task_", "<unknown>"))
            workdir = Path(
                getattr(fut, "workdir", self.manager.paths.out_dir / str(task))
            )
            stdout = workdir / "stdout"
            stderr = workdir / "stderr"

            try:
                return_code = fut.result()  # raises if the task submission/run failed
            except Exception:
                self.manager.failed_tasks.append((task, fut.job_spec))
                self.manager.logger.exception(
                    "TASK FAILED: task=%s | workdir=%s | stdout=%s | stderr=%s",
                    task,
                    workdir,
                    stdout,
                    stderr,
                )
                continue

            if return_code != 0:
                self.manager.failed_tasks.append((task, fut.job_spec))
                self.manager.logger.error(
                    "TASK NONZERO EXIT: task=%s rc=%s | workdir=%s | stdout=%s | stderr=%s",
                    task,
                    return_code,
                    workdir,
                    stdout,
                    stderr,
                )
                continue

            self.manager.completed_tasks.append(task)

            if len(self.manager.completed_tasks) % self.manager.write_restart_freq == 0:
                self.manager.create_restart_file()
