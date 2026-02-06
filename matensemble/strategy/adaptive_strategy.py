import concurrent.futures
import flux
import time

from matensemble.strategy.process_futures_strategy_base import FutureProcessingStrategy
from matensemble.fluxlet import Fluxlet
from pathlib import Path


class AdaptiveStrategy(FutureProcessingStrategy):
    """
    Implements the FutureProcessingStrategy interface. Processes futures
    adaptively. Every time a future object is completed it will submit a job
    right then and there rather than waiting.
    """

    def __init__(self, manager, task_arg_list=None, task_dir_list=None) -> None:
        """
        Parameters
        ----------
        manager: SuperFluxManager
            manages resources and calls this method based on its strategy
        task_arg_list: list[str | int]
            List of the order of tasks to be completed in
        task_dir_list: list[srt]
            List of directories for the completed tasks to go into

        Return
        ------
        None
        """

        self.manager = manager
        self.task_arg_list = task_arg_list
        self.task_dir_list = task_dir_list

    def submit(
        self, task, tasks_per_job, task_args, task_dir
    ) -> flux.job.executor.FluxExecutorFuture:
        """
        Creates a fluxlet object and submits the task

        Parameters
        ----------
        task: str | int
            The task to be submitted
        tasks_per_job: int
            The number of sub-tasks for the task to complete
        task_args: list[str | int | float | np.int64 | np.float64 | dict]
            The arguments for the task
        task_dir: str
            Where the task will

        Return
        ------
        flux.job.executor.FluxExecutorFuture
            A concurrent.futures.Future object representing the result of the
            task
        """

        fluxlet = Fluxlet(
            self.manager.flux_handle,
            tasks_per_job,
            self.manager.cores_per_task,
            self.manager.gpus_per_task,
        )
        fluxlet.job_submit(
            self.manager.executor,
            self.manager.gen_task_cmd,
            task,
            task_args,
            task_dir,
            base_out_dir=self.manager.paths.out_dir,
        )

        return fluxlet.future

    def adaptive_submit(self, buffer_time) -> None:
        """
        Submit pending tasks if you have resources and updates the
        status lists in the SuperFluxManager

        Parameters
        ----------
        buffer_time: int | float
            The time to sleep after submitting a job

        Return
        ------
        None
        """

        if (
            self.manager.tasks_per_job
            and self.task_arg_list is not None
            and self.manager.free_cores
            >= self.manager.tasks_per_job[0] * self.manager.cores_per_task
            and len(self.manager.pending_tasks)
        ):
            cur_tasks_per_job = int(self.manager.tasks_per_job[0])
            needed_cores = cur_tasks_per_job * self.manager.cores_per_task

            cur_task = self.manager.pending_tasks.popleft()
            cur_task_args = self.task_arg_list.popleft()

            if self.task_dir_list is not None:
                cur_task_dir = self.task_dir_list.popleft()
            else:
                cur_task_dir = None

            self.manager.futures.add(
                self.submit(
                    cur_task, self.manager.tasks_per_job[0], cur_task_args, cur_task_dir
                )
            )
            self.manager.running_tasks.add(cur_task)
            self.manager.tasks_per_job.popleft()

            self.manager.free_cores -= needed_cores

            time.sleep(buffer_time)

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
            self.adaptive_submit(buffer_time)

            if len(self.manager.completed_tasks) % self.manager.write_restart_freq == 0:
                self.manager.create_restart_file()
