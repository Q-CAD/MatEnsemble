import time
import flux

from matensemble.strategy.submission_strategy_base import TaskSubmissionStrategy
from matensemble.fluxlet import Fluxlet


class DynoproStrategy(TaskSubmissionStrategy):
    """
    Implements the TaskSubmissionStrategy interface. This is esentially the same
    as the CPUAffineStrategy but it uses the hetero_job_submit rather than the
    job_submit
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

    def submit_until_ooresources(
        self, task_arg_list, task_dir_list, buffer_time
    ) -> None:
        """
        Submit pending tasks until you are out of resources and updates the
        status lists in the SuperFluxManager

        Parameters
        ----------
        task_arg_list: list[str | int]
            List of the order of tasks to be completed in
        task_dir_list: list[srt]
            List of directories for the completed tasks to go into
        buffer_time: int | float
            The time to sleep after submitting all of your jobs

        Return
        ------
        None
        """

        while (
            self.manager.tasks_per_job
            and self.manager.free_cores
            >= self.manager.tasks_per_job[0] * self.manager.cores_per_task
            and len(self.manager.pending_tasks) > 0
        ):
            cur_tasks_per_job = int(self.manager.tasks_per_job[0])
            needed_cores = cur_tasks_per_job * self.manager.cores_per_task

            cur_task = self.manager.pending_tasks.popleft()
            cur_task_args = task_arg_list.popleft()

            if task_dir_list is not None:
                cur_task_dir = task_dir_list.popleft()
            else:
                cur_task_dir = None

            self.manager.futures.add(
                self.submit(cur_task, cur_tasks_per_job, cur_task_args, cur_task_dir)
            )
            self.manager.running_tasks.add(cur_task)
            self.manager.tasks_per_job.popleft()

            self.manager.free_cores -= needed_cores

            time.sleep(buffer_time)

    def submit(
        self, task, tasks_per_job, task_args, task_dir
    ) -> flux.job.executor.FluxExecutorFuture:
        """
        Creates a fluxlet object and submits the task with the hetero_job_submit
        rather than the regular submit

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

        if self.manager.nnodes is None or self.manager.gpus_per_node is None:
            raise ValueError(
                "ERROR: For dynopro provisioning, nnodes and gpus_per_node can not be None"
            )
        fluxlet = Fluxlet(
            self.manager.flux_handle,
            tasks_per_job,
            self.manager.cores_per_task,
            self.manager.gpus_per_task,
        )
        fluxlet.hetero_job_submit(
            self.manager.executor,
            nnodes=self.manager.nnodes,
            gpus_per_node=self.manager.gpus_per_node,
            command=self.manager.gen_task_cmd,
            task=task,
            task_args=task_args,
            task_directory=task_dir,
            base_out_dir=self.manager.paths.out_dir,
        )

        return fluxlet.future
