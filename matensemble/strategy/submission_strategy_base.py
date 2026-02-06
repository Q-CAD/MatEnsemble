"""
A strategy class allowing the manager (SuperFluxManager) to use a
different strategy for submitting tasks based on the parameters given to it at
run time
"""

import flux

from abc import ABC, abstractmethod


class TaskSubmissionStrategy(ABC):
    @abstractmethod
    def submit_until_ooresources(
        self, task_arg_list, task_dir_list, buffer_time
    ) -> None:
        pass

    @abstractmethod
    def submit(
        self, task, tasks_per_job, task_args, task_dir
    ) -> flux.job.executor.FluxExecutorFuture:
        pass
