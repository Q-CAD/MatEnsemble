# matensemble/strategy/__init__.py

from . import submission_strategy_base
from . import process_futures_strategy_base
from . import cpu_affine_strategy
from . import gpu_affine_strategy
from . import dynopro_strategy
from . import adaptive_strategy
from . import not_adaptive_strategy

__all__ = [
    "submission_strategy_base",
    "process_futures_strategy_base",
    "cpu_affine_strategy",
    "gpu_affine_strategy",
    "dynopro_strategy",
    "adaptive_strategy",
    "not_adaptive_strategy",
]


"""
Strategy implementations for submission and future-processing in MatEnsemble.

`Strategy Pattern <https://refactoring.guru/design-patterns/strategy>`__ algorithms for management of flux tasks.

The original MatEnsemble code was heavily nested inside of a big 'super loop'
that was nested up to seven times and very difficult to reason about and read.

``
while True:
    # ...
``

To solve this issue we implemented the 'Strategy Pattern' seperating the
different ways of submitting tasks and processing futures into their own modules.
Isolating them and making them much easier to maintain and taking the super loop
down from 145 lines of heavily nested code to just 9 lines.

Inside of the SuperFlux Manager when the poolexecutor method is called, based on
the parameters it is given, it will decide the strategy it will use to submit
tasks and process futures.

The User can also provide their own strategy to the poolexecutor and it will
inject it into the 'super loop' and use the user defined strategy. To implement
your own strategy it needs to follow the interface::

    ``
    # submission_strategy_base
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

    # process_future_strategy_base
    class FutureProcessingStrategy(ABC):
        @abstractmethod
        def process_futures(self, buffer_time) -> None:
            pass

    ``

Following these two interfaces you can add any functionality that you may need.
"""
