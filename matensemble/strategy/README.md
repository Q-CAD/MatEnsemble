## matensemble/strategy/ 

Holds the base classes and implementations of the TaskSubmissionStrategy and 
FutureProcessingStrategy objects that the SuperFluxManager will use

### TaskSubmissionStrategy 
```python
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
```

### FutureProcessingStrategy
```python
import flux

from abc import ABC, abstractmethod


class FutureProcessingStrategy(ABC):
    @abstractmethod
    def process_futures(self, buffer_time) -> None:
        pass
```


The user can implement their own stratgey that follows these interfaces 
and inject it into the SuperFluxManager.poolexecutor() method as a parameter
to get their own behavior. 
