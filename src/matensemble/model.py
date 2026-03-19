from dataclasses import dataclass
from enum import StrEnum, auto


@dataclass(frozen=True)
class OutputReference:
    """
    An object to encapsulate the result of a job as the input to another job
    """

    job_id: str


@dataclass
class Resources:
    """
    The resources that a wrapped flux job needs
    """

    num_tasks: int = 1
    cores_per_task: int = 1
    gpus_per_task: int = 0
    mpi: bool = False
    env: dict[str, str] | None = None
    inherit_env: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.num_tasks, int) or self.num_tasks < 1:
            raise ValueError("num_tasks must be an integer >= 1")

        if not isinstance(self.cores_per_task, int) or self.cores_per_task < 1:
            raise ValueError("cores_per_task must be an integer >= 1")

        if not isinstance(self.gpus_per_task, int) or self.gpus_per_task < 0:
            raise ValueError("gpus_per_task must be an integer >= 0")

        if not isinstance(self.mpi, bool):
            raise TypeError("mpi must be a bool")

        if self.env is not None:
            if not isinstance(self.env, dict):
                raise TypeError("env must be a dict[str, str] or None")
            for k, v in self.env.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    raise TypeError("env must be a dict[str, str]")

        if not isinstance(self.inherit_env, bool):
            raise TypeError("inherit_env must be a bool")


class JobFlavor(StrEnum):
    """
    The different flavors that a job can be. As of right now there are two types
    of Jobs, Python jobs and Executable jobs. Python jobs are delayed function
    calls and executable jobs are paths to executable files.
    """

    PYTHON = auto()
    EXECUTABLE = auto()
