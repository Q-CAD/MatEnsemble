import pickle

from pathlib import Path
from dataclasses import dataclass
from enum import StrEnum, auto


@dataclass(frozen=True)
class OutputReference:
    """
    An object to encapsulate the result of a chore as the input to another chore.
    """

    chore_id: str
    spec_file: Path

    def __str__(self) -> str:
        """
        Return the deserialized result of the referenced chore as a string.
        """
        # TODO: Make sure that this file exists and add some exception handling
        dep_result = self.spec_file.parent / "result.pkl"
        with dep_result.open("rb") as f:
            return str(pickle.load(f))


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


class ChoreType(StrEnum):
    """
    The different types that a chore can be. As of right now there are two types
    of Chores, Python chores and Executable chores. Python chores are delayed function
    calls and executable chores are paths to executable files.
    """

    PYTHON = auto()
    EXECUTABLE = auto()
