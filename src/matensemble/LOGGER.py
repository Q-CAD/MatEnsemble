import logging
import json
import sys
import os

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def update_status(path: Path, text: str) -> None:
    """
    Helper function so `watch cat status.log` never sees a half-written file.

    Parameters
    ---------
    path: Path
        The path to the status file
    text: str
        The str to write to the status file

    Return
    ------
    None
    """

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class StatusWriter:
    """
    Class to handle updating the status file

    Attributes
    ----------
    path : Path
        the path to the status file
    nnodes : int
        The number of nodes that flux is managing (total_allocation - 1 for flux borker)
    cores_per_node : int
        The number of CPU cores that are available on each node
    gpus_per_node : int
        The number of GPUs that are available on each node

    """

    def __init__(
        self, path: Path, allocation_information: tuple[int, int, int]
    ) -> None:
        self.path = path
        self.nnodes = allocation_information[0]
        self.cores_per_node = allocation_information[1]
        self.gpus_per_node = allocation_information[2]

    def update(
        self,
        pending: int,
        running: int,
        completed: int,
        failed: int,
        free_cores: int,
        free_gpus: int,
    ) -> None:
        data = {
            "nodes": self.nnodes,
            "coresPerNode": self.cores_per_node,
            "gpusPerNode": self.gpus_per_node,
            "pending": pending,
            "running": running,
            "completed": completed,
            "failed": failed,
            "freeCores": free_cores,
            "freeGpus": free_gpus,
        }
        self.path.write_text(json.dumps(data))
