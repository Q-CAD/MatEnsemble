import logging
import json
import sys
import os
import tempfile

from datetime import datetime
from pathlib import Path


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
        self, path: Path, nnodes: int, cores_per_node: int, gpus_per_node: int
    ) -> None:
        self.path = path
        self.nnodes = nnodes
        self.cores_per_node = cores_per_node
        self.gpus_per_node = gpus_per_node

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

        with tempfile.NamedTemporaryFile("w", dir=self.path.parent, delete=False) as tf:
            tf.write(json.dumps(data))
            temp_name = tf.name

        os.replace(temp_name, self.path)


def _setup_status_writer(
    path: Path, nnodes: int, cores_per_node: int, gpus_per_node: int
):  # -> StatusWriter
    return StatusWriter(
        path=path,
        nnodes=nnodes,
        cores_per_node=cores_per_node,
        gpus_per_node=gpus_per_node,
    )


def _setup_logger(base_dir: Path) -> logging.Logger:
    logger = logging.getLogger("matensemble")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Prevent duplicate handlers if setup is called twice
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_file = base_dir / f"matensemble_workflow.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    hint = (
        f"Status file: {base_dir}/status.json\n"
        f"Verbose log: {base_dir}/matensemble_workflow.log\n"
        f"Job outputs: {base_dir}/out\n"
    )
    print(hint, file=sys.stderr)

    logger.info(f"Workflow initialized at {base_dir}")
    return logger
