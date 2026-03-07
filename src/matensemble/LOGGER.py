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
        self.path.write_text(json.dumps(data))


def setup_workflow_logging(
    logger_name: str = "matensemble",
    base_dir: str | Path | None = None,
    console: bool | None = None,
) -> tuple[logging.Logger, StatusWriter, WorkflowPaths]:
    """
    Creates:
      - workflow directory tree
      - status writer (status.log)
      - verbose python logger (timestamped file, optional console)

    Parameters
    ----------
    logger_name: str
        The name of the logger
    base_dir: str | Path
        Where the matensemble_workflow directory will be setup
    console: bool | None
        Whether or not we are in an interactive environment

    Return
    ------
    tuple
        a three element tuple with the logger, StatusWriter and WorkflowPaths


    """
    paths = create_workflow_paths(base_dir)
    status = StatusWriter(paths.status_file)

    logger = logging.getLogger(logger_name)
    logger.setLevel(
        logging.DEBUG
    )  # file gets everything; handler levels control output
    logger.propagate = False

    # Prevent duplicate handlers if setup is called twice
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(paths.verbose_log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    if console is None:
        console = sys.stderr.isatty()

    if console:
        console_handler = logging.StreamHandler(stream=sys.stderr)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)
    else:
        pass

    # print hint for user to watch the file
    hint = (
        f"Status file: {paths.status_file}\n"
        f"Watch it with: watch -n 1 cat {paths.status_file}\n"
        f"Verbose log: {paths.verbose_log_file}\n"
        f"Task outputs: {paths.out_dir}\n"
    )
    print(hint, file=sys.stderr)

    logger.info("Workflow initialized at %s", paths.base_dir)
    return logger, status, paths
