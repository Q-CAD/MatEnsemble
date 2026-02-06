"""
Logging + status utilities for MatEnsemble.

This module creates a per-run workflow directory:

.. code-block:: text

   <JOBID>_matensemble_workflow/
     status.log
     logs/
       <timestamp>_matensemble_workflow.log
     out/
       <task-id>/
         stdout
         stderr

The status file is overwritten on each update so users can monitor progress:

.. code-block:: console

   watch -n 1 cat <JOBDIR>/status.log

The verbose log file contains timestamped messages and is always written to disk.
"""

from __future__ import annotations

import logging
import sys
import os

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class WorkflowPaths:
    """
    Dataclass to hold the locations for the status file, log file, and output
    directory.
    """

    base_dir: Path
    status_file: Path
    logs_dir: Path
    out_dir: Path
    verbose_log_file: Path


def job_id() -> str:
    """
    Helper function to get the job_id to set the name of the
    <SLURM_JOB_ID>_matensemble_workflow directory

    Return
    ------
    str
        The SLURM_JOB_ID environment variable or the process ID
    """

    return os.environ.get("SLURM_JOB_ID") or f"local-{os.getpid()}"


def timestamp_for_filename() -> str:
    """
    Helper function to get the date and time

    Return
    ------
    str
        The date and time in the format YYYY-MM-DD_HH-mm-SS
    """

    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def atomic_write_text(path: Path, text: str) -> None:
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
    Class to handle writing to the status file

    Attributes
    ----------
    path: Path
        the path to the status file

    Methods
    -------
    render(
        pending: int,
        running: int,
        completed: int,
        failed: int,
        free_cores: int,
        free_gpus: int,
        include_updated_line: bool = True,
    )
        Creates the output for the status of the program as a str and returns it
    update(
        self,
        pending: int,
        running: int,
        completed: int,
        failed: int,
        free_cores: int,
        free_gpus: int,
    )
        Calls the render method to create the text and then calls
        atomic_write_text to write to the status file
    """

    def __init__(self, path: Path):
        self.path = path

    @staticmethod
    def render(
        pending: int,
        running: int,
        completed: int,
        failed: int,
        free_cores: int,
        free_gpus: int,
        include_updated_line: bool = True,
    ) -> str:
        updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # fixed-width columns to keep it stable in watch/tmux
        lines = []
        if include_updated_line:
            lines.append(f"UPDATED:   {updated}")
            lines.append("")

        lines.append("JOBS:        Pending     Running   Completed     Failed")
        lines.append(
            f"            {pending:>8}   {running:>8}   {completed:>8}   {failed:>8}"
        )
        lines.append("")
        lines.append("RESOURCES:  Free Cores   Free GPUs")
        lines.append(f"            {free_cores:>8}   {free_gpus:>8}")
        lines.append("")  # trailing newline-friendly

        return "\n".join(lines)

    def update(
        self,
        pending: int,
        running: int,
        completed: int,
        failed: int,
        free_cores: int,
        free_gpus: int,
    ) -> None:
        text = self.render(pending, running, completed, failed, free_cores, free_gpus)
        atomic_write_text(self.path, text)


def create_workflow_paths(base_dir: str | Path | None = None) -> WorkflowPaths:
    """
    Helper function to create the output directory

    Returns
    -------
    WorkflowPaths
        An object that encapsulates all of the paths for the output
        files/directoriesof the matensemble workflow

    Notes
    -----
    Structure of the output directory:

        <SLURM_JOB_ID>_matensemble_workflow/
            |- status.log
            |- logs/
                |- <timestamp>_matensemble_workflow.log
            |- out/
                |- <output_of_workflow>

    """

    base_dir = Path(base_dir) if base_dir is not None else Path.cwd()

    workflow_dir = base_dir / f"{job_id()}_matensemble_workflow"
    logs_dir = workflow_dir / "logs"
    out_dir = workflow_dir / "out"
    status_file = workflow_dir / "status.log"

    workflow_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    verbose_log_file = (
        workflow_dir / f"{timestamp_for_filename()}_matensemble_workflow.log"
    )

    return WorkflowPaths(
        base_dir=workflow_dir,
        status_file=status_file,
        logs_dir=logs_dir,
        out_dir=out_dir,
        verbose_log_file=verbose_log_file,
    )


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
