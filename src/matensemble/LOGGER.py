import logging
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
