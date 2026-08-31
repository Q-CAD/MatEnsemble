import traceback

import concurrent.futures
import pickle

from datetime import datetime
from pathlib import Path

from abc import ABC, abstractmethod

from matensemble.chore import Chore
from matensemble.model import OutputReference


class FutureProcessingStrategy(ABC):
    """
    The Base Class that all FutureProcessingStrategy's must extend in order to
    be compliant with how the :obj:`FluxManager` uses them
    """

    def __init__(self, manager) -> None:
        self.manager = manager

    @abstractmethod
    def process_futures(self, buffer_time) -> None:
        """
        Must be implemented by the child classes
        """
        pass

    def _process_future(self, fut) -> tuple[bool, Chore]:
        """Record one completed future and return ``(succeeded, chore)``."""

        chore_id = getattr(fut, "chore_id")
        chore = getattr(fut, "chore_obj")

        try:
            rc = fut.result()
        except Exception as e:
            tb = traceback.format_exc()
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            append_text(
                chore.workdir / "stderr",
                (
                    f"\n\n===== MATENSEMBLE WRAPPER ERROR ({stamp}) =====\n"
                    f"chore={chore_id}\n"
                    f"workdir={chore.workdir}\n"
                    f"{type(e).__name__}: {e}"
                    f"{tb}\n"
                ),
            )
            self.manager._logger.exception("CHORE FAILED: chore=%s", chore_id)
            released_cores, released_gpus = self.manager._finish_chore(
                chore,
                failure_reason="exception",
                exception=f"{type(e).__name__}: {e}",
            )
            succeeded = False
        else:
            # rc 134 is a double free or corruption error caused by
            # lammps-symmetrix cleanup. The function still produces a valid
            # result.pickle, so preserve the existing successful treatment.
            if rc != 0 and rc != 134:
                append_text(
                    chore.workdir / "stderr",
                    f"\n\n===== MATENSEMBLE: NONZERO EXIT =====\nchore={chore_id} rc={rc}\n",
                )
                self.manager._logger.error(
                    "CHORE NONZERO EXIT: chore=%s rc=%s | workdir=%s | stdout=%s | stderr=%s",
                    chore_id,
                    rc,
                    chore.workdir,
                    chore.workdir / "stdout",
                    chore.workdir / "stderr",
                )
                released_cores, released_gpus = self.manager._finish_chore(
                    chore,
                    failure_reason=f"nonzero_exit:{rc}",
                )
                succeeded = False
            else:
                released_cores, released_gpus = self.manager._finish_chore(chore)
                succeeded = True

        self.manager._logger.info(
            "CHORE FINISHED: chore=%s state=%s released_cores=%d released_gpus=%d",
            chore_id,
            "completed" if succeeded else "failed",
            released_cores,
            released_gpus,
        )
        self.manager._log_progress()

        if succeeded and self.manager._write_restart_freq and (
            len(self.manager._completed_chores) % self.manager._write_restart_freq == 0
        ):
            self.manager._make_restart()

        return succeeded, chore


class AdaptiveStrategy(FutureProcessingStrategy):
    """
    An implementation of the :obj:`FutureProcessingStrategy` which will adaptively
    submit new :obj:`Chore`'s as incoming chores are completed.
    """

    def __init__(self, manager) -> None:
        """
        AdaptiveStrategy constructor

        Parameters
        ----------
        manager : FluxManager
            The :obj:`FluxManager` that holds all of the queues and functions
            to handle them.
        """

        super().__init__(manager)

    def process_futures(self, buffer_time: float) -> None:
        """
        Process the future objects as :obj:`Chore`'s complete

        Parameters
        ----------
        buffer_time : float
            The amount of time to wait between chores being completed.
        """

        if not self.manager._futures:
            return

        completed, self.manager._futures = concurrent.futures.wait(
            self.manager._futures,
            timeout=buffer_time,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        for fut in completed:
            self._process_future(fut)

        if completed:
            # All resources released by this completion batch are visible before
            # filling the newly available capacity.
            self.manager._submit_until_ooresources(
                buffer_time=buffer_time,
                dynopro=getattr(self.manager, "_dynopro", False),
            )


class NonAdaptiveStrategy(FutureProcessingStrategy):
    """
    Process chores in discrete waves.

    All futures submitted in the current wave are allowed to finish before
    their newly ready dependents, or any remaining ready chores, can be
    submitted by the manager's next outer-loop iteration.
    """

    def __init__(self, manager) -> None:
        super().__init__(manager)

    def process_futures(self, buffer_time) -> None:
        if not self.manager._futures:
            return

        # Freeze the wave before waiting. Process and log each future as it
        # finishes so status is current, but never submit here; the manager's
        # next fill phase cannot begin until this entire snapshot is drained.
        wave = set(self.manager._futures)
        for fut in concurrent.futures.as_completed(wave):
            self.manager._futures.discard(fut)
            self._process_future(fut)


class UserStrategy(FutureProcessingStrategy):
    def __init__(
        self, manager, pipeline, processing_chore, processing_chore_resources, bolo_list
    ) -> None:
        super().__init__(manager)
        self.pipeline = pipeline
        self.proc_chore = processing_chore
        self.proc_chore_res = processing_chore_resources
        self.bolo_list = set(bolo_list)

        # if not isinstance(chore, Callable[..., Chore]):
        #     raise Exception(
        #         f"Error: Failed to construct UserStrategy due to Type Error"
        #     )

    def process_futures(self, buffer_time) -> None:
        if not self.manager._futures:
            return

        completed, self.manager._futures = concurrent.futures.wait(
            self.manager._futures,
            timeout=buffer_time,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        for fut in completed:
            chore_id = getattr(fut, "chore_id")
            succeeded, chore = self._process_future(fut)
            if not succeeded:
                continue

            chore_name = chore_id.removeprefix("chore-").rsplit("-", 1)[0]

            # --- Processing the chore and spawning the new one ---
            if chore_name == self.proc_chore:
                try:
                    # Trust boundary: result.pickle is written by matensemble.runtime_worker
                    # in this workflow's chore workdir only—do not load pickles from
                    # untrusted paths or third-party producers.
                    with (chore.workdir / "result.pickle").open("rb") as f:
                        chore_spec = pickle.load(f)
                    if chore_spec:
                        new_chore, new_out = self.pipeline._spawn_chore_from_spec(
                            chore_spec
                        )
                        self.pipeline._admit_spawned_chore(
                            new_chore, new_out, self.manager
                        )
                except Exception as e:
                    self.manager._logger.exception(
                        f"FAILED TO SPAWN CHORE: chore={self.proc_chore} | due the following Exception ->\n{e}"
                    )
            else:
                for bolo_name in self.bolo_list:
                    if bolo_name == chore_name:
                        try:
                            out_ref = OutputReference(chore_id, chore.workdir)
                            new_chore, new_out = self.pipeline._spawn_chore_from_name(
                                self.proc_chore, self.proc_chore_res, dependent=out_ref
                            )
                            self.pipeline._admit_spawned_chore(
                                new_chore, new_out, self.manager
                            )
                        except Exception as e:
                            self.manager._logger.exception(
                                f"FAILED TO SPAWN CHORE: proc_chore={self.proc_chore} "
                                f"bolo_match={chore_name} | due the following Exception ->\n{e}"
                            )

        if completed:
            self.manager._submit_until_ooresources(
                buffer_time=buffer_time,
                dynopro=getattr(self.manager, "_dynopro", False),
            )


def append_text(path: Path, text: str) -> None:
    """
    Append some text to the end of a given file. Used for writing error messages
    to stderr on a specific chore

    Parameters
    ----------
    path : Path
        The path to the file to write to
    text : str
        The text to append to the file

    Return
    ------
    None
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
