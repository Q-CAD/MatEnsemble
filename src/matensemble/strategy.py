import traceback

import concurrent.futures
import pickle

import flux
import flux.job.executor

from datetime import datetime
from pathlib import Path

from abc import ABC, abstractmethod

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

        completed, self.manager._futures = concurrent.futures.wait(
            self.manager._futures, timeout=buffer_time
        )

        had_failure = False
        for fut in completed:
            chore_id = getattr(fut, "chore_id")
            chore = getattr(fut, "chore_obj")
            self.manager._running_chores.remove(chore_id)

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
                self.manager._record_failure(
                    chore_id,
                    reason="exception",
                    exception=f"{type(e).__name__}: {e}",
                )
                self.manager._fail_dependents(chore_id)
                had_failure = True
                continue

            # rc 134 is a double free or corruption error caused by lammps-symmetrix in the
            # in the frontier image during lammps cleanup
            # the function will still complete successfully and produce a result.pickle file
            # so if we can safely ignore the return code
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
                self.manager._record_failure(
                    chore_id,
                    reason=f"nonzero_exit:{rc}",
                )
                self.manager._fail_dependents(chore_id)
                had_failure = True
                continue

            self.manager._completed_chores.append(chore_id)

            for dep_id in self.manager._dependents.get(chore_id, []):
                self.manager._remaining_deps[dep_id] -= 1
                if self.manager._remaining_deps[dep_id] == 0:
                    self.manager._ready.append(dep_id)
                    self.manager._blocked.discard(dep_id)

            # adaptively submit another chore
            self.manager._submit_until_ooresources(
                buffer_time=buffer_time,
                dynopro=getattr(self.manager, "_dynopro", False),
            )

            if self.manager._write_restart_freq and (
                len(self.manager._completed_chores) % self.manager._write_restart_freq
                == 0
            ):
                self.manager._make_restart()

        if had_failure:
            return


class NonAdaptiveStrategy(FutureProcessingStrategy):
    """
    An implementation of the :obj:`FutureProcessingStrategy` which will not adaptively
    submit new :obj:`Chore`'s as incoming chores are completed.
    """

    def __init__(self, manager) -> None:
        super().__init__(manager)

    def process_futures(self, buffer_time) -> None:
        completed, self.manager._futures = concurrent.futures.wait(
            self.manager._futures, timeout=buffer_time
        )

        had_failure = False
        for fut in completed:
            chore_id = getattr(fut, "chore_id")
            chore = getattr(fut, "chore_obj")
            self.manager._running_chores.remove(chore_id)

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
                self.manager._record_failure(
                    chore_id,
                    reason="exception",
                    exception=f"{type(e).__name__}: {e}",
                )
                self.manager._fail_dependents(chore_id)
                had_failure = True
                continue

            # rc 134 is a double free or corruption error caused by lammps-symmetrix in the
            # in the frontier image during lammps cleanup
            # the function will still complete successfully and produce a result.pickle file
            # so if we can safely ignore the return code
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
                self.manager._record_failure(
                    chore_id,
                    reason=f"nonzero_exit:{rc}",
                )
                self.manager._fail_dependents(chore_id)
                had_failure = True
                continue

            self.manager._completed_chores.append(chore_id)

            for dep_id in self.manager._dependents.get(chore_id, []):
                self.manager._remaining_deps[dep_id] -= 1
                if self.manager._remaining_deps[dep_id] == 0:
                    self.manager._ready.append(dep_id)
                    self.manager._blocked.discard(dep_id)

            if self.manager._write_restart_freq and (
                len(self.manager._completed_chores) % self.manager._write_restart_freq
                == 0
            ):
                self.manager._make_restart()

        if had_failure:
            return


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
        completed, self.manager._futures = concurrent.futures.wait(
            self.manager._futures, timeout=buffer_time
        )

        had_failure = False
        for fut in completed:
            chore_id = getattr(fut, "chore_id")
            chore = getattr(fut, "chore_obj")
            chore_name = chore_id.removeprefix("chore-").rsplit("-", 1)[0]
            self.manager._running_chores.remove(chore_id)

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
                self.manager._record_failure(
                    chore_id,
                    reason="exception",
                    exception=f"{type(e).__name__}: {e}",
                )
                self.manager._fail_dependents(chore_id)
                had_failure = True
                continue

            # rc 134 is a double free or corruption error caused by lammps-symmetrix in the
            # in the frontier image during lammps cleanup
            # the function will still complete successfully and produce a result.pickle file
            # so if we can safely ignore the return code
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
                self.manager._record_failure(
                    chore_id,
                    reason=f"nonzero_exit:{rc}",
                )
                self.manager._fail_dependents(chore_id)
                had_failure = True
                continue

            self.manager._completed_chores.append(chore_id)

            for dep_id in self.manager._dependents.get(chore_id, []):
                self.manager._remaining_deps[dep_id] -= 1
                if self.manager._remaining_deps[dep_id] == 0:
                    self.manager._ready.append(dep_id)
                    self.manager._blocked.discard(dep_id)

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

            # adaptively submit another chore
            self.manager._submit_until_ooresources(
                buffer_time=buffer_time,
                dynopro=getattr(self.manager, "_dynopro", False),
            )

            if self.manager._write_restart_freq and (
                len(self.manager._completed_chores) % self.manager._write_restart_freq
                == 0
            ):
                self.manager._make_restart()

        if had_failure:
            return


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
