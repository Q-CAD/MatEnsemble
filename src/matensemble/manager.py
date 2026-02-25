# matensemble/manager.py

from __future__ import annotations

import time
import pickle
import flux
import concurrent.futures
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

from matensemble.logger import setup_workflow_logging
from matensemble.fluxlet import Fluxlet

if TYPE_CHECKING:
    from matensemble.pipeline.compile import TaskSpec


class SuperFluxManager:
    """
    DAG-aware Flux submission manager (MVP).

    Takes compiled TaskSpecs, submits tasks when dependencies are satisfied,
    and tracks completion/failure.
    """

    def __init__(
        self,
        tasks: list["TaskSpec"],
        *,
        base_dir: str | Path | None = None,
        write_restart_freq: int = 100,
        set_cpu_affinity: bool = True,
        set_gpu_affinity: bool = True,
    ) -> None:
        self.tasks_by_id = {t.id: t for t in tasks}
        self.dependents: dict[str, list[str]] = {t.id: [] for t in tasks}
        self.remaining_deps: dict[str, int] = {t.id: len(t.deps) for t in tasks}

        for t in tasks:
            for dep in t.deps:
                self.dependents[dep].append(t.id)

        self.ready = deque([tid for tid, n in self.remaining_deps.items() if n == 0])
        self.blocked = set(self.tasks_by_id.keys()) - set(self.ready)

        self.running_tasks: set[str] = set()
        self.completed_tasks: list[str] = []
        self.failed_tasks: list[tuple[str, object]] = []
        self.futures: set = set()

        self.flux_handle = flux.Flux()
        self.fluxlet = Fluxlet(self.flux_handle)

        self.write_restart_freq = write_restart_freq
        self.set_cpu_affinity = set_cpu_affinity
        self.set_gpu_affinity = set_gpu_affinity

        self.logger, self.status, self.paths = setup_workflow_logging(base_dir=base_dir)

    def create_restart_file(self) -> None:
        """
        MVP restart: store completed/running/ready/blocked/failed.
        """
        state = {
            "completed": self.completed_tasks,
            "running": list(self.running_tasks),
            "ready": list(self.ready),
            "blocked": list(self.blocked),
            "failed": self.failed_tasks,
        }
        out = self.paths.base_dir / f"restart_{len(self.completed_tasks)}.dat"
        pickle.dump(state, open(out, "wb"))

    def check_resources(self) -> None:
        """
        Same as your existing implementation (free cores/gpus from Flux).
        """
        res = flux.resource.list.resource_list(self.flux_handle).get()
        self.free_gpus = res.free.ngpus
        self.free_cores = res.free.ncores

    def log_progress(self) -> None:
        pending = len(self.ready) + len(self.blocked)
        self.status.update(
            pending=pending,
            running=len(self.running_tasks),
            completed=len(self.completed_tasks),
            failed=len(self.failed_tasks),
            free_cores=self.free_cores,
            free_gpus=self.free_gpus,
        )

    def _can_submit(self, spec: "TaskSpec") -> bool:
        need_cores = int(spec.resources.cores_per_task) * int(spec.resources.num_tasks)
        need_gpus = int(spec.resources.gpus_per_task) * int(spec.resources.num_tasks)
        return self.free_cores >= need_cores and self.free_gpus >= need_gpus

    def _submit_one(self, spec: "TaskSpec") -> None:
        fut = self.fluxlet.submit_spec(
            self.executor,
            spec,
            set_cpu_affinity=self.set_cpu_affinity,
            set_gpu_affinity=self.set_gpu_affinity,
        )
        self.futures.add(fut)
        self.running_tasks.add(spec.id)

        # optimistic local decrement (next loop refreshes from Flux anyway)
        self.free_cores -= int(spec.resources.cores_per_task) * int(
            spec.resources.num_tasks
        )
        self.free_gpus -= int(spec.resources.gpus_per_task) * int(
            spec.resources.num_tasks
        )

    def run(self, *, buffer_time: float = 0.2) -> None:
        """
        Minimal DAG scheduler loop:
        - submit from ready while resources allow
        - wait for completions
        - on completion: mark done, unlock dependents
        """
        with flux.job.FluxExecutor() as executor:
            self.executor = executor

            done = (
                (len(self.ready) == 0)
                and (len(self.running_tasks) == 0)
                and (len(self.blocked) == 0)
            )
            while not done:
                self.check_resources()
                self.log_progress()

                # submit as many READY tasks as possible
                while self.ready:
                    tid = self.ready[0]
                    spec = self.tasks_by_id[tid]
                    if not self._can_submit(spec):
                        break
                    self.ready.popleft()
                    self.blocked.discard(tid)
                    self._submit_one(spec)

                # process futures
                completed, self.futures = concurrent.futures.wait(
                    self.futures, timeout=buffer_time
                )
                for fut in completed:
                    tid = fut.task
                    self.running_tasks.remove(tid)

                    try:
                        rc = fut.result()
                    except Exception:
                        self.failed_tasks.append((tid, fut.job_spec))
                        self.logger.exception("TASK FAILED: task=%s", tid)
                        continue

                    if rc != 0:
                        self.failed_tasks.append((tid, fut.job_spec))
                        self.logger.error("TASK NONZERO EXIT: task=%s rc=%s", tid, rc)
                        continue

                    self.completed_tasks.append(tid)

                    # unlock dependents
                    for dep_id in self.dependents.get(tid, []):
                        self.remaining_deps[dep_id] -= 1
                        if self.remaining_deps[dep_id] == 0:
                            self.ready.append(dep_id)
                            self.blocked.discard(dep_id)

                    if self.write_restart_freq and (
                        len(self.completed_tasks) % self.write_restart_freq == 0
                    ):
                        self.create_restart_file()

                done = (
                    (len(self.ready) == 0)
                    and (len(self.running_tasks) == 0)
                    and (len(self.blocked) == 0)
                )
