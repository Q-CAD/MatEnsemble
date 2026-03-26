from __future__ import annotations

import json
import pickle

from matensemble.fluxlet import Fluxlet
from matensemble.chore import Chore
from matensemble.model import ChoreType, Resources


class RecordingExecutor:
    def __init__(self):
        self.submitted_jobspecs = []

    def submit(self, jobspec):
        from tests.conftest import FakeFuture

        fut = FakeFuture(jobspec=jobspec, result_value=0)
        self.submitted_jobspecs.append(jobspec)
        return fut


def test_fluxlet_submit_writes_chore_spec_and_sets_jobspec_fields(tmp_path):
    workdir = tmp_path / "out" / "chore-1"
    chore = Chore(
        id="chore-1",
        command=["python", "-m", "task"],
        chore_type=ChoreType.PYTHON,
        resources=Resources(
            num_tasks=2, cores_per_task=3, gpus_per_task=1, mpi=True, env={"A": "B"}
        ),
        workdir=workdir,
        func_module="tasks",
        func_qualname="run",
    )

    executor = RecordingExecutor()
    fluxlet = Fluxlet(handle=None)
    fut = fluxlet.submit(
        executor,
        chore,
        set_cpu_affinity=True,
        set_gpu_affinity=True,
        nnodes=2,
    )

    jobspec = executor.submitted_jobspecs[0]
    assert jobspec.command == ["python", "-m", "task"]
    assert jobspec.cwd == str(workdir.resolve())
    assert jobspec.stdout.endswith("stdout")
    assert jobspec.stderr.endswith("stderr")
    # assert jobspec.env == {"A": "B"}
    assert jobspec.num_nodes == 2
    assert jobspec.shell_options["mpi"] == "pmi2"
    assert jobspec.shell_options["cpu-affinity"] == "per-task"
    assert jobspec.shell_options["gpu-affinity"] == "per-task"

    with chore.spec_path.open("rb") as f:
        stored_chore = pickle.load(f)
    assert stored_chore.id == "chore-1"

    debug = json.loads((workdir / "chore.json").read_text())
    assert debug["id"] == "chore-1"
    assert fut.chore_id == "chore-1"
    assert fut.chore_obj.id == "chore-1"
    assert fut.chore_spec is jobspec


def test_fluxlet_submit_inherit_env_true_copies_manager_environment(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MATENSEMBLE_MANAGER_ENV", "from-manager")

    workdir = tmp_path / "out" / "chore-inherit"
    chore = Chore(
        id="chore-inherit",
        command=["python", "-m", "task"],
        chore_type=ChoreType.EXECUTABLE,
        resources=Resources(
            inherit_env=True,
            env={"LOCAL_ONLY": "1", "MATENSEMBLE_MANAGER_ENV": "overridden"},
        ),
        workdir=workdir,
    )

    executor = RecordingExecutor()
    fluxlet = Fluxlet(handle=None)
    fluxlet.submit(executor, chore)

    jobspec = executor.submitted_jobspecs[0]
    assert jobspec.environment["MATENSEMBLE_MANAGER_ENV"] == "overridden"
    assert jobspec.environment["LOCAL_ONLY"] == "1"


def test_fluxlet_submit_inherit_env_false_uses_only_chore_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MATENSEMBLE_MANAGER_ENV", "from-manager")

    workdir = tmp_path / "out" / "chore-no-inherit"
    chore = Chore(
        id="chore-no-inherit",
        command=["python", "-m", "task"],
        chore_type=ChoreType.EXECUTABLE,
        resources=Resources(inherit_env=False, env={"LOCAL_ONLY": "1"}),
        workdir=workdir,
    )

    executor = RecordingExecutor()
    fluxlet = Fluxlet(handle=None)
    fluxlet.submit(executor, chore)

    jobspec = executor.submitted_jobspecs[0]
    assert jobspec.environment == {"LOCAL_ONLY": "1"}
