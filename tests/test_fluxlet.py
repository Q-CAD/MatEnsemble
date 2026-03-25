from __future__ import annotations

import json
import pickle

from matensemble.fluxlet import Fluxlet
from matensemble.chore import Chore
from matensemble.model import ChoreType, Resources


class RecordingExecutor:
    def __init__(self):
        self.submitted_chorespecs = []

    def submit(self, chorespec):
        from tests.conftest import FakeFuture

        fut = FakeFuture(chorespec=chorespec, result_value=0)
        self.submitted_chorespecs.append(chorespec)
        return fut


def test_fluxlet_submit_writes_chore_spec_and_sets_chorespec_fields(tmp_path):
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

    chorespec = executor.submitted_chorespecs[0]
    assert chorespec.command == ["python", "-m", "task"]
    assert chorespec.cwd == str(workdir.resolve())
    assert chorespec.stdout.endswith("stdout")
    assert chorespec.stderr.endswith("stderr")
    # assert chorespec.env == {"A": "B"}
    assert chorespec.num_nodes == 2
    assert chorespec.shell_options["mpi"] == "pmi2"
    assert chorespec.shell_options["cpu-affinity"] == "per-task"
    assert chorespec.shell_options["gpu-affinity"] == "per-task"

    with chore.spec_path.open("rb") as f:
        stored_chore = pickle.load(f)
    assert stored_chore.id == "chore-1"

    debug = json.loads((workdir / "chore.json").read_text())
    assert debug["id"] == "chore-1"
    assert fut.chore_id == "chore-1"
    assert fut.chore_obj.id == "chore-1"
    assert fut.chore_spec is chorespec
