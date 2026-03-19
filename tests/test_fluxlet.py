from __future__ import annotations

import json
import pickle

from matensemble.fluxlet import Fluxlet
from matensemble.job import Job
from matensemble.model import JobFlavor, Resources


class RecordingExecutor:
    def __init__(self):
        self.submitted_jobspecs = []

    def submit(self, jobspec):
        from tests.conftest import FakeFuture

        fut = FakeFuture(jobspec=jobspec, result_value=0)
        self.submitted_jobspecs.append(jobspec)
        return fut


def test_fluxlet_submit_writes_job_spec_and_sets_jobspec_fields(tmp_path):
    workdir = tmp_path / "out" / "job-1"
    job = Job(
        id="job-1",
        command=["python", "-m", "task"],
        flavor=JobFlavor.PYTHON,
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
        job,
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

    with job.spec_path.open("rb") as f:
        stored_job = pickle.load(f)
    assert stored_job.id == "job-1"

    debug = json.loads((workdir / "job.json").read_text())
    assert debug["id"] == "job-1"
    assert fut.job_id == "job-1"
    assert fut.job_obj.id == "job-1"
    assert fut.job_spec is jobspec
