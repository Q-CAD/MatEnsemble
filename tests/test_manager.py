from __future__ import annotations

from collections import deque

import pytest

from matensemble.job import Job
from matensemble.manager import FluxManager
from matensemble.model import JobFlavor, Resources


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, *args, **kwargs):
        self.messages.append(("info", args, kwargs))

    def error(self, *args, **kwargs):
        self.messages.append(("error", args, kwargs))

    def exception(self, *args, **kwargs):
        self.messages.append(("exception", args, kwargs))


class DummyStatusWriter:
    def __init__(self):
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


class FakeFluxlet:
    def __init__(self, handle=None):
        self.handle = handle
        self.calls = []
        self.raise_exc = None

    def submit(self, executor, job, set_cpu_affinity, set_gpu_affinity, nnodes):
        if self.raise_exc is not None:
            raise self.raise_exc
        from tests.conftest import FakeFuture

        fut = FakeFuture(result_value=0)
        fut.job_id = job.id
        fut.job_obj = job
        fut.job_spec = {"job": job.id}
        fut.workdir = str(job.workdir)
        self.calls.append(job.id)
        return fut


@pytest.fixture
def manager_factory(monkeypatch, tmp_path):
    import matensemble.manager as manager_mod

    monkeypatch.setattr(manager_mod.FluxManager, "_get_allocation_info", lambda self: (2, 4, 1))
    monkeypatch.setattr(manager_mod, "_setup_status_writer", lambda *args, **kwargs: DummyStatusWriter())
    monkeypatch.setattr(manager_mod, "_setup_logger", lambda *args, **kwargs: DummyLogger())
    monkeypatch.setattr(manager_mod, "Fluxlet", FakeFluxlet)

    def _build(job_list):
        return FluxManager(job_list=job_list, base_dir=tmp_path / "workflow")

    return _build


def _job(tmp_path, job_id, deps=(), num_tasks=1, cores_per_task=1, gpus_per_task=0):
    return Job(
        id=job_id,
        command=["echo", job_id],
        flavor=JobFlavor.EXECUTABLE,
        resources=Resources(num_tasks=num_tasks, cores_per_task=cores_per_task, gpus_per_task=gpus_per_task),
        workdir=tmp_path / job_id,
        deps=deps,
    )


def test_manager_initializes_dependency_queues(manager_factory, tmp_path):
    a = _job(tmp_path, "job-a")
    b = _job(tmp_path, "job-b", deps=("job-a",))
    manager = manager_factory([a, b])

    assert list(manager._ready) == ["job-a"]
    assert manager._blocked == {"job-b"}
    assert manager._dependents == {"job-a": ["job-b"], "job-b": []}
    assert manager._remaining_deps == {"job-a": 0, "job-b": 1}


def test_job_fits_allocation(manager_factory, tmp_path):
    job = _job(tmp_path, "job-a", num_tasks=2, cores_per_task=4, gpus_per_task=1)
    manager = manager_factory([job])
    assert manager._job_fits_allocation(job) is True

    too_big = _job(tmp_path, "job-b", num_tasks=3, cores_per_task=4)
    assert manager._job_fits_allocation(too_big) is False


def test_record_failure_deduplicates(manager_factory, tmp_path):
    manager = manager_factory([_job(tmp_path, "job-a")])
    manager._record_failure("job-a", reason="x")
    manager._record_failure("job-a", reason="y")
    assert manager._failed_jobs == [{"job_id": "job-a", "reason": "x", "upstream": None, "exception": None}]


def test_fail_dependents_cascades(manager_factory, tmp_path):
    a = _job(tmp_path, "job-a")
    b = _job(tmp_path, "job-b", deps=("job-a",))
    c = _job(tmp_path, "job-c", deps=("job-b",))
    manager = manager_factory([a, b, c])

    manager._fail_dependents("job-a")
    reasons = {item["job_id"]: item["reason"] for item in manager._failed_jobs}
    assert reasons["job-b"] == "dependency_failed"
    assert reasons["job-c"] == "dependency_failed"
    assert "job-b" not in manager._blocked
    assert "job-c" not in manager._blocked


def test_validate_jobs_marks_oversized_job_and_dependents_failed(manager_factory, tmp_path):
    a = _job(tmp_path, "job-a", num_tasks=20, cores_per_task=4)
    b = _job(tmp_path, "job-b", deps=("job-a",))
    manager = manager_factory([a, b])

    manager._validate_jobs()
    reasons = {item["job_id"]: item["reason"] for item in manager._failed_jobs}
    assert reasons["job-a"] == "job_exceeds_allocation"
    assert reasons["job-b"] == "dependency_failed"


def test_submit_one_success_updates_running_and_resources(manager_factory, tmp_path, monkeypatch):
    manager = manager_factory([_job(tmp_path, "job-a", num_tasks=2, cores_per_task=2, gpus_per_task=1)])
    manager._executor = object()
    manager._free_cores = 8
    manager._free_gpus = 2
    monkeypatch.setattr("matensemble.manager.time.sleep", lambda *_: None)

    manager._submit_one("job-a", buffer_time=0.0)

    assert manager._running_jobs == {"job-a"}
    assert len(manager._futures) == 1
    assert manager._free_cores == 4
    assert manager._free_gpus == 0


def test_submit_one_failure_records_submit_exception(manager_factory, tmp_path, monkeypatch):
    manager = manager_factory([_job(tmp_path, "job-a")])
    manager._executor = object()
    manager._free_cores = 8
    manager._free_gpus = 2
    manager._fluxlet.raise_exc = RuntimeError("submit broke")
    monkeypatch.setattr("matensemble.manager.time.sleep", lambda *_: None)

    manager._submit_one("job-a", buffer_time=0.0)

    assert manager._failed_jobs[0]["job_id"] == "job-a"
    assert manager._failed_jobs[0]["reason"] == "submit_exception"
    assert "job-a" not in manager._running_jobs


def test_submit_until_ooresources_defers_jobs_that_do_not_fit_now(manager_factory, tmp_path, monkeypatch):
    a = _job(tmp_path, "job-a", num_tasks=1, cores_per_task=4)
    b = _job(tmp_path, "job-b", num_tasks=1, cores_per_task=6)
    manager = manager_factory([a, b])
    manager._ready = deque(["job-a", "job-b"])
    manager._executor = object()
    manager._free_cores = 4
    manager._free_gpus = 1
    monkeypatch.setattr("matensemble.manager.time.sleep", lambda *_: None)

    submitted_any = manager._submit_until_ooresources(buffer_time=0.0)

    assert submitted_any is True
    assert manager._running_jobs == {"job-a"}
    assert list(manager._ready) == ["job-b"]


def test_log_progress_updates_status_writer(manager_factory, tmp_path):
    manager = manager_factory([_job(tmp_path, "job-a")])
    manager._ready = deque(["job-a"])
    manager._blocked = {"job-b"}
    manager._running_jobs = {"job-c"}
    manager._completed_jobs = ["job-d"]
    manager._failed_jobs = [{"job_id": "job-e", "reason": "x", "upstream": None, "exception": None}]
    manager._free_cores = 3
    manager._free_gpus = 1

    manager._log_progress()

    assert manager._status_writer.updates[-1] == {
        "pending": 2,
        "running": 1,
        "completed": 1,
        "failed": 1,
        "free_cores": 3,
        "free_gpus": 1,
    }


def test_run_uses_explicit_processing_strategy(manager_factory, tmp_path, monkeypatch):
    manager = manager_factory([_job(tmp_path, "job-a")])
    manager._check_resources = lambda: setattr(manager, "_free_cores", 8) or setattr(manager, "_free_gpus", 1)
    manager._log_progress = lambda: None
    manager._validate_jobs = lambda: None
    monkeypatch.setattr("matensemble.manager.time.perf_counter", lambda: 0.0)
    monkeypatch.setattr("matensemble.manager.time.sleep", lambda *_: None)

    calls = []

    class ExplicitStrategy:
        def process_futures(self, buffer_time):
            calls.append(buffer_time)
            manager._running_jobs.clear()
            manager._blocked.clear()
            manager._ready.clear()

    def fake_submit_until(buffer_time):
        if manager._ready:
            manager._running_jobs.add(manager._ready.popleft())
        return True

    manager._submit_until_ooresources = fake_submit_until

    manager.run(buffer_time=None, adaptive=False, processing_strategy=ExplicitStrategy())

    assert calls == [0.0]
