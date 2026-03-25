from __future__ import annotations

from collections import deque

import pytest

from matensemble.chore import Chore
from matensemble.manager import FluxManager
from matensemble.model import ChoreType, Resources


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

    def submit(self, executor, chore, set_cpu_affinity, set_gpu_affinity, nnodes):
        if self.raise_exc is not None:
            raise self.raise_exc
        from tests.conftest import FakeFuture

        fut = FakeFuture(result_value=0)
        fut.chore_id = chore.id
        fut.chore_obj = chore
        fut.chore_spec = {"chore": chore.id}
        fut.workdir = str(chore.workdir)
        self.calls.append(chore.id)
        return fut


@pytest.fixture
def manager_factory(monkeypatch, tmp_path):
    import matensemble.manager as manager_mod

    monkeypatch.setattr(
        manager_mod.FluxManager, "_get_allocation_info", lambda self: (2, 4, 1)
    )
    monkeypatch.setattr(
        manager_mod, "_setup_status_writer", lambda *args, **kwargs: DummyStatusWriter()
    )
    monkeypatch.setattr(
        manager_mod, "_setup_logger", lambda *args, **kwargs: DummyLogger()
    )
    monkeypatch.setattr(manager_mod, "Fluxlet", FakeFluxlet)

    def _build(chore_list):
        return FluxManager(chore_list=chore_list, base_dir=tmp_path / "workflow")

    return _build


def _chore(tmp_path, chore_id, deps=(), num_tasks=1, cores_per_task=1, gpus_per_task=0):
    return Chore(
        id=chore_id,
        command=["echo", chore_id],
        chore_type=ChoreType.EXECUTABLE,
        resources=Resources(
            num_tasks=num_tasks,
            cores_per_task=cores_per_task,
            gpus_per_task=gpus_per_task,
        ),
        workdir=tmp_path / chore_id,
        deps=deps,
    )


def test_manager_initializes_dependency_queues(manager_factory, tmp_path):
    a = _chore(tmp_path, "chore-a")
    b = _chore(tmp_path, "chore-b", deps=("chore-a",))
    manager = manager_factory([a, b])

    assert list(manager._ready) == ["chore-a"]
    assert manager._blocked == {"chore-b"}
    assert manager._dependents == {"chore-a": ["chore-b"], "chore-b": []}
    assert manager._remaining_deps == {"chore-a": 0, "chore-b": 1}


def test_chore_fits_allocation(manager_factory, tmp_path):
    chore = _chore(tmp_path, "chore-a", num_tasks=2, cores_per_task=4, gpus_per_task=1)
    manager = manager_factory([chore])
    assert manager._chore_fits_allocation(chore) is True

    too_big = _chore(tmp_path, "chore-b", num_tasks=3, cores_per_task=4)
    assert manager._chore_fits_allocation(too_big) is False


def test_record_failure_deduplicates(manager_factory, tmp_path):
    manager = manager_factory([_chore(tmp_path, "chore-a")])
    manager._record_failure("chore-a", reason="x")
    manager._record_failure("chore-a", reason="y")
    assert manager._failed_chores == [
        {"chore_id": "chore-a", "reason": "x", "upstream": None, "exception": None}
    ]


def test_fail_dependents_cascades(manager_factory, tmp_path):
    a = _chore(tmp_path, "chore-a")
    b = _chore(tmp_path, "chore-b", deps=("chore-a",))
    c = _chore(tmp_path, "chore-c", deps=("chore-b",))
    manager = manager_factory([a, b, c])

    manager._fail_dependents("chore-a")
    reasons = {item["chore_id"]: item["reason"] for item in manager._failed_chores}
    assert reasons["chore-b"] == "dependency_failed"
    assert reasons["chore-c"] == "dependency_failed"
    assert "chore-b" not in manager._blocked
    assert "chore-c" not in manager._blocked


def test_validate_chores_marks_oversized_chore_and_dependents_failed(
    manager_factory, tmp_path
):
    a = _chore(tmp_path, "chore-a", num_tasks=20, cores_per_task=4)
    b = _chore(tmp_path, "chore-b", deps=("chore-a",))
    manager = manager_factory([a, b])

    manager._validate_chores()
    reasons = {item["chore_id"]: item["reason"] for item in manager._failed_chores}
    assert reasons["chore-a"] == "chore_exceeds_allocation"
    assert reasons["chore-b"] == "dependency_failed"


def test_submit_one_success_updates_running_and_resources(
    manager_factory, tmp_path, monkeypatch
):
    manager = manager_factory(
        [_chore(tmp_path, "chore-a", num_tasks=2, cores_per_task=2, gpus_per_task=1)]
    )
    manager._executor = object()
    manager._free_cores = 8
    manager._free_gpus = 2
    monkeypatch.setattr("matensemble.manager.time.sleep", lambda *_: None)

    manager._submit_one("chore-a", buffer_time=0.0)

    assert manager._running_chores == {"chore-a"}
    assert len(manager._futures) == 1
    assert manager._free_cores == 4
    assert manager._free_gpus == 0


def test_submit_one_failure_records_submit_exception(
    manager_factory, tmp_path, monkeypatch
):
    manager = manager_factory([_chore(tmp_path, "chore-a")])
    manager._executor = object()
    manager._free_cores = 8
    manager._free_gpus = 2
    manager._fluxlet.raise_exc = RuntimeError("submit broke")
    monkeypatch.setattr("matensemble.manager.time.sleep", lambda *_: None)

    manager._submit_one("chore-a", buffer_time=0.0)

    assert manager._failed_chores[0]["chore_id"] == "chore-a"
    assert manager._failed_chores[0]["reason"] == "submit_exception"
    assert "chore-a" not in manager._running_chores


def test_submit_until_ooresources_defers_chores_that_do_not_fit_now(
    manager_factory, tmp_path, monkeypatch
):
    a = _chore(tmp_path, "chore-a", num_tasks=1, cores_per_task=4)
    b = _chore(tmp_path, "chore-b", num_tasks=1, cores_per_task=6)
    manager = manager_factory([a, b])
    manager._ready = deque(["chore-a", "chore-b"])
    manager._executor = object()
    manager._free_cores = 4
    manager._free_gpus = 1
    monkeypatch.setattr("matensemble.manager.time.sleep", lambda *_: None)

    submitted_any = manager._submit_until_ooresources(buffer_time=0.0)

    assert submitted_any is True
    assert manager._running_chores == {"chore-a"}
    assert list(manager._ready) == ["chore-b"]


def test_log_progress_updates_status_writer(manager_factory, tmp_path):
    manager = manager_factory([_chore(tmp_path, "chore-a")])
    manager._ready = deque(["chore-a"])
    manager._blocked = {"chore-b"}
    manager._running_chores = {"chore-c"}
    manager._completed_chores = ["chore-d"]
    manager._failed_chores = [
        {"chore_id": "chore-e", "reason": "x", "upstream": None, "exception": None}
    ]
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
    manager = manager_factory([_chore(tmp_path, "chore-a")])
    manager._check_resources = lambda: (
        setattr(manager, "_free_cores", 8) or setattr(manager, "_free_gpus", 1)
    )
    manager._log_progress = lambda: None
    manager._validate_chores = lambda: None
    monkeypatch.setattr("matensemble.manager.time.perf_counter", lambda: 0.0)
    monkeypatch.setattr("matensemble.manager.time.sleep", lambda *_: None)

    calls = []

    class ExplicitStrategy:
        def process_futures(self, buffer_time):
            calls.append(buffer_time)
            manager._running_chores.clear()
            manager._blocked.clear()
            manager._ready.clear()

    def fake_submit_until(buffer_time):
        if manager._ready:
            manager._running_chores.add(manager._ready.popleft())
        return True

    manager._submit_until_ooresources = fake_submit_until

    manager.run(
        buffer_time=None, adaptive=False, processing_strategy=ExplicitStrategy()
    )

    assert calls == [0.0]
