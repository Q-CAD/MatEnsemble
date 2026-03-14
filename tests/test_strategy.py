from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from matensemble.strategy import AdaptiveStrategy, NonAdaptiveStrategy, append_text


class FakeLogger:
    def __init__(self):
        self.errors = []
        self.exceptions = []

    def error(self, *args, **kwargs):
        self.errors.append((args, kwargs))

    def exception(self, *args, **kwargs):
        self.exceptions.append((args, kwargs))


class FakeManager:
    def __init__(self, tmp_path):
        self._futures = set()
        self._running_jobs = set()
        self._completed_jobs = []
        self._dependents = {"job-a": ["job-b"]}
        self._remaining_deps = {"job-b": 1}
        self._ready = deque()
        self._blocked = {"job-b"}
        self._write_restart_freq = None
        self._logger = FakeLogger()
        self.recorded_failures = []
        self.failed_dependents = []
        self.submit_calls = []
        self.restart_calls = 0
        self.tmp_path = tmp_path

    def _record_failure(self, *args, **kwargs):
        self.recorded_failures.append((args, kwargs))

    def _fail_dependents(self, job_id):
        self.failed_dependents.append(job_id)

    def _submit_until_ooresources(self, buffer_time):
        self.submit_calls.append(buffer_time)

    def _make_restart(self):
        self.restart_calls += 1


class StubFuture:
    def __init__(self, job_id, workdir, result_value=None, exc=None):
        self.job_id = job_id
        self.job_obj = SimpleNamespace(workdir=workdir)
        self._result_value = result_value
        self._exc = exc

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._result_value

    def __hash__(self):
        return id(self)


def test_append_text_creates_parent_dirs_and_appends(tmp_path):
    path = tmp_path / "logs" / "stderr"
    append_text(path, "first")
    append_text(path, "second")
    assert path.read_text() == "firstsecond"


def test_adaptive_strategy_success_releases_dependents_and_submits(monkeypatch, tmp_path):
    manager = FakeManager(tmp_path)
    fut = StubFuture("job-a", tmp_path / "job-a", result_value=0)
    manager._futures = {fut}
    manager._running_jobs = {"job-a"}

    monkeypatch.setattr(
        "matensemble.strategy.concurrent.futures.wait",
        lambda futures, timeout: ({fut}, set()),
    )

    AdaptiveStrategy(manager).process_futures(buffer_time=0.5)

    assert manager._completed_jobs == ["job-a"]
    assert list(manager._ready) == ["job-b"]
    assert "job-b" not in manager._blocked
    assert manager.submit_calls == [0.5]


def test_nonadaptive_strategy_success_does_not_submit_more(monkeypatch, tmp_path):
    manager = FakeManager(tmp_path)
    fut = StubFuture("job-a", tmp_path / "job-a", result_value=0)
    manager._futures = {fut}
    manager._running_jobs = {"job-a"}

    monkeypatch.setattr(
        "matensemble.strategy.concurrent.futures.wait",
        lambda futures, timeout: ({fut}, set()),
    )

    NonAdaptiveStrategy(manager).process_futures(buffer_time=0.5)

    assert manager._completed_jobs == ["job-a"]
    assert manager.submit_calls == []


def test_strategy_exception_records_failure_and_writes_stderr(monkeypatch, tmp_path):
    manager = FakeManager(tmp_path)
    workdir = tmp_path / "job-a"
    fut = StubFuture("job-a", workdir, exc=RuntimeError("boom"))
    manager._futures = {fut}
    manager._running_jobs = {"job-a"}

    monkeypatch.setattr(
        "matensemble.strategy.concurrent.futures.wait",
        lambda futures, timeout: ({fut}, set()),
    )

    AdaptiveStrategy(manager).process_futures(buffer_time=0.0)

    stderr_text = (workdir / "stderr").read_text()
    assert "MATENSEMBLE WRAPPER ERROR" in stderr_text
    assert manager.recorded_failures[0][0] == ("job-a",)
    assert manager.recorded_failures[0][1]["reason"] == "exception"
    assert manager.failed_dependents == ["job-a"]


def test_strategy_nonzero_exit_records_failure(monkeypatch, tmp_path):
    manager = FakeManager(tmp_path)
    workdir = tmp_path / "job-a"
    fut = StubFuture("job-a", workdir, result_value=3)
    manager._futures = {fut}
    manager._running_jobs = {"job-a"}

    monkeypatch.setattr(
        "matensemble.strategy.concurrent.futures.wait",
        lambda futures, timeout: ({fut}, set()),
    )

    NonAdaptiveStrategy(manager).process_futures(buffer_time=0.0)

    stderr_text = (workdir / "stderr").read_text()
    assert "NONZERO EXIT" in stderr_text
    assert manager.recorded_failures[0][1]["reason"] == "nonzero_exit:3"
    assert manager.failed_dependents == ["job-a"]
