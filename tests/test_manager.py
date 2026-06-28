from collections import deque
from pathlib import Path

from matensemble.chore import Chore
from matensemble.manager import FluxManager
from matensemble.model import ChoreType, Resources


def _chore(chore_id: str, deps=()):
    return Chore(
        id=chore_id,
        workdir=Path.cwd() / "tmp" / chore_id,
        command=["echo", "ok"],
        chore_type=ChoreType.EXECUTABLE,
        resources=Resources(),
        deps=deps,
    )


def test_record_failure_is_idempotent():
    manager = FluxManager.__new__(FluxManager)
    manager._failed_chores = []
    manager._record_failure("a", "x")
    manager._record_failure("a", "x")
    assert len(manager._failed_chores) == 1
    assert manager._failed_chores[0]["exception"] is None


def test_fail_dependents_marks_children():
    manager = FluxManager.__new__(FluxManager)
    manager._dependents = {"a": ["b"], "b": []}
    manager._completed_chores = []
    manager._running_chores = set()
    manager._ready = deque(["b"])
    manager._blocked = {"b"}
    manager._failed_chores = []
    manager._logger = type("L", (), {"error": staticmethod(lambda *args, **kwargs: None)})()
    manager._has_failed = FluxManager._has_failed.__get__(manager, FluxManager)
    manager._record_failure = FluxManager._record_failure.__get__(manager, FluxManager)
    manager._fail_dependents = FluxManager._fail_dependents.__get__(manager, FluxManager)
    manager._fail_dependents("a")
    assert manager._has_failed("b")
