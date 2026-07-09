from collections import deque
from pathlib import Path

from matensemble.chore import Chore
from matensemble.manager import FluxManager
from matensemble.model import ChoreType, Resources


def _chore(chore_id: str, deps=(), nice=0):
    return Chore(
        id=chore_id,
        workdir=Path.cwd() / "tmp" / chore_id,
        command=["echo", "ok"],
        chore_type=ChoreType.EXECUTABLE,
        resources=Resources(),
        deps=deps,
        nice=nice,
    )


def _bare_manager():
    """Partial FluxManager (no __init__) with fields required by _add_chore."""
    manager = FluxManager.__new__(FluxManager)
    manager._chores_by_id = {}
    manager._dependents = {}
    manager._remaining_deps = {}
    manager._ready = deque()
    manager._blocked = set()
    manager._completed_chores = []
    manager._failed_chores = []
    manager._nnodes_on_allocation = 1
    manager._cores_per_node = 1
    manager._gpus_per_node = 0
    manager._ready_order = {}
    manager._ready_order_counter = 0
    return manager


def test_add_chore_adds_ready_chore_without_deps():
    manager = _bare_manager()

    chore = _chore("chore-a")
    manager._add_chore(chore)

    assert manager._chores_by_id["chore-a"] is chore
    assert manager._remaining_deps["chore-a"] == 0
    assert list(manager._ready) == ["chore-a"]


def test_add_chore_with_completed_dependency_becomes_ready():
    manager = _bare_manager()
    manager._chores_by_id = {"dep-1": _chore("dep-1")}
    manager._dependents = {"dep-1": []}
    manager._remaining_deps = {"dep-1": 0}
    manager._completed_chores = ["dep-1"]

    chore = _chore("chore-b", deps=("dep-1",))
    manager._add_chore(chore)

    assert manager._remaining_deps["chore-b"] == 0
    assert list(manager._ready) == ["chore-b"]
    assert "chore-b" in manager._dependents["dep-1"]


def test_add_chore_sorts_ready_queue_by_nice():
    manager = _bare_manager()

    manager._add_chore(_chore("normal", nice=0))
    manager._add_chore(_chore("urgent", nice=-10))
    manager._add_chore(_chore("background", nice=10))

    assert list(manager._ready) == ["urgent", "normal", "background"]


def test_equal_nice_preserves_ready_admission_order():
    manager = _bare_manager()

    manager._add_chore(_chore("first", nice=0))
    manager._add_chore(_chore("second", nice=0))
    manager._add_chore(_chore("third", nice=0))

    assert list(manager._ready) == ["first", "second", "third"]


def test_dependency_unblocked_chore_sorts_into_ready_queue():
    manager = _bare_manager()
    manager._chores_by_id = {
        "ready": _chore("ready", nice=5),
        "dep": _chore("dep", nice=0),
        "child": _chore("child", deps=("dep",), nice=-5),
    }
    manager._dependents = {"ready": [], "dep": ["child"], "child": []}
    manager._remaining_deps = {"ready": 0, "dep": 0, "child": 1}
    manager._ready = deque()
    manager._mark_ready("ready")

    manager._remaining_deps["child"] -= 1
    if manager._remaining_deps["child"] == 0:
        manager._mark_ready("child")
        manager._blocked.discard("child")

    assert list(manager._ready) == ["child", "ready"]
