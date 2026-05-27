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
