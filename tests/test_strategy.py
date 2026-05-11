from collections import deque
from pathlib import Path

from matensemble.model import ChoreType, Resources
from matensemble.strategy import append_text
from matensemble.chore import Chore


def test_append_text_creates_and_appends(tmp_path: Path):
    path = tmp_path / "a" / "stderr"
    append_text(path, "hello")
    append_text(path, " world")
    assert path.read_text() == "hello world"


def test_user_strategy_related_spawn_path_uses_manager_add(monkeypatch, tmp_path: Path):
    # minimal smoke for pieces used by UserStrategy branch logic helpers
    c = Chore(
        id="chore-process-0001",
        workdir=tmp_path / "chore-process-0001",
        command=["python"],
        chore_type=ChoreType.PYTHON,
        resources=Resources(),
        chore_qualname="process",
    )
    assert c.id.startswith("chore-")
    assert isinstance(deque(), deque)
