from __future__ import annotations

import json
from pathlib import Path

import pytest

from matensemble.chore import Chore
from matensemble.model import ChoreType, Resources


def test_chore_splits_string_command_and_builds_spec_path(tmp_path):
    chore = Chore(
        id="chore-1",
        command="python -m thing --flag",
        chore_type=ChoreType.EXECUTABLE,
        resources=Resources(),
        workdir=tmp_path / "out" / "chore-1",
    )

    assert chore.command == ["python", "-m", "thing", "--flag"]
    assert chore.spec_path == (tmp_path / "out" / "chore-1" / "chore.pkl").resolve()
    assert chore.kwargs == {}


@pytest.mark.parametrize(
    ("func_module", "func_qualname"),
    [(None, "func"), ("mod", None)],
)
def test_python_chores_require_importable_function_metadata(
    tmp_path, func_module, func_qualname
):
    with pytest.raises(ValueError):
        Chore(
            id="chore-1",
            command=["python"],
            chore_type=ChoreType.PYTHON,
            resources=Resources(),
            workdir=tmp_path / "w",
            func_module=func_module,
            func_qualname=func_qualname,
        )


def test_to_debug_dict_and_write_debug_json(tmp_path):
    workdir = tmp_path / "out" / "chore-1"
    chore = Chore(
        id="chore-1",
        command=["python", "-m", "pkg.worker"],
        chore_type=ChoreType.PYTHON,
        resources=Resources(env={"A": "B"}),
        workdir=workdir,
        func_module="user_tasks",
        func_qualname="add",
        deps=("chore-0",),
        args=(1,),
        kwargs={"x": 2},
    )

    data = chore._to_debug_dict()
    assert data["id"] == "chore-1"
    assert data["func_module"] == "user_tasks"
    assert data["deps"] == ["chore-0"]

    chore._write_debug_json()
    written = json.loads((workdir / "chore.json").read_text())
    assert written["resources"]["env"] == {"A": "B"}
    assert written["kwargs"] == {"x": 2}


def test_chore_string_contains_json(tmp_path):
    chore = Chore(
        id="chore-1",
        command=["echo", "hi"],
        chore_type=ChoreType.EXECUTABLE,
        resources=Resources(),
        workdir=tmp_path / "w",
    )
    rendered = str(chore)
    assert '"id": "chore-1"' in rendered
    assert '"command": [' in rendered
