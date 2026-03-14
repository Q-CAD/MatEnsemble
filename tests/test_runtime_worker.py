from __future__ import annotations

import json
import pickle
import sys
import types
from pathlib import Path

import pytest

from matensemble.job import Job
from matensemble.model import JobFlavor, Resources
from matensemble.runtime_worker import (
    _load_dep_result,
    _resolve_qualname,
    _try_write_result_json,
    main,
)


def test_resolve_qualname_unwraps_function():
    module = types.ModuleType("fake_tasks")

    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.__wrapped__ = func
        return wrapper

    def add(x, y):
        return x + y

    module.add = decorator(add)
    resolved = _resolve_qualname(module, "add")
    assert resolved is add


def test_resolve_qualname_rejects_locals():
    module = types.ModuleType("fake_tasks")
    module.outer = types.SimpleNamespace()
    with pytest.raises(ValueError, match="top-level callables"):
        _resolve_qualname(module, "outer.<locals>.inner")


def test_load_dep_result_reads_pickled_result(tmp_path):
    spec_file = tmp_path / "workflow" / "out" / "job-2" / "job.pkl"
    dep_dir = spec_file.parent.parent / "job-1"
    dep_dir.mkdir(parents=True)
    with (dep_dir / "result.pkl").open("wb") as f:
        pickle.dump({"ok": True}, f)

    value = _load_dep_result(spec_file, "job-1")
    assert value == {"ok": True}


def test_try_write_result_json_writes_json_safe_content(tmp_path):
    out_file = tmp_path / "result.json"
    _try_write_result_json({"path": Path("x")}, out_file)
    assert json.loads(out_file.read_text()) == {"path": "x"}


def test_main_executes_job_resolves_dependencies_and_writes_outputs(tmp_path, monkeypatch):
    source_root = tmp_path / "srcroot"
    workflow_dir = source_root / "matensemble_workflow-20260101_010101"
    dep_dir = workflow_dir / "out" / "job-make-0001"
    job_dir = workflow_dir / "out" / "job-add-0002"
    dep_dir.mkdir(parents=True)
    job_dir.mkdir(parents=True)

    with (dep_dir / "result.pkl").open("wb") as f:
        pickle.dump(10, f)

    module = types.ModuleType("user_tasks")

    def add_one(x, y=0):
        return x + 1 + y

    module.add_one = add_one
    sys.modules["user_tasks"] = module

    job = Job(
        id="job-add-0002",
        command=[sys.executable, "-m", "matensemble.runtime_worker"],
        flavor=JobFlavor.PYTHON,
        resources=Resources(),
        workdir=job_dir,
        func_module="user_tasks",
        func_qualname="add_one",
        deps=("job-make-0001",),
        args=(__import__("matensemble").OutputReference("job-make-0001"),),
        kwargs={"y": 5},
    )

    spec_file = job_dir / "job.pkl"
    with spec_file.open("wb") as f:
        pickle.dump(job, f)

    monkeypatch.setattr(sys, "argv", [
        "runtime_worker.py",
        "--job-id",
        "job-add-0002",
        "--spec-file",
        str(spec_file),
    ])

    main()

    with (job_dir / "result.pkl").open("rb") as f:
        assert pickle.load(f) == 16
    assert json.loads((job_dir / "result.json").read_text()) == 16


def test_main_rejects_job_id_mismatch(tmp_path, monkeypatch):
    workflow_dir = tmp_path / "srcroot" / "matensemble_workflow-20260101_010101" / "out" / "job-a"
    workflow_dir.mkdir(parents=True)
    job = Job(
        id="job-a",
        command=["python"],
        flavor=JobFlavor.PYTHON,
        resources=Resources(),
        workdir=workflow_dir,
        func_module="mod",
        func_qualname="func",
    )
    spec_file = workflow_dir / "job.pkl"
    with spec_file.open("wb") as f:
        pickle.dump(job, f)

    monkeypatch.setattr(sys, "argv", [
        "runtime_worker.py",
        "--job-id",
        "job-b",
        "--spec-file",
        str(spec_file),
    ])

    with pytest.raises(ValueError, match="Job ID mismatch"):
        main()
