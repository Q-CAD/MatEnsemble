from __future__ import annotations

import json
from pathlib import Path

import pytest

from matensemble.job import Job
from matensemble.model import JobFlavor, Resources


def test_job_splits_string_command_and_builds_spec_path(tmp_path):
    job = Job(
        id="job-1",
        command="python -m thing --flag",
        flavor=JobFlavor.EXECUTABLE,
        resources=Resources(),
        workdir=tmp_path / "out" / "job-1",
    )

    assert job.command == ["python", "-m", "thing", "--flag"]
    assert job.spec_path == (tmp_path / "out" / "job-1" / "job.pkl").resolve()
    assert job.kwargs == {}


@pytest.mark.parametrize(
    ("func_module", "func_qualname"),
    [(None, "func"), ("mod", None)],
)
def test_python_jobs_require_importable_function_metadata(tmp_path, func_module, func_qualname):
    with pytest.raises(ValueError):
        Job(
            id="job-1",
            command=["python"],
            flavor=JobFlavor.PYTHON,
            resources=Resources(),
            workdir=tmp_path / "w",
            func_module=func_module,
            func_qualname=func_qualname,
        )


def test_to_debug_dict_and_write_debug_json(tmp_path):
    workdir = tmp_path / "out" / "job-1"
    job = Job(
        id="job-1",
        command=["python", "-m", "pkg.worker"],
        flavor=JobFlavor.PYTHON,
        resources=Resources(env={"A": "B"}),
        workdir=workdir,
        func_module="user_tasks",
        func_qualname="add",
        deps=("job-0",),
        args=(1,),
        kwargs={"x": 2},
    )

    data = job._to_debug_dict()
    assert data["id"] == "job-1"
    assert data["func_module"] == "user_tasks"
    assert data["deps"] == ["job-0"]

    job._write_debug_json()
    written = json.loads((workdir / "job.json").read_text())
    assert written["resources"]["env"] == {"A": "B"}
    assert written["kwargs"] == {"x": 2}


def test_job_string_contains_json(tmp_path):
    job = Job(
        id="job-1",
        command=["echo", "hi"],
        flavor=JobFlavor.EXECUTABLE,
        resources=Resources(),
        workdir=tmp_path / "w",
    )
    rendered = str(job)
    assert '"id": "job-1"' in rendered
    assert '"command": [' in rendered
