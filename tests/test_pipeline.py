from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from matensemble.job import Job
from matensemble.model import JobFlavor, OutputReference, Resources
from matensemble.pipeline import Pipeline


class DummyManager:
    created = None
    run_args = None

    def __init__(self, job_list, base_dir, write_restart_freq, set_cpu_affinity, set_gpu_affinity):
        DummyManager.created = {
            "job_list": job_list,
            "base_dir": base_dir,
            "write_restart_freq": write_restart_freq,
            "set_cpu_affinity": set_cpu_affinity,
            "set_gpu_affinity": set_gpu_affinity,
        }

    def run(self, **kwargs):
        DummyManager.run_args = kwargs


@staticmethod
def _top_level_add(x, y=0):
    return x + y


def top_level_add(x, y=0):
    return x + y


def test_job_decorator_builds_python_job_and_output_reference(tmp_path):
    pipe = Pipeline(basedir=str(tmp_path))
    wrapped = pipe.job(name="add", num_tasks=2, cores_per_task=3, gpus_per_task=1, mpi=True, env={"EXTRA": "1"})(top_level_add)

    ref = wrapped(4, y=5)
    assert ref == OutputReference("job-add-0001")
    assert len(pipe._job_list) == 1

    job = pipe._job_list[0]
    assert job.id == "job-add-0001"
    assert job.flavor == JobFlavor.PYTHON
    assert job.func_module == __name__
    assert job.func_qualname == "top_level_add"
    assert job.args == (4,)
    assert job.kwargs == {"y": 5}
    assert job.resources.num_tasks == 2
    assert job.resources.cores_per_task == 3
    assert job.resources.gpus_per_task == 1
    assert job.resources.mpi is True
    assert job.resources.env["EXTRA"] == "1"
    assert "PYTHONPATH" in job.resources.env
    assert job.command[1:3] == ["-m", "matensemble.runtime_worker"]


def test_job_decorator_collects_dependencies(tmp_path):
    pipe = Pipeline(basedir=str(tmp_path))
    producer = pipe.job()(top_level_add)
    consumer = pipe.job(name="consumer")(top_level_add)

    a = producer(1, y=2)
    b = consumer(a, y=3)

    assert a.job_id == "job-top_level_add-0001"
    assert b.job_id == "job-consumer-0002"
    assert pipe._job_list[1].deps == ("job-top_level_add-0001",)


def test_nested_local_function_is_rejected(tmp_path):
    pipe = Pipeline(basedir=str(tmp_path))

    def outer():
        def inner(x):
            return x
        return inner

    wrapped = pipe.job()(outer())
    with pytest.raises(ValueError, match="top-level callables"):
        wrapped(1)


def test_exec_builds_executable_job(tmp_path):
    pipe = Pipeline(basedir=str(tmp_path))
    job = pipe.exec("echo hello", name="hello", num_tasks=2, cores_per_task=4)

    assert job.id == "job-hello-0001"
    assert job.flavor == JobFlavor.EXECUTABLE
    assert job.command == ["echo", "hello"]
    assert job.resources.num_tasks == 2
    assert job.resources.cores_per_task == 4


def test_create_graph_and_sort_graph(tmp_path):
    pipe = Pipeline(basedir=str(tmp_path))
    a_job = Job("job-a", ["echo", "a"], JobFlavor.EXECUTABLE, Resources(), tmp_path / "a")
    b_job = Job("job-b", ["echo", "b"], JobFlavor.EXECUTABLE, Resources(), tmp_path / "b", deps=("job-a",))
    pipe._job_list = [a_job, b_job]

    graph = pipe._create_graph()
    assert list(graph.edges()) == [("job-a", "job-b")]
    assert pipe._sort_graph(graph) == ["job-a", "job-b"]


def test_create_graph_rejects_unknown_dependencies(tmp_path):
    pipe = Pipeline(basedir=str(tmp_path))
    bad_job = Job("job-b", ["echo", "b"], JobFlavor.EXECUTABLE, Resources(), tmp_path / "b", deps=("missing",))
    pipe._job_list = [bad_job]

    with pytest.raises(ValueError, match="unknown dependencies"):
        pipe._create_graph()


def test_sort_graph_rejects_cycles(tmp_path):
    pipe = Pipeline(basedir=str(tmp_path))
    graph = nx.DiGraph()
    graph.add_edge("a", "b")
    graph.add_edge("b", "a")

    with pytest.raises(Exception, match="cannot contain cycles"):
        pipe._sort_graph(graph)


def test_submit_builds_sorted_job_list_and_calls_manager(monkeypatch, tmp_path):
    import matensemble.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "FluxManager", DummyManager)

    pipe = Pipeline(basedir=str(tmp_path))
    a = pipe.job(name="a")(top_level_add)
    b = pipe.job(name="b")(top_level_add)
    ref = a(1, y=2)
    b(ref, y=3)

    pipe.submit(
        write_restart_freq=7,
        buffer_time=0.25,
        set_cpu_affinity=False,
        set_gpu_affinity=True,
        adaptive=False,
        dynopro=True,
        processing_strategy="sentinel",
    )

    ordered_ids = [job.id for job in DummyManager.created["job_list"]]
    assert ordered_ids == ["job-a-0001", "job-b-0002"]
    assert DummyManager.created["write_restart_freq"] == 7
    assert DummyManager.created["set_cpu_affinity"] is False
    assert DummyManager.created["set_gpu_affinity"] is True
    assert DummyManager.run_args == {
        "buffer_time": 0.25,
        "adaptive": False,
        "dynopro": True,
        "processing_strategy": "sentinel",
    }
