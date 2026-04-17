from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from matensemble.chore import Chore
from matensemble.model import ChoreType, OutputReference, Resources
from matensemble.pipeline import Pipeline


class DummyManager:
    created = None
    run_args = None

    def __init__(
        self,
        chore_list,
        base_dir,
        write_restart_freq,
        set_cpu_affinity,
        set_gpu_affinity,
    ):
        DummyManager.created = {
            "chore_list": chore_list,
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


def test_chore_decorator_builds_python_chore_and_output_reference(tmp_path):
    pipe = Pipeline(basedir=str(tmp_path))
    wrapped = pipe.chore(
        name="add",
        num_tasks=2,
        cores_per_task=3,
        gpus_per_task=1,
        mpi=True,
        env={"EXTRA": "1"},
    )(top_level_add)

    ref = wrapped(4, y=5)
    assert ref == OutputReference("chore-add-0001")
    assert len(pipe._chore_list) == 1

    chore = pipe._chore_list[0]
    assert chore.id == "chore-add-0001"
    assert chore.chore_type == ChoreType.PYTHON
    assert chore.func_module == __name__
    assert chore.func_qualname == "top_level_add"
    assert chore.args == (4,)
    assert chore.kwargs == {"y": 5}
    assert chore.resources.num_tasks == 2
    assert chore.resources.cores_per_task == 3
    assert chore.resources.gpus_per_task == 1
    assert chore.resources.mpi is True
    assert chore.resources.env["EXTRA"] == "1"
    assert "PYTHONPATH" in chore.resources.env
    assert chore.command[1:3] == ["-m", "matensemble.runtime_worker"]


def test_chore_decorator_collects_dependencies(tmp_path):
    pipe = Pipeline(basedir=str(tmp_path))
    producer = pipe.chore()(top_level_add)
    consumer = pipe.chore(name="consumer")(top_level_add)

    a = producer(1, y=2)
    b = consumer(a, y=3)

    assert a.chore_id == "chore-top_level_add-0001"
    assert b.chore_id == "chore-consumer-0002"
    assert pipe._chore_list[1].deps == ("chore-top_level_add-0001",)


def test_nested_local_function_is_serialized(tmp_path):
    pipe = Pipeline(basedir=str(tmp_path))

    def outer():
        def inner(x):
            return x

        return inner

    wrapped = pipe.chore()(outer())
    ref = wrapped(1)

    assert ref.chore_id == "chore-inner-0001"
    chore = pipe._chore_list[0]
    assert chore.func_module is None
    assert chore.func_qualname is None
    assert chore.serialized_callable is not None


def test_exec_builds_executable_chore(tmp_path):
    pipe = Pipeline(basedir=str(tmp_path))
    chore = pipe.exec("echo hello", name="hello", num_tasks=2, cores_per_task=4)

    assert chore.id == "chore-hello-0001"
    assert chore.chore_type == ChoreType.EXECUTABLE
    assert chore.command == ["echo", "hello"]
    assert chore.resources.num_tasks == 2
    assert chore.resources.cores_per_task == 4


def test_create_graph_and_sort_graph(tmp_path):
    pipe = Pipeline(basedir=str(tmp_path))
    a_chore = Chore(
        "chore-a", ["echo", "a"], ChoreType.EXECUTABLE, Resources(), tmp_path / "a"
    )
    b_chore = Chore(
        "chore-b",
        ["echo", "b"],
        ChoreType.EXECUTABLE,
        Resources(),
        tmp_path / "b",
        deps=("chore-a",),
    )
    pipe._chore_list = [a_chore, b_chore]

    graph = pipe._create_graph()
    assert list(graph.edges()) == [("chore-a", "chore-b")]
    assert pipe._sort_graph(graph) == ["chore-a", "chore-b"]


def test_create_graph_rejects_unknown_dependencies(tmp_path):
    pipe = Pipeline(basedir=str(tmp_path))
    bad_chore = Chore(
        "chore-b",
        ["echo", "b"],
        ChoreType.EXECUTABLE,
        Resources(),
        tmp_path / "b",
        deps=("missing",),
    )
    pipe._chore_list = [bad_chore]

    with pytest.raises(ValueError, match="unknown dependencies"):
        pipe._create_graph()


def test_sort_graph_rejects_cycles(tmp_path):
    pipe = Pipeline(basedir=str(tmp_path))
    graph = nx.DiGraph()
    graph.add_edge("a", "b")
    graph.add_edge("b", "a")

    with pytest.raises(Exception, match="cannot contain cycles"):
        pipe._sort_graph(graph)


def test_submit_builds_sorted_chore_list_and_calls_manager(monkeypatch, tmp_path):
    import matensemble.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "FluxManager", DummyManager)

    pipe = Pipeline(basedir=str(tmp_path))
    a = pipe.chore(name="a")(top_level_add)
    b = pipe.chore(name="b")(top_level_add)
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

    ordered_ids = [chore.id for chore in DummyManager.created["chore_list"]]
    assert ordered_ids == ["chore-a-0001", "chore-b-0002"]
    assert DummyManager.created["write_restart_freq"] == 7
    assert DummyManager.created["set_cpu_affinity"] is False
    assert DummyManager.created["set_gpu_affinity"] is True
    assert DummyManager.run_args == {
        "buffer_time": 0.25,
        "adaptive": False,
        "dynopro": True,
        "processing_strategy": "sentinel",
        "dashboard": False,
    }
