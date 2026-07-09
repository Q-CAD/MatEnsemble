import time
import threading

from concurrent.futures import Future
from pathlib import Path

from matensemble.chore import ChoreRegistry, ChoreSpec
from matensemble.model import Resources
from matensemble.model import OutputReference
from matensemble.pipeline import Pipeline


def test_spawn_chore_from_name_infers_dependency(tmp_path: Path):
    pipeline = Pipeline(basedir=str(tmp_path))
    upstream_ref = OutputReference("upstream-001", tmp_path / "out" / "upstream-001")

    chore, _out = pipeline._spawn_chore_from_name(
        "process_done", dependent=upstream_ref
    )

    assert chore.deps == ("upstream-001",)
    assert chore.args[0] == upstream_ref
    assert str(chore.spec_path).endswith("chore.pickle")


def test_pipe_chore_preserves_incrementing_ids_and_function_kwargs(tmp_path: Path):
    pipeline = Pipeline(basedir=str(tmp_path))

    @pipeline.chore(name="evaluate")
    def evaluate(candidate, nice=None, resources=None):
        return candidate, nice, resources

    first = evaluate("a", nice="function-nice")
    second = evaluate("b", resources="function-resources")

    assert first.chore_id.endswith("chore-evaluate-0001")
    assert second.chore_id.endswith("chore-evaluate-0002")
    assert pipeline._chore_list[0].kwargs["nice"] == "function-nice"
    assert pipeline._chore_list[1].kwargs["resources"] == "function-resources"


def test_pipeline_call_uses_external_registry(tmp_path: Path):
    registry = ChoreRegistry()

    @registry.chore(name="evaluate", cores_per_task=2)
    def evaluate(candidate):
        return candidate

    pipeline = Pipeline(basedir=str(tmp_path), registry=registry)
    out_ref = pipeline.call("evaluate", "candidate", queue_nice=-5)
    chore = pipeline._chore_list[0]

    assert out_ref.chore_id.endswith("chore-evaluate-0001")
    assert chore.chore_qualname == "evaluate"
    assert chore.args == ("candidate",)
    assert chore.resources.cores_per_task == 2
    assert chore.nice == -5


def test_spawn_chore_from_spec_copies_nice(tmp_path: Path):
    pipeline = Pipeline(basedir=str(tmp_path))
    spec = ChoreSpec(
        args=("candidate",),
        kwargs=None,
        qualname="evaluate",
        resources=Resources(),
        nice=-10,
    )

    chore, _out = pipeline._spawn_chore_from_spec(spec)

    assert chore.nice == -10


def test_pipeline_graph_returns_networkx_dag(tmp_path: Path):
    pipeline = Pipeline(basedir=str(tmp_path))

    @pipeline.chore(name="first")
    def first():
        return 1

    @pipeline.chore(name="second")
    def second(value):
        return value + 1

    first_ref = first()
    second(first_ref)

    dag = pipeline.graph()

    assert list(dag.nodes) == [first_ref.chore_id, "chore-second-0002"]
    assert list(dag.edges) == [(first_ref.chore_id, "chore-second-0002")]


def test_pipeline_graph_can_render_image(tmp_path: Path):
    pipeline = Pipeline(basedir=str(tmp_path))

    @pipeline.chore(name="first")
    def first():
        return 1

    @pipeline.chore(name="second")
    def second(value):
        return value + 1

    second(first())
    output_path = tmp_path / "dag.png"

    dag = pipeline.graph(output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert dag.number_of_edges() == 1


def test_submit_returns_before_background_work_finishes(monkeypatch, tmp_path: Path):
    pipeline = Pipeline(basedir=str(tmp_path))

    def fake_submit(*_args, **_kwargs):
        time.sleep(0.2)
        return {"ok": 1}

    monkeypatch.setattr(pipeline, "_submit", fake_submit)

    fut = pipeline.submit()

    assert isinstance(fut, Future)
    assert not fut.done()
    assert fut.result(timeout=2) == {"ok": 1}


def test_submit_rejects_restart_checkpointing(tmp_path: Path):
    pipeline = Pipeline(basedir=str(tmp_path))

    fut = pipeline.submit(write_restart_freq=1)

    assert isinstance(fut.exception(timeout=2), NotImplementedError)


def test_results_waits_for_finished_flag(tmp_path: Path):
    pipeline = Pipeline(basedir=str(tmp_path))
    pipeline._finished = False
    pipeline._output_reference_list = [
        OutputReference("missing", tmp_path / "does-not-exist"),
    ]

    timer = threading.Timer(0.01, setattr, args=(pipeline, "_finished", True))
    timer.start()
    out = pipeline.results(timeout=1.0)
    timer.join()
    assert "missing" in out
