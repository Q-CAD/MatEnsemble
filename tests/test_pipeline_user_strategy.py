import time

from concurrent.futures import Future
from pathlib import Path

from matensemble.model import OutputReference
from matensemble.pipeline import Pipeline


def test_spawn_chore_from_name_infers_dependency(tmp_path: Path):
    pipeline = Pipeline(basedir=str(tmp_path))
    upstream_ref = OutputReference("upstream-001", tmp_path / "out" / "upstream-001")

    chore = pipeline._spawn_chore_from_name("process_done", dependent=upstream_ref)

    assert chore.deps == ("upstream-001",)
    assert chore.args[0] == upstream_ref
    assert str(chore.spec_path).endswith("chore.pickle")


def test_submit_returns_before_background_work_finishes(monkeypatch, tmp_path: Path):
    pipeline = Pipeline(basedir=str(tmp_path))

    def fake_submit(*_args, **_kwargs):
        time.sleep(0.2)
        return {"ok": 1}

    monkeypatch.setattr(pipeline, "_submit", fake_submit)

    started = time.perf_counter()
    fut = pipeline.submit()
    elapsed = time.perf_counter() - started

    assert isinstance(fut, Future)
    assert elapsed < 0.1
    assert fut.result(timeout=2) == {"ok": 1}


def test_results_waits_for_finished_flag(tmp_path: Path):
    pipeline = Pipeline(basedir=str(tmp_path))
    pipeline._finished = True
    pipeline._output_reference_list = [
        OutputReference("missing", tmp_path / "does-not-exist"),
    ]
    out = pipeline.results(timeout=0.01)
    assert "missing" in out

