import pickle
import sys

from pathlib import Path

import cloudpickle

from matensemble.chore import Chore
from matensemble.model import ChoreType, Resources
from matensemble.runtime_worker import _load_callable, main


def test_load_callable_reads_registry_entry(tmp_path: Path):
    registry = tmp_path / "registry"
    registry.mkdir()

    def f(x):
        return x + 1

    with (registry / "f").open("wb") as fh:
        cloudpickle.dump(f, fh)

    loaded = _load_callable("f", registry)
    assert loaded(2) == 3


def test_main_executes_chore_and_writes_result(monkeypatch, tmp_path: Path):
    workflow = tmp_path / "wf"
    out_dir = workflow / "out"
    registry = out_dir / "registry"
    chore_dir = out_dir / "chore-work-0001"
    registry.mkdir(parents=True)
    chore_dir.mkdir(parents=True)

    def work(x):
        return x * 2

    with (registry / "work").open("wb") as fh:
        cloudpickle.dump(work, fh)

    chore = Chore(
        id="chore-work-0001",
        workdir=chore_dir,
        command=["python"],
        chore_type=ChoreType.PYTHON,
        resources=Resources(),
        chore_qualname="work",
        args=(21,),
    )
    with (chore_dir / "chore.pickle").open("wb") as fh:
        pickle.dump(chore, fh)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runtime_worker",
            "--chore-id",
            chore.id,
            "--spec-file",
            str(chore_dir / "chore.pickle"),
        ],
    )
    main()

    with (chore_dir / "result.pickle").open("rb") as fh:
        assert pickle.load(fh) == 42
