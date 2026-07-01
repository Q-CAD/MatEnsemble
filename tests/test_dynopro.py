import pickle
import sys

from pathlib import Path

import cloudpickle

from matensemble.chore import Chore
from matensemble.dynopro.driver import _run_chore
from matensemble.fluxlet import Fluxlet
from matensemble.model import ChoreType, Resources
from matensemble.pipeline import Pipeline


class _FakeJobspec:
    def __init__(self):
        self.cwd = None
        self.stdout = None
        self.stderr = None
        self.environment = {}
        self.shell_opts = {}

    def setattr_shell_option(self, key, value):
        self.shell_opts[key] = value


def test_pipeline_dynopro_builds_registered_subprocess_chore(tmp_path: Path):
    pipe = Pipeline(basedir=tmp_path)

    @pipe.chore(name="gpu-work")
    def gpu_work():
        return "gpu"

    @pipe.chore(name="cpu-work")
    def cpu_work():
        return "cpu"

    chore = pipe.dynopro(
        "gpu-work",
        "cpu-work",
        nnodes=2,
        gpus_per_node=4,
        cores_per_node=8,
        subprocess_args=("payload",),
        subprocess_kwargs={"scale": 2},
    )

    assert chore.command[:3] == [sys.executable, "-m", "matensemble.dynopro.driver"]
    assert "--gpu-subprocess=gpu-work" in chore.command
    assert "--cpu-subprocess=cpu-work" in chore.command
    assert "--gpus-per-node=4" in chore.command
    assert "--cores-per-node=8" in chore.command
    assert chore.resources.num_tasks == 16
    assert chore.resources.mpi is True
    assert chore.nnodes == 2
    assert chore.args == ("payload",)
    assert chore.kwargs == {"scale": 2}


def test_pipeline_dynopro_rejects_unregistered_subprocess(tmp_path: Path):
    pipe = Pipeline(basedir=tmp_path)

    @pipe.chore(name="gpu-work")
    def gpu_work():
        return "gpu"

    try:
        pipe.dynopro("gpu-work", "missing", nnodes=1, gpus_per_node=1, cores_per_node=2)
    except ValueError as exc:
        assert "not registered" in str(exc)
    else:
        raise AssertionError("expected unregistered subprocess to be rejected")


def test_fluxlet_writes_dynopro_spec_in_per_resource_branch(monkeypatch, tmp_path: Path):
    fake_jobspec = _FakeJobspec()

    class _JobspecV1:
        @staticmethod
        def per_resource(*_args, **_kwargs):
            return fake_jobspec

    class _Future:
        pass

    class _Executor:
        def submit(self, jobspec):
            f = _Future()
            f.jobspec = jobspec
            return f

    class _Handle:
        def rpc(self, *_args, **_kwargs):
            class _Done:
                def get(self):
                    return None

            return _Done()

    monkeypatch.setattr("flux.job.JobspecV1", _JobspecV1, raising=False)
    monkeypatch.setattr(Fluxlet, "get_gpus_per_node", lambda _self: (1, 1))

    chore = Chore(
        id="chore-dynopro-0001",
        workdir=tmp_path / "chore-dynopro-0001",
        command=["python", "-m", "matensemble.dynopro.driver"],
        chore_type=ChoreType.EXECUTABLE,
        resources=Resources(num_tasks=2, mpi=True),
        nnodes=1,
    )

    fluxlet = Fluxlet(_Handle())
    fluxlet.submit(_Executor(), chore)

    with (chore.workdir / "chore.pickle").open("rb") as f:
        loaded = pickle.load(f)

    assert loaded.id == chore.id
    assert fake_jobspec.cwd == str(chore.workdir)
    assert fake_jobspec.shell_opts["mpi"] == "pmi2"


def test_dynopro_driver_runs_registered_callable_with_runtime_kwargs(tmp_path: Path):
    out_dir = tmp_path / "wf" / "out"
    registry = out_dir / "registry"
    chore_dir = out_dir / "chore-dynopro-0001"
    registry.mkdir(parents=True)
    chore_dir.mkdir(parents=True)

    def gpu_func(payload, *, scale, comm, split, rank_color, chore_dir):
        return {
            "payload": payload,
            "scale": scale,
            "rank": comm.Get_rank(),
            "split": split.name,
            "color": rank_color,
            "chore_dir": chore_dir.name,
        }

    with (registry / "gpu").open("wb") as f:
        cloudpickle.dump(gpu_func, f)

    chore = Chore(
        id="chore-dynopro-0001",
        workdir=chore_dir,
        command=["python"],
        chore_type=ChoreType.EXECUTABLE,
        resources=Resources(),
        args=("abc",),
        kwargs={"scale": 3},
        nnodes=1,
    )
    with (chore_dir / "chore.pickle").open("wb") as f:
        pickle.dump(chore, f)

    class _Comm:
        def Get_rank(self):
            return 7

    class _Split:
        name = "gpu-split"

    result = _run_chore("gpu", chore_dir, split=_Split(), comm=_Comm(), color=0)

    assert result == {
        "payload": "abc",
        "scale": 3,
        "rank": 7,
        "split": "gpu-split",
        "color": 0,
        "chore_dir": "chore-dynopro-0001",
    }
