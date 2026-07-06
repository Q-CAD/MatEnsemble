import pickle
import sys

from pathlib import Path

import cloudpickle
import pytest

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
    assert chore.dynopro_args == {
        "gpu-work": ("payload",),
        "cpu-work": ("payload",),
    }
    assert chore.dynopro_kwargs == {
        "gpu-work": {"scale": 2},
        "cpu-work": {"scale": 2},
    }


def test_pipeline_dynopro_accepts_per_subprocess_args_and_collects_deps(
    tmp_path: Path,
):
    pipe = Pipeline(basedir=tmp_path)

    @pipe.chore(name="upstream-gpu")
    def upstream_gpu():
        return "gpu-ref"

    @pipe.chore(name="upstream-cpu")
    def upstream_cpu():
        return "cpu-ref"

    @pipe.chore(name="gpu-work")
    def gpu_work():
        return "gpu"

    @pipe.chore(name="cpu-work")
    def cpu_work():
        return "cpu"

    gpu_ref = upstream_gpu()
    cpu_ref = upstream_cpu()

    chore = pipe.dynopro(
        "gpu-work",
        "cpu-work",
        nnodes=1,
        gpus_per_node=2,
        cores_per_node=4,
        gpu_args=("gpu-payload", gpu_ref),
        gpu_kwargs={"scale": 2},
        cpu_args=("cpu-payload",),
        cpu_kwargs={"source": cpu_ref},
    )

    assert chore.dynopro_args == {
        "gpu-work": ("gpu-payload", gpu_ref),
        "cpu-work": ("cpu-payload",),
    }
    assert chore.dynopro_kwargs == {
        "gpu-work": {"scale": 2},
        "cpu-work": {"source": cpu_ref},
    }
    assert chore.deps == (gpu_ref.chore_id, cpu_ref.chore_id)


def test_pipeline_dynopro_rejects_mixed_shared_and_per_subprocess_args(
    tmp_path: Path,
):
    pipe = Pipeline(basedir=tmp_path)

    @pipe.chore(name="gpu-work")
    def gpu_work():
        return "gpu"

    @pipe.chore(name="cpu-work")
    def cpu_work():
        return "cpu"

    with pytest.raises(ValueError, match="cannot be mixed"):
        pipe.dynopro(
            "gpu-work",
            "cpu-work",
            nnodes=1,
            gpus_per_node=1,
            cores_per_node=2,
            subprocess_args=("shared",),
            gpu_args=("gpu-only",),
        )


def test_pipeline_dynopro_rejects_per_subprocess_args_for_same_registered_name(
    tmp_path: Path,
):
    pipe = Pipeline(basedir=tmp_path)

    @pipe.chore(name="work")
    def work():
        return "work"

    with pytest.raises(ValueError, match="require distinct"):
        pipe.dynopro(
            "work",
            "work",
            nnodes=1,
            gpus_per_node=1,
            cores_per_node=2,
            gpu_args=("gpu-only",),
        )


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


def test_dynopro_driver_runs_registered_callables_with_per_subprocess_args(
    tmp_path: Path,
):
    out_dir = tmp_path / "wf" / "out"
    registry = out_dir / "registry"
    chore_dir = out_dir / "chore-dynopro-0001"
    registry.mkdir(parents=True)
    chore_dir.mkdir(parents=True)

    def gpu_func(payload, *, scale, comm, split, rank_color, chore_dir):
        return {
            "kind": "gpu",
            "payload": payload,
            "scale": scale,
            "rank": comm.Get_rank(),
            "split": split.name,
            "color": rank_color,
            "chore_dir": chore_dir.name,
        }

    def cpu_func(payload, *, mode, comm, split, rank_color, chore_dir):
        return {
            "kind": "cpu",
            "payload": payload,
            "mode": mode,
            "rank": comm.Get_rank(),
            "split": split.name,
            "color": rank_color,
            "chore_dir": chore_dir.name,
        }

    with (registry / "gpu").open("wb") as f:
        cloudpickle.dump(gpu_func, f)
    with (registry / "cpu").open("wb") as f:
        cloudpickle.dump(cpu_func, f)

    chore = Chore(
        id="chore-dynopro-0001",
        workdir=chore_dir,
        command=["python"],
        chore_type=ChoreType.EXECUTABLE,
        resources=Resources(),
        dynopro_args={
            "gpu": ("gpu-abc",),
            "cpu": ("cpu-xyz",),
        },
        dynopro_kwargs={
            "gpu": {"scale": 3},
            "cpu": {"mode": "analysis"},
        },
        nnodes=1,
    )
    with (chore_dir / "chore.pickle").open("wb") as f:
        pickle.dump(chore, f)

    class _Comm:
        def Get_rank(self):
            return 7

    class _Split:
        name = "gpu-split"

    gpu_result = _run_chore("gpu", chore_dir, split=_Split(), comm=_Comm(), color=0)
    cpu_result = _run_chore("cpu", chore_dir, split=_Split(), comm=_Comm(), color=1)

    assert gpu_result == {
        "kind": "gpu",
        "payload": "gpu-abc",
        "scale": 3,
        "rank": 7,
        "split": "gpu-split",
        "color": 0,
        "chore_dir": "chore-dynopro-0001",
    }
    assert cpu_result == {
        "kind": "cpu",
        "payload": "cpu-xyz",
        "mode": "analysis",
        "rank": 7,
        "split": "gpu-split",
        "color": 1,
        "chore_dir": "chore-dynopro-0001",
    }
