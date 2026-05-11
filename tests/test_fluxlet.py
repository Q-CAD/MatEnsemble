from pathlib import Path

from matensemble.chore import Chore
from matensemble.fluxlet import Fluxlet
from matensemble.model import ChoreType, Resources


class _FakeJobspec:
    def __init__(self):
        self.cwd = None
        self.stdout = None
        self.stderr = None
        self.environment = {}
        self.num_nodes = None
        self.shell_opts = {}

    def setattr_shell_option(self, key, value):
        self.shell_opts[key] = value


def test_fluxlet_submit_sets_workdir_and_streams(monkeypatch, tmp_path: Path):
    fake_jobspec = _FakeJobspec()

    class _JobspecV1:
        @staticmethod
        def from_command(*_args, **_kwargs):
            return fake_jobspec

    class _Future:
        pass

    class _Executor:
        def submit(self, jobspec):
            f = _Future()
            f.jobspec = jobspec
            return f

    monkeypatch.setattr("flux.job.JobspecV1", _JobspecV1, raising=False)
    monkeypatch.setattr(Fluxlet, "get_gpus_per_node", lambda _self: (1, 0))

    class _Handle:
        def rpc(self, *_args, **_kwargs):
            class _Done:
                def get(self):
                    return None

            return _Done()

    chore = Chore(
        id="chore-fx-1",
        workdir=tmp_path / "chore-fx-1",
        command=["echo", "ok"],
        chore_type=ChoreType.EXECUTABLE,
        resources=Resources(),
    )
    fluxlet = Fluxlet(_Handle())
    fut = fluxlet.submit(_Executor(), chore)
    assert fut.chore_id == "chore-fx-1"
    assert fake_jobspec.cwd == str(chore.workdir)
