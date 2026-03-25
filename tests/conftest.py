from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class _FakeRPC:
    def get(self):
        return None


class FakeFlux:
    def rpc(self, *args, **kwargs):
        return _FakeRPC()


class FakeJobspec:
    def __init__(self, command, num_tasks, cores_per_task, gpus_per_task):
        self.command = list(command)
        self.num_tasks = num_tasks
        self.cores_per_task = cores_per_task
        self.gpus_per_task = gpus_per_task
        self.cwd = None
        self.stdout = None
        self.stderr = None
        self.env = None
        self.num_nodes = None
        self.shell_options: dict[str, str] = {}

    def setattr_shell_option(self, key, value):
        self.shell_options[key] = value


class FakeJobspecV1:
    @staticmethod
    def from_command(command, num_tasks, cores_per_task, gpus_per_task):
        return FakeJobspec(command, num_tasks, cores_per_task, gpus_per_task)


class FakeFuture:
    def __init__(self, jobspec=None, result_value=0, exc=None):
        self.jobspec = jobspec
        self._result_value = result_value
        self._exc = exc
        self.chore_id = None
        self.chore_obj = None
        self.chore_spec = None
        self.workdir = None

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._result_value

    def __hash__(self):
        return id(self)


class FakeFluxExecutor:
    def __init__(self):
        self.submissions = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, jobspec):
        fut = FakeFuture(jobspec=jobspec, result_value=0)
        self.submissions.append((jobspec, fut))
        return fut


class _FreeResources:
    def __init__(self, ncores=8, ngpus=2, ranks=None):
        self.ncores = ncores
        self.ngpus = ngpus
        self.ranks = {1} if ranks is None else ranks


class _ResourceListResponse:
    def __init__(self, ncores=8, ngpus=2, ranks=None):
        self.free = _FreeResources(ncores=ncores, ngpus=ngpus, ranks=ranks)


class _ResourceListGetter:
    def __init__(self, ncores=8, ngpus=2, ranks=None):
        self._response = _ResourceListResponse(ncores=ncores, ngpus=ngpus, ranks=ranks)

    def get(self):
        return self._response


def _install_fake_flux_modules():
    flux_mod = types.ModuleType("flux")
    flux_job_mod = types.ModuleType("flux.job")
    flux_job_executor_mod = types.ModuleType("flux.job.executor")
    flux_resource_mod = types.ModuleType("flux.resource")
    flux_resource_list_mod = types.ModuleType("flux.resource.list")

    flux_mod.Flux = FakeFlux
    flux_job_mod.FluxExecutor = FakeFluxExecutor
    flux_job_mod.FluxExecutorFuture = FakeFuture
    flux_job_mod.JobspecV1 = FakeJobspecV1
    flux_job_executor_mod.FluxExecutor = FakeFluxExecutor
    flux_job_executor_mod.FluxExecutorFuture = FakeFuture
    flux_resource_list_mod.resource_list = lambda handle: _ResourceListGetter()
    flux_resource_mod.list = flux_resource_list_mod

    flux_mod.job = flux_job_mod
    flux_mod.resource = flux_resource_mod

    sys.modules["flux"] = flux_mod
    sys.modules["flux.job"] = flux_job_mod
    sys.modules["flux.job.executor"] = flux_job_executor_mod
    sys.modules["flux.resource"] = flux_resource_mod
    sys.modules["flux.resource.list"] = flux_resource_list_mod


_install_fake_flux_modules()


@pytest.fixture
def fake_future_cls():
    return FakeFuture


@pytest.fixture
def fake_executor_cls():
    return FakeFluxExecutor
