from collections import deque
from concurrent.futures import Future
from pathlib import Path
import threading

from matensemble.chore import Chore
from matensemble.manager import FluxManager
from matensemble.model import ChoreType, Resources


def _chore(chore_id: str, deps=()):
    return Chore(
        id=chore_id,
        workdir=Path.cwd() / "tmp" / chore_id,
        command=["echo", "ok"],
        chore_type=ChoreType.EXECUTABLE,
        resources=Resources(),
        deps=deps,
    )


def test_record_failure_is_idempotent():
    manager = FluxManager.__new__(FluxManager)
    manager._failed_chores = []
    manager._record_failure("a", "x")
    manager._record_failure("a", "x")
    assert len(manager._failed_chores) == 1
    assert manager._failed_chores[0]["exception"] is None


def test_fail_dependents_marks_children():
    manager = FluxManager.__new__(FluxManager)
    manager._dependents = {"a": ["b"], "b": []}
    manager._completed_chores = []
    manager._running_chores = set()
    manager._ready = deque(["b"])
    manager._blocked = {"b"}
    manager._failed_chores = []
    manager._logger = type(
        "L", (), {"error": staticmethod(lambda *args, **kwargs: None)}
    )()
    manager._has_failed = FluxManager._has_failed.__get__(manager, FluxManager)
    manager._record_failure = FluxManager._record_failure.__get__(manager, FluxManager)
    manager._fail_dependents = FluxManager._fail_dependents.__get__(
        manager, FluxManager
    )
    manager._fail_dependents("a")
    assert manager._has_failed("b")


def test_manager_polls_flux_resources_once_at_start(monkeypatch, tmp_path: Path):
    calls = 0

    class _Resources:
        class free:
            ranks = [1, 2]
            ncores = 8
            ngpus = 2

    class _ResourceList:
        def get(self):
            return _Resources()

    def resource_list(_handle):
        nonlocal calls
        calls += 1
        return _ResourceList()

    monkeypatch.setattr("flux.resource.list.resource_list", resource_list)

    manager = FluxManager(
        chore_list=[_chore("a")],
        base_dir=tmp_path / "workflow",
    )

    assert calls == 1
    assert manager._total_cores == manager._free_cores == 8
    assert manager._total_gpus == manager._free_gpus == 2
    assert manager._fluxlet.num_nodes == 2
    assert manager._fluxlet.gpus_per_node == 1


def test_submit_until_resources_fills_capacity_without_delay(monkeypatch):
    chores = [_chore(f"chore-{index}") for index in range(3)]
    manager = FluxManager.__new__(FluxManager)
    manager._chores_by_id = {chore.id: chore for chore in chores}
    manager._ready = deque(chore.id for chore in chores)
    manager._blocked = set()
    manager._running_chores = set()
    manager._futures = set()
    manager._free_cores = manager._total_cores = 2
    manager._free_gpus = manager._total_gpus = 0
    manager._cores_per_node = 2
    manager._gpus_per_node = 0
    manager._state_lock = threading.RLock()
    manager._executor = object()
    manager._set_cpu_affinity = True
    manager._set_gpu_affinity = False

    class _Fluxlet:
        @staticmethod
        def submit(*_args, **_kwargs):
            return Future()

    manager._fluxlet = _Fluxlet()
    monkeypatch.setattr(
        "matensemble.manager.time.sleep",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("submission must not sleep")
        ),
    )

    submitted = manager._submit_until_ooresources(buffer_time=1.0)

    assert submitted is True
    assert len(manager._running_chores) == 2
    assert len(manager._futures) == 2
    assert list(manager._ready) == ["chore-2"]
    assert manager._free_cores == 0


class _ResourceState:
    def __init__(self, ranks, ncores, ngpus=0):
        self.ranks = ranks
        self.ncores = ncores
        self.ngpus = ngpus


class _ResourceSnapshot:
    def __init__(self, all_ranks, free_ranks, ncores, ngpus=0):
        self.all = _ResourceState(all_ranks, ncores, ngpus)
        self.free = _ResourceState(free_ranks, ncores, ngpus)


class _RpcResult:
    def get(self):
        return None


class _RecordingHandle:
    def __init__(self):
        self.calls = []

    def rpc(self, topic, payload):
        self.calls.append((topic, payload))
        return _RpcResult()


def _allocation_manager(reserve_broker_node=None, controller_cores=None):
    manager = FluxManager.__new__(FluxManager)
    manager._flux_handle = _RecordingHandle()
    manager._requested_reserve_broker_node = reserve_broker_node
    manager._requested_controller_cores = controller_cores
    manager._reserve_broker_node = False
    manager._controller_cores = 0
    manager._drained_broker_node = False
    return manager


def test_single_rank_auto_mode_shares_rank_zero_and_reserves_one_core(monkeypatch):
    manager = _allocation_manager()
    snapshot = _ResourceSnapshot([0], [0], ncores=8, ngpus=2)
    monkeypatch.setattr(
        "flux.resource.list.resource_list",
        lambda _handle: type("Result", (), {"get": lambda _self: snapshot})(),
    )

    assert manager._get_allocation_info() == (1, 7, 2)
    assert manager._reserve_broker_node is False
    assert manager._controller_cores == 1
    assert manager._flux_handle.calls == []

    manager._check_resources()
    assert manager._free_cores == 7
    assert manager._free_gpus == 2


def test_multi_rank_auto_mode_drains_and_restores_rank_zero(monkeypatch):
    manager = _allocation_manager()
    before = _ResourceSnapshot([0, 1, 2], [0, 1, 2], ncores=24, ngpus=3)
    after = _ResourceSnapshot([0, 1, 2], [1, 2], ncores=16, ngpus=2)
    snapshots = iter([before, after])
    monkeypatch.setattr(
        "flux.resource.list.resource_list",
        lambda _handle: type("Result", (), {"get": lambda _self: next(snapshots)})(),
    )

    assert manager._get_allocation_info() == (2, 8, 1)
    assert manager._flux_handle.calls == [("resource.drain", {"targets": "0"})]

    manager._restore_broker_node()
    assert manager._flux_handle.calls[-1] == (
        "resource.undrain",
        {"targets": "0"},
    )
    assert manager._drained_broker_node is False


def test_forced_shared_multi_rank_mode_does_not_reserve_cores(monkeypatch):
    manager = _allocation_manager(reserve_broker_node=False)
    snapshot = _ResourceSnapshot([0, 1], [0, 1], ncores=16, ngpus=4)
    monkeypatch.setattr(
        "flux.resource.list.resource_list",
        lambda _handle: type("Result", (), {"get": lambda _self: snapshot})(),
    )

    assert manager._get_allocation_info() == (2, 8, 2)
    assert manager._controller_cores == 0
    assert manager._flux_handle.calls == []


def test_dedicated_mode_rejects_single_rank(monkeypatch):
    manager = _allocation_manager(reserve_broker_node=True)
    snapshot = _ResourceSnapshot([0], [0], ncores=8)
    monkeypatch.setattr(
        "flux.resource.list.resource_list",
        lambda _handle: type("Result", (), {"get": lambda _self: snapshot})(),
    )

    try:
        manager._get_allocation_info()
    except ValueError as exc:
        assert "single-rank" in str(exc)
    else:
        raise AssertionError("expected dedicated single-rank mode to fail")


def test_single_rank_rejects_controller_reservation_using_all_cores(monkeypatch):
    manager = _allocation_manager(
        reserve_broker_node=False,
        controller_cores=8,
    )
    snapshot = _ResourceSnapshot([0], [0], ncores=8)
    monkeypatch.setattr(
        "flux.resource.list.resource_list",
        lambda _handle: type("Result", (), {"get": lambda _self: snapshot})(),
    )

    try:
        manager._get_allocation_info()
    except ValueError as exc:
        assert "leaves no cores" in str(exc)
    else:
        raise AssertionError("expected excessive controller reservation to fail")


def test_run_restores_owned_drain_when_workflow_raises():
    manager = FluxManager.__new__(FluxManager)
    manager._flux_handle = _RecordingHandle()
    manager._drained_broker_node = True

    def fail(**_kwargs):
        raise RuntimeError("workflow failed")

    manager._run_workflow = fail

    try:
        manager.run()
    except RuntimeError as exc:
        assert str(exc) == "workflow failed"
    else:
        raise AssertionError("expected workflow failure")

    assert manager._flux_handle.calls == [("resource.undrain", {"targets": "0"})]


def test_reserved_controller_core_limits_chore_fit():
    manager = FluxManager.__new__(FluxManager)
    manager._nnodes_on_allocation = 1
    manager._cores_per_node = 7
    manager._gpus_per_node = 1

    assert manager._chore_fits_allocation(
        Chore(
            id="fits",
            workdir=Path.cwd() / "tmp" / "fits",
            command=["true"],
            chore_type=ChoreType.EXECUTABLE,
            resources=Resources(num_tasks=7),
        )
    )
    assert not manager._chore_fits_allocation(
        Chore(
            id="too-large",
            workdir=Path.cwd() / "tmp" / "too-large",
            command=["true"],
            chore_type=ChoreType.EXECUTABLE,
            resources=Resources(num_tasks=8),
        )
    )
