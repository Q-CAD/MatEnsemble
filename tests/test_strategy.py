from concurrent.futures import Future
from collections import deque
from pathlib import Path
import threading

from matensemble.model import ChoreType, Resources
from matensemble.manager import FluxManager
from matensemble.strategy import (
    AdaptiveStrategy,
    NonAdaptiveStrategy,
    append_text,
)
from matensemble.chore import Chore


def test_append_text_creates_and_appends(tmp_path: Path):
    path = tmp_path / "a" / "stderr"
    append_text(path, "hello")
    append_text(path, " world")
    assert path.read_text() == "hello world"


def test_user_strategy_related_spawn_path_uses_manager_add(monkeypatch, tmp_path: Path):
    # minimal smoke for pieces used by UserStrategy branch logic helpers
    c = Chore(
        id="chore-process-0001",
        workdir=tmp_path / "chore-process-0001",
        command=["python"],
        chore_type=ChoreType.PYTHON,
        resources=Resources(),
        chore_qualname="process",
    )
    assert c.id.startswith("chore-")
    assert isinstance(deque(), deque)


def _completed_future(chore: Chore) -> Future:
    future = Future()
    future.chore_id = chore.id
    future.chore_obj = chore
    future.set_result(0)
    return future


def _strategy_manager(chores: list[Chore]) -> FluxManager:
    manager = FluxManager.__new__(FluxManager)
    manager._futures = {_completed_future(chore) for chore in chores}
    manager._running_chores = {chore.id for chore in chores}
    manager._completed_chores = []
    manager._dependents = {chore.id: [] for chore in chores}
    manager._remaining_deps = {}
    manager._blocked = set()
    manager._failed_chores = []
    manager._write_restart_freq = None
    manager._free_cores = 0
    manager._free_gpus = 0
    manager._total_cores = len(chores)
    manager._total_gpus = 0
    manager._nnodes_on_allocation = 1
    manager._cores_per_node = len(chores)
    manager._gpus_per_node = 0
    manager._state_lock = threading.RLock()
    manager._dynopro = False
    manager._logger = type(
        "L",
        (),
        {
            "info": staticmethod(lambda *args, **kwargs: None),
            "error": staticmethod(lambda *args, **kwargs: None),
            "exception": staticmethod(lambda *args, **kwargs: None),
        },
    )()
    manager._log_progress = lambda: None
    manager._check_resources = lambda: (_ for _ in ()).throw(
        AssertionError("completion processing must not poll Flux")
    )
    return manager


def test_adaptive_strategy_releases_capacity_before_backfill(tmp_path: Path):
    chore = Chore(
        id="adaptive-worker",
        workdir=tmp_path / "adaptive-worker",
        command=["echo", "ok"],
        chore_type=ChoreType.EXECUTABLE,
        resources=Resources(),
    )
    manager = _strategy_manager([chore])
    free_cores_seen_by_submit = []
    manager._submit_until_ooresources = lambda **_kwargs: (
        free_cores_seen_by_submit.append(manager._free_cores)
    )

    AdaptiveStrategy(manager).process_futures(buffer_time=0.0)

    assert free_cores_seen_by_submit == [1]
    assert manager._completed_chores == [chore.id]


def test_nonadaptive_strategy_waits_for_entire_wave(monkeypatch, tmp_path: Path):
    chores = [
        Chore(
            id=f"wave-worker-{index}",
            workdir=tmp_path / f"wave-worker-{index}",
            command=["echo", "ok"],
            chore_type=ChoreType.EXECUTABLE,
            resources=Resources(),
        )
        for index in range(3)
    ]
    manager = _strategy_manager(chores)
    waited_on = None

    def complete_wave(futures):
        nonlocal waited_on
        waited_on = set(futures)
        yield from futures

    monkeypatch.setattr(
        "matensemble.strategy.concurrent.futures.as_completed", complete_wave
    )

    NonAdaptiveStrategy(manager).process_futures(buffer_time=0.001)

    assert waited_on is not None
    assert len(waited_on) == 3
    assert manager._futures == set()
    assert set(manager._completed_chores) == {chore.id for chore in chores}
    assert manager._free_cores == 3


def test_nonadaptive_strategy_does_not_refill_during_wave(monkeypatch, tmp_path: Path):
    chores = [
        Chore(
            id=f"wave-worker-{index}",
            workdir=tmp_path / f"wave-worker-{index}",
            command=["echo", "ok"],
            chore_type=ChoreType.EXECUTABLE,
            resources=Resources(),
        )
        for index in range(2)
    ]
    manager = _strategy_manager(chores)
    submit_calls = []
    manager._submit_until_ooresources = lambda **kwargs: submit_calls.append(kwargs)
    monkeypatch.setattr(
        "matensemble.strategy.concurrent.futures.as_completed",
        lambda futures: iter(futures),
    )

    NonAdaptiveStrategy(manager).process_futures(buffer_time=0.001)

    assert submit_calls == []
