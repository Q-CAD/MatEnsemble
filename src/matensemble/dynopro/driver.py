from __future__ import annotations

import argparse
import inspect
import pickle
from pathlib import Path
from typing import Any

import cloudpickle

from matensemble.utils import _resolve_output_references

__author__ = "Soumendu Bagchi"
__package__ = "matensemble"


def _load_dep_result(chore_dir: Path, dep_id: str) -> Any:
    dep_result = chore_dir.parent / dep_id / "result.pickle"
    with dep_result.open("rb") as f:
        return pickle.load(f)


def _callable_accepts_kwargs(func: Any) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False

    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _maybe_add_kwarg(func: Any, kwargs: dict[str, Any], key: str, value: Any) -> None:
    if key in kwargs:
        return

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return

    if key in signature.parameters or _callable_accepts_kwargs(func):
        kwargs[key] = value


def _run_chore(
    chore_name: str,
    chore_dir: str | Path,
    *,
    split: Any,
    comm: Any,
    color: int,
) -> Any:
    chore_dir = Path(chore_dir).resolve()
    registry = chore_dir.parent / "registry"
    with (registry / chore_name).open("rb") as f:
        func = cloudpickle.load(f)
    with (chore_dir / "chore.pickle").open("rb") as f:
        metadata = pickle.load(f)

    dep_results = {dep: _load_dep_result(chore_dir, dep) for dep in metadata.deps}
    args = _resolve_output_references(metadata.args, dep_results)
    kwargs = _resolve_output_references(metadata.kwargs, dep_results)

    _maybe_add_kwarg(func, kwargs, "split", split)
    _maybe_add_kwarg(func, kwargs, "comm", comm)
    _maybe_add_kwarg(func, kwargs, "world_comm", comm)
    _maybe_add_kwarg(func, kwargs, "rank_color", color)
    _maybe_add_kwarg(func, kwargs, "chore_dir", chore_dir)

    return func(*args, **kwargs)


def _barrier(comm: Any) -> None:
    if hasattr(comm, "Barrier"):
        comm.Barrier()
    else:
        comm.barrier()


def _gather_results(comm: Any, result: Any) -> list[Any] | None:
    if hasattr(comm, "gather"):
        return comm.gather(result, root=0)
    if hasattr(comm, "Gather"):
        return comm.Gather(result, root=0)
    return [result]


def online_dynamics(
    gpus_per_node: int,
    cores_per_node: int,
    gpu_subprocess: str,
    cpu_subprocess: str,
    chore_dir: str | Path,
) -> list[Any] | None:
    """
    Split MPI ranks into per-node GPU and CPU groups and run registered chores.

    Ranks with local rank ``0 <= local_rank < gpus_per_node`` run
    ``gpu_subprocess``. The remaining ranks run ``cpu_subprocess``. Registered
    callables receive the dynopro chore's serialized args/kwargs. If they declare
    ``split``, ``comm``, ``world_comm``, ``rank_color``, or ``chore_dir`` keyword
    parameters, those runtime values are injected.
    """

    if gpus_per_node < 1:
        raise ValueError("gpus_per_node must be >= 1")
    if cores_per_node < 1:
        raise ValueError("cores_per_node must be >= 1")
    if gpus_per_node > cores_per_node:
        raise ValueError("gpus_per_node cannot exceed cores_per_node")

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    local_rank = rank % cores_per_node
    color = 0 if local_rank < gpus_per_node else 1
    split = comm.Split(color, key=local_rank)

    if color == 0:
        result = _run_chore(
            gpu_subprocess, chore_dir, split=split, comm=comm, color=color
        )
    else:
        result = _run_chore(
            cpu_subprocess, chore_dir, split=split, comm=comm, color=color
        )
        print(f"shutting down rank: {rank}")

    gathered = _gather_results(comm, {"rank": rank, "color": color, "result": result})
    _barrier(comm)

    if rank == 0:
        chore_dir = Path(chore_dir).resolve()
        with (chore_dir / "result.pickle").open("wb") as f:
            pickle.dump(gathered, f)
        print("Exiting Simulation Environment")

    if hasattr(MPI, "Finalize"):
        already_finalized = hasattr(MPI, "Is_finalized") and MPI.Is_finalized()
        if not already_finalized:
            MPI.Finalize()

    return gathered if rank == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run online dynamics simulation")
    parser.add_argument("--gpus-per-node", "--gpus_per_node", type=int, required=True)
    parser.add_argument("--cores-per-node", "--cores_per_node", type=int, required=True)
    parser.add_argument("--gpu-subprocess", "--gpu_subprocess", type=str, required=True)
    parser.add_argument("--cpu-subprocess", "--cpu_subprocess", type=str, required=True)
    parser.add_argument("--chore-dir", "--chore_dir", type=str, required=True)
    args = parser.parse_args()

    print("Starting online dynamics now . . . ")
    online_dynamics(
        args.gpus_per_node,
        args.cores_per_node,
        args.gpu_subprocess,
        args.cpu_subprocess,
        args.chore_dir,
    )


if __name__ == "__main__":
    main()
