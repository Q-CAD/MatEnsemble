from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import ArgSpec, OutputRef, Resources, TaskNode, TaskKind


@dataclass(frozen=True)
class TaskSpec:
    """
    Compiled task representation consumed by SuperFluxManager / Fluxlet.
    """

    id: str
    deps: list[str]
    kind: TaskKind
    command: list[str]
    resources: Resources
    workdir: Path

    # exec-only validation targets: output key -> absolute path
    outputs_declared: dict[str, Path]

    # python-only: where this task stores its return value (file backend MVP)
    result_path: Path | None = None

    # where the worker reads its node spec
    node_spec_path: Path | None = None


@dataclass(frozen=True)
class RunManifest:
    """
    Run-level manifest written once per pipeline.run(...).

    For MVP, this mostly exists to aid debugging/restarts.
    """

    run_id: str
    base_dir: Path
    nodes: dict[str, str]  # node_id -> relative node_spec path (string)


def write_run_manifest(base_dir: Path, manifest: RunManifest) -> Path:
    """
    Write a run manifest JSON into the run directory.
    """
    path = base_dir / "manifest.json"
    payload = {
        "run_id": manifest.run_id,
        "base_dir": str(manifest.base_dir),
        "nodes": manifest.nodes,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_node_spec(workdir: Path, node: TaskNode) -> Path:
    """
    Write a per-node spec JSON.

    The worker reads this for python tasks.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "node_spec.json"
    payload: dict[str, Any] = {
        "id": node.id,
        "kind": node.kind,
        "deps": sorted(node.deps),
        "resources": {
            "num_tasks": node.resources.num_tasks,
            "cores_per_task": node.resources.cores_per_task,
            "gpus_per_task": node.resources.gpus_per_task,
            "mpi": node.resources.mpi,
            "env": node.resources.env,
        },
        "python": {"module": node.module, "qualname": node.qualname},
        "args": [a.__dict__ for a in node.args],
        "kwargs": {k: v.__dict__ for k, v in node.kwargs.items()},
        "command": node.command,
        "outputs_declared": node.outputs_declared,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def compile_node(
    *,
    node: TaskNode,
    out_dir: Path,
) -> TaskSpec:
    """
    Compile a TaskNode into a TaskSpec (final Flux command + metadata).

    - python nodes compile into: python -m matensemble.pipeline.runtime_worker --node-spec ...
    - executable nodes compile into their direct command list, with OutputRefs already resolved
      by Pipeline before calling compile_node (MVP).
    """
    workdir = out_dir / node.id
    node_spec_path = write_node_spec(workdir, node)

    if node.kind == "python":
        result_path = workdir / "result.pkl"
        cmd = [
            "python",
            "-m",
            "matensemble.pipeline.runtime_worker",
            "--node-spec",
            str(node_spec_path),
        ]
        return TaskSpec(
            id=node.id,
            deps=sorted(node.deps),
            kind=node.kind,
            command=cmd,
            resources=node.resources,
            workdir=workdir,
            outputs_declared={},
            result_path=result_path,
            node_spec_path=node_spec_path,
        )

    # executable
    outputs_abs = {k: (workdir / rel) for k, rel in node.outputs_declared.items()}
    if node.command is None:
        raise ValueError(f"Executable node {node.id} missing command")
    return TaskSpec(
        id=node.id,
        deps=sorted(node.deps),
        kind=node.kind,
        command=node.command,
        resources=node.resources,
        workdir=workdir,
        outputs_declared=outputs_abs,
        result_path=None,
        node_spec_path=node_spec_path,
    )
