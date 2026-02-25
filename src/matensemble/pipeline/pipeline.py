from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable

from .compile import RunManifest, TaskSpec, compile_node, write_run_manifest
from .dag import topo_sort
from .model import ArgSpec, OutputRef, Resources, TaskNode, TaskTemplate


class Pipeline:
    """
    User-facing DAG builder + runner.

    Responsibilities (MVP):
    - define python tasks via @pipeline.task(...)
    - define executable tasks via pipeline.exec(...)
    - construct DAG lazily (TaskNode objects)
    - on run(): collect + topo sort, write specs, compile TaskSpecs
    - delegate execution to SuperFluxManager
    - return the final python value for the target node (file backend MVP)
    """

    def __init__(self, *, base_dir: str | Path | None = None) -> None:
        self._templates: list[TaskTemplate] = []
        self._nodes: dict[str, TaskNode] = {}
        self._base_dir = Path(base_dir) if base_dir is not None else None

    # ----------------------------
    # Task definition APIs
    # ----------------------------
    def task(
        self, *, resources: Resources | None = None
    ) -> Callable[[Callable[..., Any]], TaskTemplate]:
        """
        Decorator factory for Python tasks.

        Usage:
            @pipeline.task()
            def f(x): ...

        The decorated function becomes a TaskTemplate. Calling it returns a TaskNode
        and records dependencies from TaskNode arguments automatically.
        """
        res = resources or Resources()

        def decorator(func: Callable[..., Any]) -> TaskTemplate:
            module = func.__module__
            qualname = func.__qualname__

            tmpl = TaskTemplate(
                kind="python", resources=res, module=module, qualname=qualname
            )
            tmpl._node_factory = self._make_python_node
            self._templates.append(tmpl)
            return tmpl

        return decorator

    def exec(
        self,
        command: list[str],
        *,
        name: str | None = None,
        outputs: dict[str, str] | None = None,
        depends_on: list[TaskNode] | None = None,
        resources: Resources | None = None,
    ) -> TaskNode:
        """
        Define an executable task node.

        - command: list[str] (preferred)
        - outputs: mapping key -> relative file path (created in workdir)
        - depends_on: optional ordering-only deps
        """
        node_id = self._new_node_id(prefix=name or "exec")
        res = resources or Resources()

        node = TaskNode(
            id=node_id,
            kind="executable",
            resources=res,
            command=list(command),
            outputs_declared=dict(outputs or {}),
        )

        if depends_on:
            node.deps.update(n.id for n in depends_on)

        self._nodes[node.id] = node
        return node

    # ----------------------------
    # DAG node creation helpers
    # ----------------------------
    def _make_python_node(
        self, tmpl: TaskTemplate, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> TaskNode:
        """
        Create a python TaskNode from a TaskTemplate call, encoding args/kwargs and deps.
        """
        node_id = self._new_node_id(prefix=tmpl.qualname or "py")

        node = TaskNode(
            id=node_id,
            kind="python",
            resources=tmpl.resources,
            module=tmpl.module,
            qualname=tmpl.qualname,
        )

        node.args = [self._encode_arg(a, node) for a in args]
        node.kwargs = {k: self._encode_arg(v, node) for k, v in kwargs.items()}

        self._nodes[node.id] = node
        return node

    def _encode_arg(self, value: Any, node: TaskNode) -> ArgSpec:
        """
        Convert an argument into an ArgSpec and update node dependencies.

        Dependency inference rules (MVP):
        - TaskNode argument => node_result ref + add dep
        - OutputRef argument => output ref + add dep
        - everything else => literal
        """
        if isinstance(value, TaskNode):
            node.deps.add(value.id)
            return ArgSpec(type="node_result", node=value.id)

        if isinstance(value, OutputRef):
            node.deps.add(value.node_id)
            # store relpath in value for MVP worker; key optional
            return ArgSpec(
                type="output", node=value.node_id, key=value.key, value=value.relpath
            )

        return ArgSpec(type="lit", value=value)

    def _new_node_id(self, *, prefix: str) -> str:
        """
        Create a unique node id.

        MVP: prefix + short uuid.
        """
        return f"{prefix}-{uuid.uuid4().hex[:8]}"

    # ----------------------------
    # Running / compilation
    # ----------------------------
    def run(
        self,
        target: TaskNode | list[TaskNode] | None = None,
        *,
        include_disconnected: bool = True,
    ) -> Any:
        """Compile and execute a workflow.

        Behavior (MVP):
        - If ``target`` is provided (a TaskNode or list of TaskNodes):
            - By default (``include_disconnected=True``), all created nodes in the
              pipeline are executed, even if they are not reachable from the target.
            - If ``include_disconnected=False``, only nodes reachable from the
              target(s) are executed.
            - Return value:
                * single TaskNode target -> its python result
                * list[TaskNode] targets -> dict[target_id, result]
        - If ``target`` is None:
            - All created nodes are executed.
            - Returns a dict[node_id, result] for python tasks and a lightweight
              summary for executable tasks.
        """

        if target is None:
            targets: list[TaskNode] = []
        elif isinstance(target, list):
            targets = target
        else:
            targets = [target]

        if not self._nodes:
            return {} if target is None or isinstance(target, list) else None

        # Decide which nodes to run.
        if targets and not include_disconnected:
            node_ids = self._collect_reachable_ids([t.id for t in targets])
        else:
            node_ids = set(self._nodes.keys())

        nodes: dict[str, TaskNode] = {nid: self._nodes[nid] for nid in node_ids}

        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        base_dir = self._base_dir or Path.cwd() / f"matensemble_workflow_{run_id}"
        out_dir = base_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        order = topo_sort(nodes)

        # Write manifest + compile node specs -> TaskSpecs
        manifest = RunManifest(
            run_id=run_id,
            base_dir=base_dir,
            nodes={
                nid: str((out_dir / nid / "node_spec.json").relative_to(base_dir))
                for nid in order
            },
        )
        write_run_manifest(base_dir, manifest)

        task_specs: list[TaskSpec] = [
            compile_node(node=nodes[nid], out_dir=out_dir) for nid in order
        ]

        # Delegate execution
        from matensemble.manager import SuperFluxManager

        mgr = SuperFluxManager(tasks=task_specs, base_dir=base_dir)
        mgr.run()

        # --- Results ---
        # 1) No explicit targets -> return all results.
        if target is None:
            return self._collect_results(task_specs)

        # 2) Multi-target -> dict of those targets.
        if isinstance(target, list):
            all_results = self._collect_results(task_specs)
            return {t.id: all_results.get(t.id) for t in targets}

        # 3) Single target -> preserve old behavior: return python result only.
        target_spec = next(ts for ts in task_specs if ts.id == targets[0].id)
        if target_spec.result_path is None:
            raise ValueError(
                "pipeline.run(target) expects a python task target for MVP return-value semantics. "
                "Use pipeline.run() (no args) to get a dict of all task results."
            )
        return _load_pickle(target_spec.result_path)

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _collect_reachable_ids(self, root_ids: Iterable[str]) -> set[str]:
        """Collect all nodes reachable from the given roots by following deps."""
        reachable: set[str] = set()
        stack = list(root_ids)
        while stack:
            nid = stack.pop()
            if nid in reachable:
                continue
            if nid not in self._nodes:
                raise KeyError(f"Unknown node id '{nid}'")
            reachable.add(nid)
            stack.extend(list(self._nodes[nid].deps))
        return reachable

    def _collect_results(self, task_specs: list[TaskSpec]) -> dict[str, Any]:
        """Load python task results and summarize executable tasks."""
        results: dict[str, Any] = {}
        for ts in task_specs:
            if ts.kind == "python" and ts.result_path is not None:
                results[ts.id] = _load_pickle(ts.result_path)
            else:
                # executables: return declared outputs (absolute paths) and workdir
                outputs = {k: str(p) for k, p in (ts.outputs_declared or {}).items()}
                results[ts.id] = {
                    "workdir": str(ts.workdir),
                    "outputs": outputs,
                }
        return results


def _load_pickle(path: Path) -> Any:
    import pickle

    with path.open("rb") as f:
        return pickle.load(f)
