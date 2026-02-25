from __future__ import annotations

import hashlib
import inspect
import sys
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
            user_qualname = func.__qualname__

            # We cannot rely on the symbol name in the user's module because the
            # decorator replaces it with a TaskTemplate object. So we stash the
            # original function under a deterministic, collision-resistant name.
            src = inspect.getsourcefile(func) or inspect.getfile(func)
            source_path: str | None = None
            if src and not str(src).startswith("<"):
                source_path = str(Path(src).resolve())

            if source_path is None:
                raise RuntimeError(
                    "@pipeline.task() requires functions defined in a real .py file "
                    "(could not determine source path)."
                )

            impl_name = (
                "__matensemble_impl_"
                + hashlib.sha1(f"{source_path}:{user_qualname}".encode()).hexdigest()[
                    :12
                ]
            )

            mod_obj = sys.modules.get(module)
            if mod_obj is not None:
                setattr(mod_obj, impl_name, func)

            tmpl = TaskTemplate(
                kind="python",
                resources=res,
                module=module,
                qualname=impl_name,
                user_qualname=user_qualname,
                source_path=source_path,
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
        node_id = self._new_node_id(prefix=tmpl.user_qualname or "py")

        node = TaskNode(
            id=node_id,
            kind="python",
            resources=tmpl.resources,
            module=tmpl.module,
            qualname=tmpl.qualname,
            user_qualname=tmpl.user_qualname,
            source_path=tmpl.source_path,
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
    def run(self, target: TaskNode) -> Any:
        """
        Compile and execute the DAG ending at `target`.

        Steps:
        1) Collect reachable nodes (following deps via self._nodes)
        2) Topologically sort
        3) Create run directory and write manifest/node specs
        4) Compile TaskSpecs (final commands)
        5) Execute via SuperFluxManager
        6) Return target python result (file backend MVP)
        """
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        base_dir = self._base_dir or Path.cwd() / f"matensemble_workflow_{run_id}"
        out_dir = base_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Collect reachable nodes (DFS by deps ids)
        reachable: dict[str, TaskNode] = {}
        stack = [target.id]
        while stack:
            nid = stack.pop()
            if nid in reachable:
                continue
            node = self._nodes[nid]
            reachable[nid] = node
            stack.extend(list(node.deps))

        order = topo_sort(reachable)

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

        task_specs: list[TaskSpec] = []
        for nid in order:
            node = reachable[nid]
            # MVP: executable OutputRef substitution happens earlier (not shown here yet)
            task_specs.append(compile_node(node=node, out_dir=out_dir))

        # Delegate execution (placeholder import to avoid circulars in skeleton)
        from matensemble.manager import SuperFluxManager

        mgr = SuperFluxManager(tasks=task_specs, base_dir=base_dir)
        mgr.run()

        # Return final result for python target
        target_spec = next(ts for ts in task_specs if ts.id == target.id)
        if target_spec.result_path is None:
            raise ValueError(
                "pipeline.run() target must be a python task for MVP return-value semantics"
            )
        return _load_pickle(target_spec.result_path)


def _load_pickle(path: Path) -> Any:
    import pickle

    with path.open("rb") as f:
        return pickle.load(f)
