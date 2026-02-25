from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence


TaskKind = Literal["python", "executable"]
ArgType = Literal["lit", "node_result", "output"]


@dataclass(frozen=True)
class Resources:
    """
    Per-task resource requirements.

    These are carried by TaskNodes and later copied into TaskSpecs.
    """

    num_tasks: int = 1
    cores_per_task: int = 1
    gpus_per_task: int = 0
    mpi: bool = False
    env: dict[str, str] | None = None


@dataclass(frozen=True)
class OutputRef:
    """
    Placeholder reference to a declared output file of an executable task.

    This is used in DAG construction and later compiled into an absolute path.
    """

    node_id: str
    key: str
    relpath: str  # e.g. "result.json"


@dataclass(frozen=True)
class ArgSpec:
    """
    JSON-serializable argument specification.

    - lit: raw JSON-serializable literal (or pickle via backend if desired later)
    - node_result: reference to upstream python-task result
    - output: reference to upstream declared file output
    """

    type: ArgType
    value: Any = None
    node: str | None = None
    key: str | None = None


@dataclass
class TaskTemplate:
    """
    A reusable task definition.

    For python tasks, created by @pipeline.task(). Calling it creates TaskNodes.
    For executables, Pipeline.exec(...) directly creates TaskNodes (template optional).
    """

    kind: TaskKind
    resources: Resources

    # python-only reference
    module: str | None = None
    # Runtime-resolvable dotted qualname (may be an internal impl name)
    qualname: str | None = None
    # Human-facing name (original user qualname)
    user_qualname: str | None = None
    # Absolute source path used as a fallback loader (esp. when module == "__main__")
    source_path: str | None = None

    # set by Pipeline when constructed
    _node_factory: Callable[..., "TaskNode"] | None = field(default=None, repr=False)

    def __call__(self, *args: Any, **kwargs: Any) -> "TaskNode":
        """
        Create a TaskNode invocation (a DAG node).

        This MUST NOT execute the underlying function.
        """
        if self._node_factory is None:
            raise RuntimeError("TaskTemplate is not bound to a Pipeline.")
        return self._node_factory(self, args, kwargs)


@dataclass
class TaskNode:
    """
    One invocation in the DAG.

    This is what users manipulate and pass as arguments to create dependencies.
    """

    id: str
    kind: TaskKind
    resources: Resources

    # python-only
    module: str | None = None
    # Runtime-resolvable dotted qualname (may be an internal impl name)
    qualname: str | None = None
    # Human-facing name (original user qualname)
    user_qualname: str | None = None
    # Absolute source path used as a fallback loader (esp. when module == "__main__")
    source_path: str | None = None

    deps: set[str] = field(default_factory=set)
    args: list[ArgSpec] = field(default_factory=list)
    kwargs: dict[str, ArgSpec] = field(default_factory=dict)

    # executable-only
    command: list[str] | None = None
    outputs_declared: dict[str, str] = field(default_factory=dict)

    def output(self, key: str) -> OutputRef:
        """
        Get an OutputRef for a declared output.
        """
        if key not in self.outputs_declared:
            raise KeyError(f"TaskNode {self.id} has no declared output '{key}'")
        return OutputRef(node_id=self.id, key=key, relpath=self.outputs_declared[key])
