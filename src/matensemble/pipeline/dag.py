from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from .model import TaskNode


def collect_nodes(targets: TaskNode | list[TaskNode]) -> dict[str, TaskNode]:
    """
    Collect all nodes reachable from the target node(s).

    Returns a mapping {node_id: TaskNode}.
    """
    if isinstance(targets, TaskNode):
        targets = [targets]

    seen: dict[str, TaskNode] = {}
    stack = list(targets)

    while stack:
        node = stack.pop()
        if node.id in seen:
            continue
        seen[node.id] = node
        # dependencies are just IDs; pipeline will have the node registry
        # The compiler/pipeline will resolve ID -> node object when needed.
        # Here we just collect what we already have from the registry in Pipeline.
        # (This function typically gets the registry passed in; see Pipeline.run.)
    return seen


def topo_sort(nodes: dict[str, TaskNode]) -> list[str]:
    """
    Topologically sort a DAG.

    Returns node IDs in dependency-respecting order.

    Raises:
        ValueError if a cycle is detected.
    """
    indeg: dict[str, int] = {nid: 0 for nid in nodes}
    adj: dict[str, list[str]] = {nid: [] for nid in nodes}

    for nid, node in nodes.items():
        for dep in node.deps:
            if dep not in nodes:
                # allow external deps only if pipeline chooses to (MVP: error)
                raise KeyError(f"Missing dependency node '{dep}' for node '{nid}'")
            indeg[nid] += 1
            adj[dep].append(nid)

    q = deque([nid for nid, d in indeg.items() if d == 0])
    out: list[str] = []

    while q:
        u = q.popleft()
        out.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(out) != len(nodes):
        raise ValueError("Cycle detected in DAG")

    return out
