from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from matensemble.model import JobFlavor, OutputReference
from matensemble.utils import (
    _collect_dep_ids,
    _find_refs,
    _json_safe,
    _resolve_output_references,
)


@dataclass
class Payload:
    value: object
    extra: object


class Node:
    def __init__(self):
        self.items = []

    def __iter__(self):
        return iter(self.items)


def test_json_safe_handles_paths_enums_and_output_references():
    value = {
        "path": Path("abc"),
        "flavor": JobFlavor.PYTHON,
        "ref": OutputReference("job-1"),
        "nested": (1, {2, 3}),
    }
    safe = _json_safe(value)

    assert safe["path"] == "abc"
    assert safe["flavor"] == "PYTHON"
    assert safe["ref"] == {"__type__": "OutputReference", "job_id": "job-1"}
    assert safe["nested"][0] == 1
    assert sorted(safe["nested"][1]) == [2, 3]


def test_find_refs_walks_nested_containers_and_dataclasses_once():
    ref1 = OutputReference("job-a")
    ref2 = OutputReference("job-b")
    payload = Payload(value=[ref1, {"x": ref2}], extra=(ref1,))
    node = Node()
    node.items.append(payload)
    node.items.append(ref2)

    refs = list(_find_refs((node,), {"kw": payload}))
    assert [ref.job_id for ref in refs] == ["job-a", "job-b"]


def test_collect_dep_ids_preserves_first_seen_order_and_uniqueness():
    ref1 = OutputReference("job-a")
    ref2 = OutputReference("job-b")
    deps = _collect_dep_ids(([ref1, ref2, ref1],), {"x": {"y": ref2}})
    assert deps == ("job-a", "job-b")


def test_resolve_output_references_preserves_shape():
    value = Payload(
        value={OutputReference("job-a"): [OutputReference("job-b"), 9]},
        extra=frozenset({OutputReference("job-b")}),
    )
    resolved = _resolve_output_references(value, {"job-a": "left", "job-b": "right"})

    assert isinstance(resolved, Payload)
    assert resolved.value == {"left": ["right", 9]}
    assert resolved.extra == frozenset({"right"})


def test_resolve_output_references_raises_for_missing_dependency():
    with pytest.raises(KeyError, match="Missing dependency result"):
        _resolve_output_references(OutputReference("missing"), {})
