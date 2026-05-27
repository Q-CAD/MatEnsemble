from dataclasses import dataclass
from pathlib import Path

from matensemble.model import OutputReference
from matensemble.utils import _collect_dep_ids, _json_safe, _resolve_output_references


@dataclass
class Box:
    value: object


def test_collect_dep_ids_deduplicates_nested_refs(tmp_path: Path):
    ref = OutputReference("up-1", tmp_path / "up-1")
    args = ([ref, {"k": ref}],)
    deps = _collect_dep_ids(args, {})
    assert deps == ("up-1",)


def test_resolve_output_references_preserves_shapes(tmp_path: Path):
    ref = OutputReference("dep-1", tmp_path / "dep-1")
    value = {"x": [Box(ref), (ref,)]}
    resolved = _resolve_output_references(value, {"dep-1": 10})
    assert resolved["x"][0].value == 10
    assert resolved["x"][1] == (10,)


def test_json_safe_handles_paths_and_output_refs(tmp_path: Path):
    ref = OutputReference("abc", tmp_path / "abc")
    out = _json_safe({"p": tmp_path, "ref": ref})
    assert out["p"] == str(tmp_path)
    assert out["ref"]["chore_id"] == "abc"
