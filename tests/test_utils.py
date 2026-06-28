from dataclasses import dataclass
from pathlib import Path

from starlette.testclient import TestClient

from matensemble.logger import StatusWriter
from matensemble.model import OutputReference
from matensemble.utils import (
    _collect_dep_ids,
    _json_safe,
    _resolve_output_references,
    create_app,
)


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


def test_dashboard_api_serves_v2_status_and_history(tmp_path: Path):
    status_path = tmp_path / "status.json"
    writer = StatusWriter(status_path, 2, 8, 1)
    writer.update(
        pending=2,
        ready=1,
        blocked=1,
        running=1,
        completed=3,
        failed=0,
        free_cores=8,
        free_gpus=1,
    )

    with TestClient(create_app(str(status_path))) as client:
        status = client.get("/api/status")
        history = client.get("/api/history")

    assert status.status_code == 200
    assert status.json()["schema_version"] == 2
    assert status.json()["current"]["ready"] == 1
    assert history.status_code == 200
    assert history.json()[0]["sequence"] == 0


def test_dashboard_api_upgrades_legacy_status(tmp_path: Path):
    status_path = tmp_path / "status.json"
    status_path.write_text(
        '{"nodes":1,"cores_per_node":4,"pending":1,"running":0,"completed":0,"failed":0}',
        encoding="utf-8",
    )

    with TestClient(create_app(str(status_path))) as client:
        status = client.get("/api/status").json()
        history = client.get("/api/history").json()

    assert status["schema_version"] == 2
    assert status["source_schema_version"] == 1
    assert len(history) == 1
