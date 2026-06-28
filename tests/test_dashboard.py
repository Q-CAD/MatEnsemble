import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from matensemble.dashboard import WorkflowCatalog, create_dashboard_app
from matensemble.dashboard.app import read_history


def timestamp(*, seconds_ago: int = 0) -> str:
    value = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def status_payload(
    *,
    name: str = "demo",
    state: str = "running",
    updated_ago: int = 0,
) -> dict:
    now = timestamp(seconds_ago=updated_ago)
    return {
        "schema_version": 2,
        "workflow": {
            "id": "metadata-only-id",
            "name": name,
            "campaign": "display-only",
            "state": state,
            "started_at": timestamp(seconds_ago=120),
            "updated_at": now,
            "finished_at": now if state in {"completed", "failed", "interrupted"} else None,
            "elapsed_seconds": 120,
        },
        "allocation": {
            "nodes": 2,
            "cores_per_node": 8,
            "gpus_per_node": 1,
            "total_cores": 16,
            "total_gpus": 2,
        },
        "current": {
            "sequence": 2,
            "pending": 2,
            "ready": 1,
            "blocked": 1,
            "running": 1,
            "completed": 3,
            "failed": 0,
            "free_cores": 8,
            "free_gpus": 1,
        },
        "failures": [],
        "history_file": "status_history.jsonl",
    }


def make_workflow(
    root: Path,
    relative_parent: str,
    stamp: str,
    *,
    payload: dict | None = None,
) -> Path:
    path = root / relative_parent / f"matensemble_workflow-{stamp}"
    path.mkdir(parents=True)
    if payload is not None:
        (path / "status.json").write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_recursive_discovery_grouping_ids_and_starting(tmp_path: Path):
    first = make_workflow(
        tmp_path, "campaign_a/experiment", "20260622_143000",
        payload=status_payload(name="nested"),
    )
    second = make_workflow(
        tmp_path, "campaign_b", "20260622_143000",
        payload=status_payload(name="same stamp"),
    )
    starting = make_workflow(tmp_path, "", "20260622_151500")
    (tmp_path / "campaign_a" / "matensemble_workflow-2026_01").mkdir()

    catalog = WorkflowCatalog(tmp_path)
    response = catalog.refresh()
    records = {item["relative_path"]: item for item in response["workflows"]}

    assert set(records) == {
        first.relative_to(tmp_path).as_posix(),
        second.relative_to(tmp_path).as_posix(),
        starting.relative_to(tmp_path).as_posix(),
    }
    assert records[first.relative_to(tmp_path).as_posix()]["parent_path"] == (
        "campaign_a/experiment"
    )
    assert records[starting.name]["campaign"] is None
    assert records[starting.name]["health"] == "starting"
    assert records[first.relative_to(tmp_path).as_posix()]["id"] != (
        records[second.relative_to(tmp_path).as_posix()]["id"]
    )


def test_discovery_stops_inside_workflows_and_skips_symlinks(tmp_path: Path):
    outer = make_workflow(
        tmp_path, "campaign", "20260622_143000", payload=status_payload()
    )
    make_workflow(
        outer, "artifacts", "20260622_151500", payload=status_payload()
    )
    external = tmp_path.parent / f"{tmp_path.name}-external"
    external.mkdir()
    try:
        make_workflow(
            external, "", "20260622_160000", payload=status_payload()
        )
        (tmp_path / "linked").symlink_to(external, target_is_directory=True)

        records = WorkflowCatalog(tmp_path).refresh()["workflows"]
        assert [item["relative_path"] for item in records] == [
            "campaign/matensemble_workflow-20260622_143000"
        ]
    finally:
        shutil.rmtree(external)


def test_legacy_normalization_unreadable_schema_and_stale_health(tmp_path: Path):
    legacy = make_workflow(tmp_path, "legacy", "20260622_100000")
    (legacy / "status.json").write_text(
        json.dumps({"pending": 3, "running": 1, "completed": 4, "failed": 0}),
        encoding="utf-8",
    )
    stale = make_workflow(
        tmp_path, "active", "20260622_110000",
        payload=status_payload(updated_ago=60),
    )
    terminal = make_workflow(
        tmp_path, "done", "20260622_120000",
        payload=status_payload(state="completed", updated_ago=60),
    )
    unsupported = make_workflow(tmp_path, "bad", "20260622_130000")
    (unsupported / "status.json").write_text(
        '{"schema_version":99}', encoding="utf-8"
    )

    response = WorkflowCatalog(tmp_path, stale_after=30).refresh()
    records = {item["relative_path"]: item for item in response["workflows"]}
    assert records[legacy.relative_to(tmp_path).as_posix()]["state"] == "running"
    assert records[stale.relative_to(tmp_path).as_posix()]["health"] == "stale"
    assert records[terminal.relative_to(tmp_path).as_posix()]["health"] == "healthy"
    assert records[unsupported.relative_to(tmp_path).as_posix()]["health"] == (
        "unreadable"
    )


def test_status_cache_avoids_reparsing_unchanged_file(monkeypatch, tmp_path: Path):
    workflow = make_workflow(
        tmp_path, "", "20260622_143000", payload=status_payload()
    )
    catalog = WorkflowCatalog(tmp_path)
    catalog.refresh()

    def fail_read(*_args, **_kwargs):
        raise AssertionError("unchanged status should come from the cache")

    monkeypatch.setattr(Path, "read_text", fail_read)
    catalog.refresh()
    assert catalog.catalog()["workflows"][0]["relative_path"] == workflow.name


def test_history_tolerates_incomplete_tail_and_downsamples(tmp_path: Path):
    workflow = make_workflow(
        tmp_path, "", "20260622_143000", payload=status_payload()
    )
    rows = [
        {"sequence": index, "timestamp": timestamp(), "pending": 10 - index}
        for index in range(10)
    ]
    text = "".join(json.dumps(row) + "\n" for row in rows) + '{"sequence":'
    (workflow / "status_history.jsonl").write_text(text, encoding="utf-8")

    response = read_history(
        workflow, status_payload(), after_sequence=2, max_points=4
    )
    assert response["ignored_incomplete_final_line"] is True
    assert response["truncated"] is True
    assert response["records"][0]["sequence"] == 3
    assert response["records"][-1]["sequence"] == 9


def test_api_catalog_status_history_and_new_workflow(tmp_path: Path):
    workflow = make_workflow(
        tmp_path, "campaign", "20260622_143000", payload=status_payload()
    )
    (workflow / "status_history.jsonl").write_text(
        json.dumps({"sequence": 2, "timestamp": timestamp(), "running": 1}) + "\n",
        encoding="utf-8",
    )
    app = create_dashboard_app(tmp_path, scan_interval=60)

    with TestClient(app) as client:
        catalog = client.get("/api/catalog").json()
        identifier = catalog["workflows"][0]["id"]
        assert client.get(f"/api/workflows/{identifier}/status").json()[
            "status"
        ]["schema_version"] == 2
        assert client.get(f"/api/workflows/{identifier}/history").json()[
            "last_sequence"
        ] == 2

        make_workflow(tmp_path, "campaign", "20260622_160000")
        app.state.catalog.refresh()
        assert len(client.get("/api/catalog").json()["workflows"]) == 2


def test_api_artifacts_are_contained_and_missing_workflow_is_retained(tmp_path: Path):
    workflow = make_workflow(
        tmp_path, "", "20260622_143000", payload=status_payload()
    )
    stderr = workflow / "out" / "chore-good" / "stderr"
    stderr.parent.mkdir(parents=True)
    stderr.write_text("failure details", encoding="utf-8")
    outside = tmp_path / "secret"
    outside.write_text("nope", encoding="utf-8")
    escaped = workflow / "out" / "chore-link"
    escaped.mkdir()
    (escaped / "stderr").symlink_to(outside)
    app = create_dashboard_app(tmp_path, scan_interval=60)

    with TestClient(app) as client:
        identifier = client.get("/api/catalog").json()["workflows"][0]["id"]
        assert client.get(
            f"/api/workflows/{identifier}/artifacts/chore-good/stderr"
        ).text == "failure details"
        assert client.get(
            f"/api/workflows/{identifier}/artifacts/../stderr"
        ).status_code in {400, 404}
        assert client.get(
            f"/api/workflows/{identifier}/artifacts/chore-link/stderr"
        ).status_code == 400

        for child in sorted(workflow.rglob("*"), reverse=True):
            if child.is_symlink() or child.is_file():
                child.unlink()
            else:
                child.rmdir()
        workflow.rmdir()
        app.state.catalog.refresh()
        selected = client.get(f"/api/workflows/{identifier}/status")
        assert selected.status_code == 200
        assert selected.json()["health"] == "missing"


def test_status_and_history_symlinks_cannot_escape_root(tmp_path: Path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside-status"
    outside.mkdir()
    try:
        external_status = outside / "status.json"
        external_status.write_text(json.dumps(status_payload()), encoding="utf-8")
        workflow = make_workflow(tmp_path, "", "20260622_143000")
        (workflow / "status.json").symlink_to(external_status)

        catalog = WorkflowCatalog(tmp_path)
        record = catalog.refresh()["workflows"][0]
        assert record["health"] == "unreadable"
        assert str(tmp_path) not in (record["error"] or "")

        (workflow / "status.json").unlink()
        payload = status_payload()
        (workflow / "status.json").write_text(json.dumps(payload), encoding="utf-8")
        external_history = outside / "history.jsonl"
        external_history.write_text('{"sequence":1}\n', encoding="utf-8")
        (workflow / "status_history.jsonl").symlink_to(external_history)
        with pytest.raises(ValueError, match="outside the workflow"):
            read_history(workflow, payload)
    finally:
        shutil.rmtree(outside)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions are required")
def test_scan_error_does_not_break_catalog(tmp_path: Path):
    make_workflow(tmp_path, "good", "20260622_143000", payload=status_payload())
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0)
    try:
        response = WorkflowCatalog(tmp_path).refresh()
        assert len(response["workflows"]) == 1
    finally:
        blocked.chmod(0o700)
