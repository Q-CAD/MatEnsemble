import json
import logging
from datetime import datetime
from pathlib import Path

from matensemble.logger import (
    StatusWriter,
    _setup_logger,
    _setup_status_writer,
    normalize_status_payload,
    read_status_history,
)


def test_status_writer_update_writes_v2_json(tmp_path: Path):
    writer = StatusWriter(
        tmp_path / "status.json", nnodes=2, cores_per_node=8, gpus_per_node=4
    )
    writer.update(
        pending=1,
        ready=1,
        blocked=0,
        running=2,
        completed=3,
        failed=0,
        free_cores=4,
        free_gpus=2,
    )
    payload = json.loads((tmp_path / "status.json").read_text())
    assert payload["schema_version"] == 2
    assert payload["allocation"]["nodes"] == 2
    assert payload["current"]["completed"] == 3
    assert payload["workflow"]["state"] == "running"
    assert payload["history_file"] == "status_history.jsonl"


def test_status_history_is_append_only_and_matches_summary(tmp_path: Path):
    writer = StatusWriter(tmp_path / "status.json", 2, 8, 4)
    for completed in range(4):
        writer.update(
            pending=3 - completed,
            ready=2,
            blocked=max(0, 1 - completed),
            running=1,
            completed=completed,
            failed=0,
            free_cores=8,
            free_gpus=4,
        )

    history = read_status_history(tmp_path / "status.json")
    summary = json.loads((tmp_path / "status.json").read_text())
    assert [record["sequence"] for record in history] == [0, 1, 2, 3]
    assert summary["current"]["sequence"] == history[-1]["sequence"]
    assert summary["current"]["completed"] == history[-1]["completed"]
    assert all(record["timestamp"].endswith("Z") for record in history)
    assert all(
        datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00")).tzinfo
        for record in history
    )


def test_terminal_update_records_failure_details(tmp_path: Path):
    writer = StatusWriter(tmp_path / "status.json", 1, 8, 0)
    writer.update(
        pending=0,
        ready=0,
        blocked=0,
        running=0,
        completed=2,
        failed=1,
        free_cores=8,
        free_gpus=0,
        state="failed",
        failures=[
            {
                "chore_id": "chore-bad-0003",
                "timestamp": "2026-06-22T14:36:20Z",
                "reason": "dependency_failed",
                "upstream": "chore-root-0001",
                "message": None,
            }
        ],
    )

    payload = json.loads((tmp_path / "status.json").read_text())
    assert payload["workflow"]["finished_at"] is not None
    assert payload["failures"][0]["upstream"] == "chore-root-0001"
    assert payload["failures"][0]["stderr"] == "out/chore-bad-0003/stderr"
    assert read_status_history(tmp_path / "status.json")[-1]["state"] == "failed"


def test_terminal_state_is_inferred_from_existing_manager_counters(tmp_path: Path):
    completed_writer = StatusWriter(tmp_path / "completed" / "status.json", 1, 8, 0)
    completed_writer.update(0, 0, 3, 0, 8, 0)
    completed = json.loads(completed_writer.path.read_text())

    failed_writer = StatusWriter(tmp_path / "failed" / "status.json", 1, 8, 0)
    failed_writer.update(0, 0, 2, 1, 8, 0)
    failed = json.loads(failed_writer.path.read_text())

    assert completed["workflow"]["state"] == "completed"
    assert failed["workflow"]["state"] == "failed"


def test_new_writer_replaces_identity_and_old_history(tmp_path: Path):
    status_path = tmp_path / "status.json"
    first = StatusWriter(status_path, 1, 8, 0)
    first.update(1, 0, 0, 0, 8, 0)
    first_id = json.loads(status_path.read_text())["workflow"]["id"]

    second = StatusWriter(status_path, 1, 8, 0)
    second_id = json.loads(status_path.read_text())["workflow"]["id"]

    assert first_id != second_id
    assert read_status_history(status_path) == []


def test_legacy_status_is_upgraded_to_one_snapshot(tmp_path: Path):
    status_path = tmp_path / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "nodes": 2,
                "cores_per_node": 8,
                "gpus_per_node": 1,
                "pending": 3,
                "running": 1,
                "completed": 4,
                "failed": 0,
                "free_cores": 8,
                "free_gpus": 1,
                "state": "running",
            }
        )
    )
    normalized = normalize_status_payload(
        json.loads(status_path.read_text()), status_path=status_path
    )
    history = read_status_history(status_path, normalized)
    assert normalized["schema_version"] == 2
    assert normalized["source_schema_version"] == 1
    assert normalized["current"]["pending"] == 3
    assert len(history) == 1
    assert history[0]["running"] == 1


def test_setup_status_writer_returns_instance(tmp_path: Path):
    writer = _setup_status_writer(tmp_path / "status.json", 1, 2, 0)
    assert isinstance(writer, StatusWriter)


def test_setup_logger_creates_log_file(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("sys.stderr.isatty", lambda: False)
    logger = _setup_logger(tmp_path)
    assert isinstance(logger, logging.Logger)
    assert (tmp_path / "matensemble_workflow.log").exists()
