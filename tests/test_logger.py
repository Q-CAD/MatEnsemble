import json
import logging

from pathlib import Path

from matensemble.logger import StatusWriter, _setup_logger, _setup_status_writer


def test_status_writer_update_writes_json(tmp_path: Path):
    writer = StatusWriter(tmp_path / "status.json", nnodes=2, cores_per_node=8, gpus_per_node=4)
    writer.update(pending=1, running=2, completed=3, failed=0, free_cores=4, free_gpus=2)
    payload = json.loads((tmp_path / "status.json").read_text())
    assert payload["nodes"] == 2
    assert payload["completed"] == 3


def test_setup_status_writer_returns_instance(tmp_path: Path):
    writer = _setup_status_writer(tmp_path / "status.json", 1, 2, 0)
    assert isinstance(writer, StatusWriter)


def test_setup_logger_creates_log_file(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("sys.stderr.isatty", lambda: False)
    logger = _setup_logger(tmp_path)
    assert isinstance(logger, logging.Logger)
    assert (tmp_path / "matensemble_workflow.log").exists()
