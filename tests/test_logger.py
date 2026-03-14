from __future__ import annotations

import json
import logging

from matensemble.logger import StatusWriter, _setup_logger, _setup_status_writer


def test_status_writer_updates_json_file(tmp_path):
    path = tmp_path / "status.json"
    writer = StatusWriter(path, nnodes=2, cores_per_node=56, gpus_per_node=8)
    writer.update(pending=5, running=4, completed=3, failed=2, free_cores=10, free_gpus=1)

    data = json.loads(path.read_text())
    assert data == {
        "nodes": 2,
        "coresPerNode": 56,
        "gpusPerNode": 8,
        "pending": 5,
        "running": 4,
        "completed": 3,
        "failed": 2,
        "freeCores": 10,
        "freeGpus": 1,
    }


def test_setup_status_writer_returns_status_writer(tmp_path):
    writer = _setup_status_writer(tmp_path / "status.json", 1, 4, 0)
    assert isinstance(writer, StatusWriter)


def test_setup_logger_clears_duplicate_handlers_and_writes_log(tmp_path, capsys):
    logger = logging.getLogger("matensemble")
    logger.handlers.clear()

    first = _setup_logger(tmp_path)
    first.info("first message")
    assert len(first.handlers) == 1

    second = _setup_logger(tmp_path)
    second.info("second message")
    assert len(second.handlers) == 1

    captured = capsys.readouterr()
    assert "Status file:" in captured.err

    log_text = (tmp_path / "matensemble_workflow.log").read_text()
    assert "Workflow initialized" in log_text
    assert "second message" in log_text
