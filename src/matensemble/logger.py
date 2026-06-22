import logging
import json
import sys
import os
import tempfile
import threading
import uuid

from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 2
TERMINAL_STATES = {"completed", "failed", "interrupted"}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def normalize_status_payload(
    payload: dict, *, status_path: Path | None = None
) -> dict:
    """Return a schema-v2 status payload, upgrading legacy flat payloads."""
    if payload.get("schema_version") == SCHEMA_VERSION:
        return payload

    path = Path(status_path) if status_path is not None else None
    timestamp = format_utc(
        datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        if path is not None and path.exists()
        else utc_now()
    )
    name = path.parent.name if path is not None else "workflow"
    campaign = path.parent.parent.name if path is not None else None

    def value(snake: str, camel: str | None = None) -> int:
        return int(payload.get(snake, payload.get(camel or snake, 0)) or 0)

    pending = value("pending")
    running = value("running")
    completed = value("completed")
    failed = value("failed")
    nodes = value("nodes")
    cores_per_node = value("cores_per_node", "coresPerNode")
    gpus_per_node = value("gpus_per_node", "gpusPerNode")
    state = payload.get("state") or "running"
    current = {
        "sequence": 0,
        "pending": pending,
        "ready": pending,
        "blocked": 0,
        "running": running,
        "completed": completed,
        "failed": failed,
        "free_cores": value("free_cores", "freeCores"),
        "free_gpus": value("free_gpus", "freeGpus"),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "source_schema_version": 1,
        "workflow": {
            "id": f"legacy:{name}",
            "name": name,
            "campaign": campaign,
            "state": state,
            "started_at": None,
            "updated_at": timestamp,
            "finished_at": timestamp if state in TERMINAL_STATES else None,
            "elapsed_seconds": None,
        },
        "allocation": {
            "nodes": nodes,
            "cores_per_node": cores_per_node,
            "gpus_per_node": gpus_per_node,
            "total_cores": nodes * cores_per_node,
            "total_gpus": nodes * gpus_per_node,
        },
        "current": current,
        "failures": [],
        "history_file": None,
    }


def read_status(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return normalize_status_payload(payload, status_path=path)


def read_status_history(status_path: Path, status: dict | None = None) -> list[dict]:
    status = read_status(status_path) if status is None else status
    history_name = status.get("history_file")
    if history_name:
        history_path = status_path.parent / history_name
        try:
            records = []
            for line in history_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
            return records
        except FileNotFoundError:
            pass

    current = status.get("current", {})
    workflow = status.get("workflow", {})
    if not current:
        return []
    return [
        {
            "sequence": current.get("sequence", 0),
            "timestamp": workflow.get("updated_at"),
            "elapsed_seconds": workflow.get("elapsed_seconds"),
            "state": workflow.get("state", "running"),
            **{
                key: current.get(key, 0)
                for key in (
                    "pending",
                    "ready",
                    "blocked",
                    "running",
                    "completed",
                    "failed",
                    "free_cores",
                    "free_gpus",
                )
            },
        }
    ]


class StatusWriter:
    """
    Class to handle updating the status file

    Attributes
    ----------
    path : Path
        the path to the status file
    nnodes : int
        The number of nodes that flux is managing (total_allocation - 1 for flux borker)
    cores_per_node : int
        The number of CPU cores that are available on each node
    gpus_per_node : int
        The number of GPUs that are available on each node

    """

    def __init__(
        self,
        path: Path,
        nnodes: int,
        cores_per_node: int,
        gpus_per_node: int,
        workflow_name: str | None = None,
        campaign: str | None = None,
    ) -> None:
        self.path = path
        self.history_path = path.with_name("status_history.jsonl")
        self.nnodes = nnodes
        self.cores_per_node = cores_per_node
        self.gpus_per_node = gpus_per_node
        self.workflow_id = str(uuid.uuid4())
        self.workflow_name = workflow_name or path.parent.name
        self.campaign = campaign if campaign is not None else path.parent.parent.name
        self.started_at = utc_now()
        self.sequence = -1
        self.current = {
            "sequence": -1,
            "pending": 0,
            "ready": 0,
            "blocked": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "free_cores": nnodes * cores_per_node,
            "free_gpus": nnodes * gpus_per_node,
        }
        self.failures: list[dict] = []
        self._failure_timestamps: dict[str, str] = {}
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text("", encoding="utf-8")
        self._write_summary(state="initializing", now=self.started_at)

    def update(
        self,
        pending: int,
        running: int,
        completed: int,
        failed: int,
        free_cores: int,
        free_gpus: int,
        ready: int | None = None,
        blocked: int | None = None,
        state: str | None = None,
        failures: list[dict] | None = None,
    ) -> None:
        if state is None:
            state = (
                "failed"
                if pending == 0 and running == 0 and failed > 0
                else "completed"
                if pending == 0 and running == 0
                else "running"
            )
        if state not in {"initializing", "running", *TERMINAL_STATES}:
            raise ValueError(f"unsupported workflow state: {state}")

        ready = pending if ready is None else ready
        blocked = 0 if blocked is None else blocked
        now = utc_now()
        with self._lock:
            self.sequence += 1
            self.current = {
                "sequence": self.sequence,
                "pending": pending,
                "ready": ready,
                "blocked": blocked,
                "running": running,
                "completed": completed,
                "failed": failed,
                "free_cores": free_cores,
                "free_gpus": free_gpus,
            }
            if failures is not None:
                self.failures = [
                    self._failure_for_dashboard(item, now) for item in failures
                ]

            record = {
                "sequence": self.sequence,
                "timestamp": format_utc(now),
                "elapsed_seconds": self._elapsed(now),
                "state": state,
                **{key: self.current[key] for key in self.current if key != "sequence"},
            }
            with self.history_path.open("a", encoding="utf-8") as history:
                history.write(json.dumps(record, separators=(",", ":")) + "\n")
                history.flush()
                os.fsync(history.fileno())

            self._write_summary(state=state, now=now)

    def _elapsed(self, now: datetime) -> float:
        return round((now - self.started_at).total_seconds(), 3)

    def _failure_for_dashboard(self, failure: dict, now: datetime) -> dict:
        chore_id = failure.get("chore_id")
        timestamp = failure.get("timestamp")
        if chore_id and not timestamp:
            timestamp = self._failure_timestamps.setdefault(chore_id, format_utc(now))
        return {
            "chore_id": chore_id,
            "timestamp": timestamp,
            "reason": failure.get("reason"),
            "upstream": failure.get("upstream"),
            "message": failure.get("message") or failure.get("exception"),
            "stderr": f"out/{chore_id}/stderr" if chore_id else None,
        }

    def _write_summary(self, *, state: str, now: datetime) -> None:
        data = {
            "schema_version": SCHEMA_VERSION,
            "workflow": {
                "id": self.workflow_id,
                "name": self.workflow_name,
                "campaign": self.campaign,
                "state": state,
                "started_at": format_utc(self.started_at),
                "updated_at": format_utc(now),
                "finished_at": format_utc(now) if state in TERMINAL_STATES else None,
                "elapsed_seconds": self._elapsed(now),
            },
            "allocation": {
                "nodes": self.nnodes,
                "cores_per_node": self.cores_per_node,
                "gpus_per_node": self.gpus_per_node,
                "total_cores": self.nnodes * self.cores_per_node,
                "total_gpus": self.nnodes * self.gpus_per_node,
            },
            "current": self.current,
            "failures": self.failures,
            "history_file": self.history_path.name,
        }

        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.path.parent, delete=False
        ) as tf:
            json.dump(data, tf, indent=2)
            tf.flush()
            os.fsync(tf.fileno())
            temp_name = tf.name

        os.replace(temp_name, self.path)


def _setup_status_writer(
    path: Path, nnodes: int, cores_per_node: int, gpus_per_node: int
):
    """
    Setup the status writer for the :obj:`FluxManager` to

    Parameters
    ----------
    path : Path
        The path to the status file
    nnodes : int
        The number of nodes that are on the allocation minus one for the Flux
        borker
    cores_per_node : int
        The number of CPU cores per node
    gpus_per_node : int
        The number of GPUs per nod
    """

    return StatusWriter(
        path=path,
        nnodes=nnodes,
        cores_per_node=cores_per_node,
        gpus_per_node=gpus_per_node,
    )


def _setup_logger(base_dir: Path) -> logging.Logger:
    """
    setup the status writer for the :obj:`FluxManager`
    """

    logger = logging.getLogger("matensemble")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Prevent duplicate handlers if setup is called twice
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_file = base_dir / f"matensemble_workflow.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    if sys.stderr.isatty():
        console_handler = logging.StreamHandler(stream=sys.stderr)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)

    hint = (
        f"Logs: {base_dir}/matensemble_workflow.log\n"
        f"Outputs: {base_dir}/out\n\n"
        f"Watch logs: watch tail -n 5 {base_dir}/matensemble_workflow.log"
    )
    print(hint, file=sys.stderr)

    logger.info(f"Workflow initialized at {base_dir}")
    return logger
