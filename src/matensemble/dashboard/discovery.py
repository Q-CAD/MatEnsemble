from __future__ import annotations

import base64
import hashlib
import os
import re
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from matensemble.logger import TERMINAL_STATES, normalize_status_payload

from .models import WorkflowRecord


WORKFLOW_DIRECTORY_RE = re.compile(
    r"^matensemble_workflow-[0-9]{8}_[0-9]{6}$"
)
ACTIVE_STATES = {"initializing", "running"}
CURRENT_FIELDS = (
    "sequence",
    "pending",
    "ready",
    "blocked",
    "running",
    "completed",
    "failed",
    "free_cores",
    "free_gpus",
)


def workflow_id(relative_path: str) -> str:
    digest = hashlib.sha256(relative_path.encode("utf-8")).digest()[:18]
    encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"w_{encoded}"


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def validate_status(
    payload: dict[str, Any], *, status_path: Path | None = None
) -> dict[str, Any]:
    schema = payload.get("schema_version")
    if schema not in (None, 1, 2):
        raise ValueError(f"unsupported status schema version: {schema}")
    normalized = normalize_status_payload(payload, status_path=status_path)
    if normalized.get("schema_version") != 2:
        raise ValueError("status could not be normalized to schema version 2")

    workflow = normalized.get("workflow")
    allocation = normalized.get("allocation")
    current = normalized.get("current")
    if not isinstance(workflow, dict):
        raise ValueError("status.workflow must be an object")
    if not isinstance(allocation, dict):
        raise ValueError("status.allocation must be an object")
    if not isinstance(current, dict):
        raise ValueError("status.current must be an object")
    state = workflow.get("state")
    if state not in {"initializing", "running", *TERMINAL_STATES}:
        raise ValueError(f"unsupported workflow state: {state!r}")

    for key in CURRENT_FIELDS:
        try:
            current[key] = int(current.get(key, 0) or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"status.current.{key} must be an integer") from exc
        if current[key] < 0 and key != "sequence":
            raise ValueError(f"status.current.{key} must not be negative")
    for key in (
        "nodes",
        "cores_per_node",
        "gpus_per_node",
        "total_cores",
        "total_gpus",
    ):
        try:
            allocation[key] = int(allocation.get(key, 0) or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"status.allocation.{key} must be an integer") from exc

    failures = normalized.get("failures", [])
    if not isinstance(failures, list):
        raise ValueError("status.failures must be an array")
    normalized["failures"] = failures
    return normalized


class WorkflowCatalog:
    """Thread-safe, metadata-cached view of workflows beneath one root."""

    def __init__(self, root: str | Path, *, stale_after: float = 30.0) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"dashboard root is not a directory: {self.root}")
        self.stale_after = max(0.0, float(stale_after))
        self._records: dict[str, WorkflowRecord] = {}
        self._status_cache: dict[
            Path, tuple[tuple[int, int, int], dict[str, Any] | None, str | None]
        ] = {}
        self._scan_errors: list[dict[str, str]] = []
        self._scanned_at: str | None = None
        self._lock = threading.RLock()

    @property
    def scanned_at(self) -> str | None:
        with self._lock:
            return self._scanned_at

    def refresh(self) -> dict[str, Any]:
        found_paths: list[Path] = []
        scan_errors: list[dict[str, str]] = []
        inaccessible_prefixes: list[str] = []

        def walk(directory: Path) -> None:
            try:
                with os.scandir(directory) as entries:
                    children = sorted(entries, key=lambda item: item.name)
                    for entry in children:
                        try:
                            if entry.is_symlink() or not entry.is_dir(
                                follow_symlinks=False
                            ):
                                continue
                        except OSError as exc:
                            record_scan_error(Path(entry.path), exc)
                            continue
                        child = Path(entry.path)
                        if WORKFLOW_DIRECTORY_RE.fullmatch(entry.name):
                            found_paths.append(child)
                        else:
                            walk(child)
            except OSError as exc:
                record_scan_error(directory, exc)

        def record_scan_error(path: Path, exc: OSError) -> None:
            relative = self._relative_display(path)
            inaccessible_prefixes.append(relative)
            message = exc.strerror or type(exc).__name__
            scan_errors.append({"path": relative, "message": message})

        walk(self.root)
        now = datetime.now(timezone.utc)
        records = {
            workflow_id(self._relative_display(path)): self._record_for(path, now)
            for path in found_paths
        }

        with self._lock:
            for identifier, old in self._records.items():
                if identifier in records:
                    continue
                if any(
                    old.relative_path == prefix
                    or old.relative_path.startswith(f"{prefix}/")
                    for prefix in inaccessible_prefixes
                ):
                    records[identifier] = old
                    continue
                records[identifier] = replace(
                    old,
                    health="missing",
                    error="The workflow directory is no longer available.",
                )
            self._records = records
            self._scan_errors = scan_errors
            self._scanned_at = (
                now.isoformat(timespec="seconds").replace("+00:00", "Z")
            )
            return self.catalog()

    def catalog(self) -> dict[str, Any]:
        with self._lock:
            records = sorted(self._records.values(), key=self._sort_key)
            return {
                "schema_version": 1,
                "scanned_at": self._scanned_at,
                "root_name": self.root.name,
                "workflows": [record.summary() for record in records],
                "scan_errors": list(self._scan_errors),
            }

    def get(self, identifier: str) -> WorkflowRecord | None:
        with self._lock:
            record = self._records.get(identifier)
        if record is None:
            return None
        if record.health != "missing" and not record.path.is_dir():
            return replace(
                record,
                health="missing",
                error="The workflow directory is no longer available.",
            )
        return record

    def status(self, identifier: str) -> WorkflowRecord | None:
        record = self.get(identifier)
        if record is None or record.health == "missing":
            return record
        return self._record_for(record.path, datetime.now(timezone.utc))

    def contains(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.root)
        except (OSError, ValueError):
            return False
        return True

    def _record_for(self, path: Path, now: datetime) -> WorkflowRecord:
        relative = self._relative_display(path)
        parent = Path(relative).parent.as_posix()
        parent = "" if parent == "." else parent
        campaign = relative.split("/", 1)[0] if "/" in relative else None
        status_path = path / "status.json"
        if not status_path.exists():
            return WorkflowRecord(
                id=workflow_id(relative),
                directory_name=path.name,
                relative_path=relative,
                parent_path=parent,
                campaign=campaign,
                path=path,
                display_name=path.name,
                state="initializing",
                health="starting",
                started_at=None,
                updated_at=None,
                current={key: 0 for key in CURRENT_FIELDS},
                error="Waiting for status.json to be created.",
            )
        try:
            resolved_status = status_path.resolve()
            resolved_status.relative_to(path.resolve())
            resolved_status.relative_to(self.root)
        except (OSError, ValueError):
            return WorkflowRecord(
                id=workflow_id(relative),
                directory_name=path.name,
                relative_path=relative,
                parent_path=parent,
                campaign=campaign,
                path=path,
                display_name=path.name,
                state=None,
                health="unreadable",
                started_at=None,
                updated_at=None,
                current={key: 0 for key in CURRENT_FIELDS},
                error="status.json resolves outside the workflow directory.",
            )

        status, error = self._read_cached_status(status_path)
        if error or status is None:
            return WorkflowRecord(
                id=workflow_id(relative),
                directory_name=path.name,
                relative_path=relative,
                parent_path=parent,
                campaign=campaign,
                path=path,
                display_name=path.name,
                state=None,
                health="unreadable",
                started_at=None,
                updated_at=None,
                current={key: 0 for key in CURRENT_FIELDS},
                error=error or "status.json is unreadable",
            )

        workflow = status["workflow"]
        updated = _parse_timestamp(workflow.get("updated_at"))
        health = "healthy"
        if (
            workflow.get("state") in ACTIVE_STATES
            and updated is not None
            and (now - updated).total_seconds() > self.stale_after
        ):
            health = "stale"
        return WorkflowRecord(
            id=workflow_id(relative),
            directory_name=path.name,
            relative_path=relative,
            parent_path=parent,
            campaign=campaign,
            path=path,
            display_name=workflow.get("name") or path.name,
            state=workflow.get("state"),
            health=health,
            started_at=workflow.get("started_at"),
            updated_at=workflow.get("updated_at"),
            current=status["current"],
            status=status,
        )

    def _read_cached_status(
        self, path: Path
    ) -> tuple[dict[str, Any] | None, str | None]:
        try:
            stat = path.stat()
            fingerprint = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        except OSError as exc:
            return None, (
                f"Could not inspect status.json: "
                f"{exc.strerror or type(exc).__name__}"
            )
        with self._lock:
            cached = self._status_cache.get(path)
        if cached and cached[0] == fingerprint:
            return cached[1], cached[2]
        try:
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("status.json must contain an object")
            status = validate_status(payload, status_path=path)
            error = None
        except OSError as exc:
            status = None
            error = (
                f"Could not read status.json: "
                f"{exc.strerror or type(exc).__name__}"
            )
        except (ValueError, TypeError) as exc:
            status = None
            error = f"Could not read status.json: {exc}"
        with self._lock:
            self._status_cache[path] = (fingerprint, status, error)
        return status, error

    def _relative_display(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.name

    @staticmethod
    def _sort_key(record: WorkflowRecord) -> tuple[Any, ...]:
        unhealthy = record.health != "healthy"
        active = record.state in ACTIVE_STATES
        priority = 0 if active or unhealthy else 1
        started = _parse_timestamp(record.started_at)
        timestamp = started.timestamp() if started else 0.0
        return (
            record.campaign or "",
            record.parent_path,
            priority,
            -timestamp,
            "".join(chr(255 - ord(char)) for char in record.directory_name),
        )
