from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class WorkflowRecord:
    id: str
    directory_name: str
    relative_path: str
    parent_path: str
    campaign: str | None
    path: Path
    display_name: str
    state: str | None
    health: str
    started_at: str | None
    updated_at: str | None
    current: dict[str, int]
    error: str | None = None
    status: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "directory_name": self.directory_name,
            "relative_path": self.relative_path,
            "parent_path": self.parent_path,
            "campaign": self.campaign,
            "display_name": self.display_name,
            "state": self.state,
            "health": self.health,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "current": {
                key: int(self.current.get(key, 0))
                for key in ("pending", "running", "completed", "failed")
            },
            "error": self.error,
        }
