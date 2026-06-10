from __future__ import annotations

import re
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def slugify(value: str, *, fallback: str = "matensemble_campaign") -> str:
    """Return a filesystem-friendly name."""

    slug = _SLUG_RE.sub("_", value.strip()).strip("._-")
    return slug or fallback


def resolve_campaign_dir(
    output_dir: str | None,
    campaign_name: str,
    *,
    cwd: Path | None = None,
) -> Path:
    """Resolve a campaign path while keeping generated files under cwd."""

    root = (cwd or Path.cwd()).resolve()
    base = root if output_dir is None else (root / output_dir).resolve()

    try:
        base.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"output_dir must stay within the current workspace: {root}"
        ) from exc

    return base / slugify(campaign_name)


def workspace_root(cwd: Path | None = None) -> Path:
    return (cwd or Path.cwd()).resolve()


def resolve_within_workspace(path: str | Path, *, cwd: Path | None = None) -> Path:
    """Resolve *path* and reject paths outside the current workspace."""

    root = workspace_root(cwd)
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path must stay within the current workspace: {root}") from exc
    return resolved


def append_audit_event(event: dict[str, Any], *, cwd: Path | None = None) -> Path:
    """Append one audit event under the current workspace."""

    root = workspace_root(cwd)
    audit_dir = root / ".matensemble-mcp"
    audit_dir.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    audit_path = audit_dir / "audit.jsonl"
    with audit_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, sort_keys=True) + "\n")
    return audit_path
