from __future__ import annotations

import re
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_CAMPAIGN_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

DANGEROUS_PATTERNS = (
    (re.compile(r"\brm\s+-rf\s+/"), "rm -rf /"),
    (re.compile(r"\bsudo\b"), "sudo"),
    (re.compile(r"\bcurl\b.+\|\s*bash"), "curl pipe to bash"),
    (re.compile(r"\bwget\b.+\|\s*bash"), "wget pipe to bash"),
    (re.compile(r"\bchmod\s+-R\s+777\s+/"), "chmod -R 777 /"),
    (re.compile(r">\s*~/\.bashrc"), "write ~/.bashrc"),
    (re.compile(r">>\s*~/\.bashrc"), "append ~/.bashrc"),
    (re.compile(r">\s*~/\.zshrc"), "write ~/.zshrc"),
    (re.compile(r">>\s*~/\.zshrc"), "append ~/.zshrc"),
    (re.compile(r"AWS_SECRET_ACCESS_KEY"), "AWS secret"),
    (re.compile(r"GITHUB_TOKEN"), "GitHub token"),
    (re.compile(r"ssh-rsa"), "SSH public key"),
    (re.compile(r"-----BEGIN PRIVATE KEY-----"), "private key"),
)


def slugify(value: str, *, fallback: str = "matensemble_campaign") -> str:
    """Return a filesystem-friendly name."""

    slug = _SLUG_RE.sub("_", value.strip()).strip("._-")
    return slug or fallback


def validate_campaign_name(value: str) -> str:
    name = value.strip()
    if not name:
        raise ValueError("campaign_name is required")
    if not _CAMPAIGN_NAME_RE.match(name):
        raise ValueError(f"campaign_name contains unsupported characters: {value}")
    candidate = Path(name)
    if candidate.is_absolute() or ".." in candidate.parts or "/" in name or "\\" in name:
        raise ValueError(f"campaign_name must be a simple directory name: {value}")
    return name


def scratch_workspace_root(*, env: dict[str, str] | None = None, cwd: Path | None = None) -> Path:
    """Return the allowed campaign workspace root.

    Production uses $SCRATCH/matensemble_campaigns. Tests may pass ``cwd`` to
    create an isolated scratch-like workspace without mutating the user env.
    """

    if cwd is not None:
        return cwd.resolve()
    source = env if env is not None else os.environ
    scratch = source.get("SCRATCH")
    if not scratch:
        raise RuntimeError("$SCRATCH is not set. MatEnsemble MCP requires an HPC scratch directory.")
    return (Path(scratch).expanduser() / "matensemble_campaigns").resolve()


def resolve_campaign_dir(
    output_dir: str | None,
    campaign_name: str,
    *,
    cwd: Path | None = None,
) -> Path:
    """Resolve a campaign path while keeping generated files under the campaign root."""

    root = scratch_workspace_root(cwd=cwd)
    base = root if output_dir is None else (root / output_dir).resolve()

    try:
        base.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"output_dir must stay within the MatEnsemble campaign workspace: {root}"
        ) from exc

    return base / slugify(campaign_name)


def workspace_root(cwd: Path | None = None) -> Path:
    return scratch_workspace_root(cwd=cwd)


def resolve_within_workspace(path: str | Path, *, cwd: Path | None = None) -> Path:
    """Resolve *path* and reject paths outside the MatEnsemble campaign workspace."""

    root = workspace_root(cwd)
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path must stay within the MatEnsemble campaign workspace: {root}") from exc
    return resolved


def relative_to_workspace(path: str | Path, *, cwd: Path | None = None) -> str:
    resolved = resolve_within_workspace(path, cwd=cwd)
    return str(resolved.relative_to(workspace_root(cwd)))


def scan_dangerous_text(text: str) -> list[str]:
    """Return descriptions of dangerous patterns found in generated script text."""

    return [label for pattern, label in DANGEROUS_PATTERNS if pattern.search(text)]


def ensure_safe_generated_text(text: str) -> None:
    matches = scan_dangerous_text(text)
    if matches:
        raise ValueError(f"generated script contains forbidden patterns: {', '.join(matches)}")


def append_audit_event(event: dict[str, Any], *, cwd: Path | None = None) -> Path:
    """Append one audit event under the campaign workspace."""

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
