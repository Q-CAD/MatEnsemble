from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check MatEnsemble MCP agent installation.")
    parser.add_argument("--workspace", default=".")
    ns = parser.parse_args(argv)

    workspace = Path(ns.workspace).resolve()
    checks = {
        "workspace_exists": workspace.is_dir(),
        "mcp_json_exists": (workspace / ".vscode" / "mcp.json").is_file(),
        "mcp_command_on_path": shutil.which("mcp-matensemble") is not None,
        "site_cli_on_path": shutil.which("matensemble") is not None,
    }
    for name, ok in checks.items():
        print(f"{name}: {'ok' if ok else 'missing'}")
    if not all(checks.values()):
        raise SystemExit(1)
