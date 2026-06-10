from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path

from mcp_matensemble.installers.site_cli import site_cli_script
from mcp_matensemble.systems import normalize_system


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Install a MatEnsemble MCP agent workspace.")
    parser.add_argument("--system", required=True, help="frontier, perlmutter, pathfinder, linux, or conda")
    parser.add_argument("--workspace", default=None, help="Campaign workspace to create")
    parser.add_argument("--install-dir", default=str(Path.home() / ".local" / "bin"))
    parser.add_argument("--config-dir", default=str(Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "matensemble"))
    ns = parser.parse_args(argv)

    system = normalize_system(ns.system)
    workspace = Path(ns.workspace or _default_workspace()).expanduser().resolve()
    install_dir = Path(ns.install_dir).expanduser().resolve()
    config_dir = Path(ns.config_dir).expanduser().resolve()

    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / ".vscode").mkdir(parents=True, exist_ok=True)
    install_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    wrapper = install_dir / f"mcp-matensemble-{system}"
    _write_executable(
        wrapper,
        f"""#!/usr/bin/env bash
set -euo pipefail
export PATH="{install_dir}:$PATH"
exec "{install_dir / "mcp-matensemble"}"
""",
    )

    if system in {"frontier", "perlmutter", "pathfinder"}:
        _write_executable(install_dir / "matensemble", site_cli_script(system))

    mcp_config = {
        "servers": {
            "matensemble": {
                "type": "stdio",
                "command": str(wrapper),
                "cwd": "${workspaceFolder}",
            }
        }
    }
    (workspace / ".vscode" / "mcp.json").write_text(
        json.dumps(mcp_config, indent=2) + "\n",
        encoding="utf-8",
    )
    (workspace / ".matensemble-mcp.toml").write_text(
        f'system = "{system}"\nworkspace = "{workspace}"\n',
        encoding="utf-8",
    )
    (config_dir / "system").write_text(system + "\n", encoding="utf-8")
    _write_readme(workspace, system)

    print(f"Installed MatEnsemble MCP workspace for {system}")
    print(f"Workspace: {workspace}")
    print(f"VS Code MCP config: {workspace / '.vscode' / 'mcp.json'}")
    print(f"Wrapper: {wrapper}")


def _default_workspace() -> str:
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return str(Path(scratch) / "matensemble_campaigns")
    return str(Path.home() / "matensemble_campaigns")


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _write_readme(workspace: Path, system: str) -> None:
    workspace.joinpath("README.md").write_text(
        f"""# MatEnsemble Campaign Workspace

This folder is configured for the MatEnsemble MCP server on `{system}`.

Open this directory with VS Code Remote, start the `matensemble` MCP server,
and ask the agent to create workflows here.

Try:

```text
I want to run a LAMMPS GPU smoke test with MatEnsemble on {system}. Use the MatEnsemble MCP server to inspect examples, plan setup, and create workflow and launch scripts. Do not execute setup commands or submit jobs yet.
```

Scheduler and setup tools are dry-run by default. Use `execute=true` only after
reviewing the planned command.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
