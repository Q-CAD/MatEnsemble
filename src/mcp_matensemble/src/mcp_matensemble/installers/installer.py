from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import tomllib
from pathlib import Path

from mcp_matensemble.systems import normalize_system


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Install a MatEnsemble MCP agent workspace."
    )
    parser.add_argument(
        "--system",
        required=True,
        help="frontier, perlmutter, pathfinder, or linux",
    )
    parser.add_argument(
        "--workspace", default=None, help="Campaign workspace to create"
    )
    parser.add_argument("--install-dir", default=str(Path.home() / ".local" / "bin"))
    parser.add_argument(
        "--install-site-cli",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--confirm-cli-install",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--config-dir",
        default=str(
            Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
            / "matensemble"
        ),
    )
    parser.add_argument(
        "--codex-config",
        default=str(Path.home() / ".codex" / "config.toml"),
        help="Codex config.toml to update with the MatEnsemble MCP server",
    )
    parser.add_argument(
        "--skip-codex-config",
        action="store_true",
        help="Do not update the Codex config.toml MCP server entry",
    )
    ns = parser.parse_args(argv)

    system = normalize_system(ns.system)
    workspace = Path(ns.workspace or _default_workspace()).expanduser().resolve()
    scratch_workspace = Path(_default_workspace()).expanduser().resolve()
    scratch_root = scratch_workspace.parent
    try:
        workspace.relative_to(scratch_workspace)
    except ValueError as exc:
        raise SystemExit(
            "workspace must be inside $SCRATCH/matensemble_campaigns: "
            f"{scratch_workspace}"
        ) from exc
    install_dir = Path(ns.install_dir).expanduser().resolve()
    config_dir = Path(ns.config_dir).expanduser().resolve()
    codex_config = Path(ns.codex_config).expanduser().resolve()

    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / ".vscode").mkdir(parents=True, exist_ok=True)
    install_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    wrapper = install_dir / f"mcp-matensemble-{system}"
    wrapper_changed = _write_executable(
        wrapper,
        _server_wrapper_script(install_dir=install_dir),
    )

    site_cli = None
    site_cli_changed = False
    if system in {"frontier", "perlmutter", "pathfinder"}:
        site_cli = install_dir / "matensemble"
        site_cli_changed = _install_repo_site_cli(system=system, target=site_cli)

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
    codex_config_changed = False
    if not ns.skip_codex_config:
        codex_config_changed = _write_codex_config(
            codex_config,
            wrapper=wrapper,
            workspace=workspace,
            scratch=scratch_root,
        )
    _write_readme(workspace, system)

    print(f"Installed MatEnsemble MCP workspace for {system}")
    print(f"Workspace: {workspace}")
    print(f"VS Code MCP config: {workspace / '.vscode' / 'mcp.json'}")
    if not ns.skip_codex_config:
        state = "updated" if codex_config_changed else "already current"
        print(f"Codex MCP config: {codex_config} ({state})")
    print(f"Wrapper: {wrapper} ({'updated' if wrapper_changed else 'already current'})")
    if site_cli is not None:
        print(
            f"MatEnsemble site CLI: {site_cli} "
            f"({'updated' if site_cli_changed else 'already current'})"
        )


def _default_workspace() -> str:
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        raise SystemExit(
            "$SCRATCH is not set. MatEnsemble MCP requires an HPC scratch directory."
        )
    return str(Path(scratch) / "matensemble_campaigns")


def _write_executable(path: Path, text: str) -> bool:
    changed = not path.exists() or path.read_text(encoding="utf-8") != text
    if not changed:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return False
    path.write_text(text, encoding="utf-8")
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return True


def _server_wrapper_script(*, install_dir: Path) -> str:
    repo = _repo_root()
    uv = shutil.which("uv") or "uv"
    return f"""#!/usr/bin/env bash
set -euo pipefail
export PATH="{install_dir}:$PATH"
cd "{repo}"
exec "{uv}" run mcp-matensemble
"""


def _write_codex_config(
    path: Path,
    *,
    wrapper: Path,
    workspace: Path,
    scratch: Path,
) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    block = _codex_mcp_server_block(
        wrapper=wrapper,
        workspace=workspace,
        scratch=scratch,
    )
    updated = _replace_toml_table(
        original,
        "mcp_servers.matensemble",
        block,
    )
    tomllib.loads(updated)
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    path.chmod(0o600)
    return True


def _codex_mcp_server_block(
    *,
    wrapper: Path,
    workspace: Path,
    scratch: Path,
) -> str:
    return "\n".join(
        [
            "[mcp_servers.matensemble]",
            f"command = {_toml_string(str(wrapper))}",
            f"cwd = {_toml_string(str(workspace))}",
            "startup_timeout_sec = 120",
            "",
            "[mcp_servers.matensemble.env]",
            f"SCRATCH = {_toml_string(str(scratch))}",
            "",
        ]
    )


def _replace_toml_table(text: str, table: str, block: str) -> str:
    lines = text.splitlines()
    start = None
    end = None
    for index, line in enumerate(lines):
        name = _toml_table_name(line)
        if name == table:
            start = index
            end = len(lines)
            for next_index in range(index + 1, len(lines)):
                next_name = _toml_table_name(lines[next_index])
                if next_name is None:
                    continue
                if next_name == table or next_name.startswith(f"{table}."):
                    continue
                end = next_index
                break
            break

    block_lines = block.rstrip("\n").splitlines()
    if start is None:
        prefix = lines + ([""] if lines and lines[-1].strip() else [])
        updated_lines = prefix + block_lines
    else:
        updated_lines = lines[:start] + block_lines + lines[end:]

    return "\n".join(updated_lines).rstrip() + "\n"


def _toml_table_name(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("[") or stripped.startswith("[["):
        return None
    if "]" not in stripped:
        return None
    closing = stripped.find("]")
    rest = stripped[closing + 1 :].strip()
    if rest and not rest.startswith("#"):
        return None
    return stripped[1:closing].strip()


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (
            parent / "src" / "mcp_matensemble"
        ).is_dir():
            return parent
    raise SystemExit("could not locate the MatEnsemble MCP repository root")


def _install_repo_site_cli(*, system: str, target: Path) -> bool:
    source = _repo_site_cli_path(system)
    return _write_executable(target, source.read_text(encoding="utf-8"))


def _repo_site_cli_path(system: str) -> Path:
    filename = f"matensemble-{system}"
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "src" / "cli" / filename
        if candidate.is_file():
            return candidate
    raise SystemExit(
        f"could not locate stable MatEnsemble CLI script: src/cli/{filename}"
    )


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

Use prepare/confirm tools for launch, build, pull, cancel, and delete actions.
The server will not execute those actions without a matching plan id.

For MatEnsemble containers, ask the agent to use `prepare_container_pull_plan`.
The MCP server forms GHCR image tags from the local MatEnsemble version, for
example `ghcr.io/freddude2004/matensemble:{system}-vX.Y.Z`; agents should not
query registry tag APIs.

For HPC systems, generated Slurm resources use at least 2 nodes because Flux
uses one node as a broker/orchestrator and leaves the remaining node(s) for
MatEnsemble chores.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
