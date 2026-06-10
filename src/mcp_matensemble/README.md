# MatEnsemble MCP Server

This package provides a Model Context Protocol server for MatEnsemble.

It is intentionally conservative:

- exposes MatEnsemble API and example context to AI agents
- generates campaign directories containing `workflow.py`, `LAUNCH.md`, and `manifest.json`
- does not execute generated scripts or submit jobs

Run locally with:

```bash
uv run --package mcp-matensemble mcp-matensemble
```

The primary tool is `create_campaign`.

For development from any arbitrary workspace, point an MCP client at this package
with an editable install while keeping the server working directory set to the
research project folder:

```json
{
  "servers": {
    "matensemble": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "run",
        "--with-editable",
        "/path/to/MatEnsemble/src/mcp_matensemble",
        "mcp-matensemble"
      ],
      "cwd": "${workspaceFolder}"
    }
  }
}
```

Useful tools:

- `get_api_overview`
- `list_matensemble_examples`
- `get_matensemble_example`
- `list_matensemble_systems`
- `get_matensemble_system`
- `get_matensemble_environment_setup`
- `get_matensemble_container_contents`
- `get_matensemble_container_install`
- `get_matensemble_version_info`
- `plan_matensemble_container_setup`
- `run_matensemble_container_setup`
- `create_matensemble_campaign`

`run_matensemble_container_setup` is dry-run by default. It returns the exact
allowlisted command unless the caller explicitly passes `execute=true`.
