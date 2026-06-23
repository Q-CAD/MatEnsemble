# MatEnsemble MCP Server

This package provides a Model Context Protocol server for MatEnsemble.

It is intentionally conservative:

- exposes MatEnsemble API and example context to AI agents
- generates campaign directories containing `workflow.py`, `LAUNCH.md`, and `manifest.json`
- defaults execution-capable tools to dry-run mode unless `execute=true` is explicitly passed

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

- `get_examples`
- `get_container_build_info`
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
- `plan_matensemble_dashboard_access`
- `start_matensemble_dashboard`
- `get_matensemble_dashboard_status`
- `stop_matensemble_dashboard`
- `create_matensemble_campaign`

`run_matensemble_container_setup` and `start_matensemble_dashboard` are dry-run by
default. They return the exact command unless the caller explicitly passes
`execute=true`.

For dashboard viewing on HPC systems, run the dashboard on the login node against
the shared campaign directory and forward it from your laptop:

```bash
matensemble dashboard /path/to/matensemble_campaign --host 127.0.0.1 --port 8000
ssh -N -L 8000:127.0.0.1:8000 <user>@<login.host>
```

Then open `http://localhost:8000`.
