# mcp-matensemble

Small MCP server for MatEnsemble agents.

The server intentionally exposes only MatEnsemble context, deterministic
container tag/build guidance, example batch-script context, and dashboard
lifecycle helpers. It does not create campaigns, submit jobs, cancel jobs,
execute container setup, or add a separate safety/guardrail layer.

```bash
uv run --package mcp-matensemble mcp-matensemble --system frontier
```

Useful additions:

- `get_example_batch_scripts`: returns `submit.slurm` files from
  `example_workflows/<system>/<workflow>/submit.slurm`.
- `start_dashboard` prompt: asks the agent to start the dashboard in
  `matensemble_campaigns` and provide the SSH tunnel command for localhost.

Dashboard launches intentionally run through the source checkout with uv:

```bash
uv run --project /path/to/MatEnsemble matensemble dashboard /path/to/matensemble_campaigns --host 127.0.0.1 --port 8000
```

The MCP `launch_dashboard` result includes the exact `command`, `command_text`,
`cwd`, `project_root`, and `log_path`. If a launch exits immediately, verify
that `cwd` is the `matensemble_campaigns` directory and that the command starts
with `uv run --project <MatEnsemble checkout>`. The source checkout is used for
uv project resolution; the campaigns directory is used as the process working
directory.
