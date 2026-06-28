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
